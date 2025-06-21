import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import sys
sys.path.append('/home/keli/Decompose_Retrieval/raptor')
import psutil

import cv2
from transformers import CLIPModel, AutoProcessor
import torch
from image_utils import *
import argparse
from sklearn.metrics import top_k_accuracy_score
from beir.retrieval.evaluation import EvaluateRetrieval
from retrieval_utils import *
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.dense.exact_search import one, two, three, four
from beir.retrieval import models
from beir import LoggingHandler
from datasets import load_dataset
from datasets.download.download_config import DownloadConfig
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
import time
# from beir.retrieval.models.clip_model import clip_model
from clustering import *
from text_utils import *
import copy
import os, shutil
from bbox_utils import *
from utils import *
from sparse_index import *
from baselines.llm_ranker import *
from baselines.bm25 import *
from derive_sub_query_dependencies import group_dependent_segments_seq_all
import random
from dessert_minheap_torch import *
import pynvml
from LLM4split.prompt_utils import *
from raptor.raptor_embeddings import *

import os
import base64
import torch
import json
from tqdm import tqdm
import numpy as np
from vllm import LLM, SamplingParams
import os
import PIL
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"


image_retrieval_datasets = ["flickr", "AToMiC", "crepe", "crepe_full", "mscoco", "mscoco_40k", "multiqa" ,"multiqa_few", "art", 'manyqa','webqa']
text_retrieval_datasets = ["trec-covid", "nq", "climate-fever", "hotpotqa", "msmarco", "webis-touche2020", "scifact", "fiqa"]

sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=512)
llm = LLM(
    model="/home/icml01/Models/llava",
    # tensor_parallel_size=4,
    max_model_len=2048,
    max_num_seqs=512,
)
    
def embed_queries(filename_ls, filename_cap_mappings, processor, model, device):
    text_emb_ls = []
    with torch.no_grad():
        # for filename, caption in tqdm(filename_cap_mappings.items()):
        for file_name in filename_ls:
            caption = filename_cap_mappings[file_name]
            inputs = processor(caption)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            text_features = model.get_text_features(**inputs)
            # text_features = outputs.last_hidden_state[:, 0, :]
            text_emb_ls.append(text_features.cpu())
    return text_emb_ls

def embed_queries_with_input_queries(model_name, query_ls, processor, model, device):
    text_emb_ls = []
    with torch.no_grad():
        # for filename, caption in tqdm(filename_cap_mappings.items()):
        for caption in query_ls:
            # caption = filename_cap_mappings[file_name]
            inputs = processor(caption)
            if model_name == "default":
                inputs = {key: val.to(device) for key, val in inputs.items()}
                text_features = model.get_text_features(**inputs)
            elif model_name == "blip":
                text_features = model.extract_features({"text_input":inputs}, mode="text").text_embeds_proj[:,0,:].view(1,-1)
            else:
                raise ValueError("Invalid model name")
            # text_features = outputs.last_hidden_state[:, 0, :]
            text_emb_ls.append(text_features.cpu())
    return text_emb_ls

def embed_queries_ls(model_name, full_sub_queries_ls, processor, model, device):
    text_emb_ls = []
    with torch.no_grad():
        # for filename, caption in tqdm(filename_cap_mappings.items()):
        # for file_name in filename_ls:
        for sub_queries_ls in tqdm(full_sub_queries_ls):
            sub_text_emb_ls = []
            for sub_queries in sub_queries_ls:
                sub_text_feature_ls = []
                for subquery in sub_queries:
                    # caption = filename_cap_mappings[file_name]
                    inputs = processor(subquery)
                    if model_name == "default":
                        inputs = {key: val.to(device) for key, val in inputs.items()}
                        text_features = model.get_text_features(**inputs)
                    elif model_name == "blip":
                        text_features = model.extract_features({"text_input":inputs}, mode="text").text_embeds_proj[:,0,:].view(1,-1)
                    else:
                        raise ValueError("Invalid model name")
                    sub_text_feature_ls.append(text_features.cpu())
                # text_features = outputs.last_hidden_state[:, 0, :]
                sub_text_emb_ls.append(sub_text_feature_ls)
            text_emb_ls.append(sub_text_emb_ls)
    return text_emb_ls

        
def retrieve_by_full_query(img_emb, text_emb_ls):
    text_emb_tensor = torch.cat(text_emb_ls).cpu()
    scores = (img_emb @ text_emb_tensor.T).squeeze()/ (img_emb.norm(dim=-1) * text_emb_tensor.norm(dim=-1))
    true_rank = torch.tensor([i for i in range(len(text_emb_ls))])
    top_k_acc = top_k_accuracy_score(true_rank, scores, k=1)
    
    print(f"Top-k accuracy: {top_k_acc:.2f}")
    
    print()
    
    # for idx in range(len(text_emb_ls)):
    #     text_emb = text_emb_ls[idx]
    #     scores = (img_emb @ text_emb.T).squeeze()/ (img_emb.norm(dim=-1) * text_emb.norm(dim=-1))
        
        # print(scores)
        # print(scores.shape)
        # print(scores.argmax())
        # print()
    
def set_rand_seed(seed_value):
    # Set seed for Python's built-in random module
    random.seed(seed_value)
    
    # Set seed for NumPy
    np.random.seed(seed_value)
    
    # Set seed for PyTorch
    torch.manual_seed(seed_value)
    
    # Set seed for CUDA (if using a GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # If using multi-GPU.
    
    # Ensure deterministic operations for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='CUB concept learning')
    parser.add_argument('--data_path', type=str, default="/data6/wuyinjun/", help='config file')
    parser.add_argument('--dataset_name', type=str, default="crepe", help='config file')
    parser.add_argument('--model_name', type=str, default="default", help='config file')
    parser.add_argument('--query_count', type=int, default=-1, help='config file')
    parser.add_argument('--random_seed', type=int, default=0, help='config file')
    parser.add_argument('--query_concept', action="store_true", help='config file')
    parser.add_argument('--img_concept', action="store_true", help='config file')
    parser.add_argument('--total_count', type=int, default=-1, help='config file')
    parser.add_argument("--parallel", action="store_true", help="config file")
    parser.add_argument("--save_mask_bbox", action="store_true", help="config file")
    parser.add_argument("--search_by_cluster", action="store_true", help="config file")
    parser.add_argument('--algebra_method', type=str, default=one, help='config file')
    # closeness_threshold
    parser.add_argument('--closeness_threshold', type=float, default=0.1, help='config file')
    parser.add_argument('--subset_img_id', type=int, default=None, help='config file')
    parser.add_argument('--prob_agg', type=str, default="prod", choices=["prod", "sum"], help='config file')
    parser.add_argument('--segmentation_method', type=str, default="default", choices=["default", "scene_graph"], help='config file')
    parser.add_argument('--dependency_topk', type=int, default=20, help='config file')
    parser.add_argument('--clustering_topk', type=int, default=500, help='config file')
    parser.add_argument("--add_sparse_index", action="store_true", help="config file")
    
    parser.add_argument('--retrieval_method', type=str, default="ours", help='config file')
    parser.add_argument('--index_method', type=str, default="default", choices=["default", "dessert", "dessert0"], help='config file')
    parser.add_argument('--hashes_per_table', type=int, default=5, help='config file')
    # num_tables
    parser.add_argument('--num_tables', type=int, default=100, help='config file')
    parser.add_argument('--clustering_doc_count_factor', type=int, default=1, help='config file')
    parser.add_argument('--clustering_number', type=float, default=0.1, help='config file')
    # 
    parser.add_argument('--nprobe_query', type=int, default=2, help='config file')
    parser.add_argument('--subset_patch_count', type=int, default=-1, help='config file')
    parser.add_argument('--cached_file_suffix', type=str, default="", help='config file')
    parser.add_argument("--is_test", action="store_true", help="config file")
    parser.add_argument("--store_res", action="store_true", help="config file")
    parser.add_argument("--use_phi", action="store_true", help="config file")
    parser.add_argument('--use_raptor', action="store_true", help='config file')
  
    args = parser.parse_args(args=['--dataset_name', 'manyqa',
                                   '--data_path', '/home/icml01/images/',
                                   '--query_concept', '--img_concept', '--algebra_method', 'two'])
    return args


def obtain_memory_usage():
    process = psutil.Process(os.getpid())
    # print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
    memory_usage = process.memory_info().rss / 1024 ** 3
    return memory_usage


def obtain_gpu_memory_usage():
    # pynvml.nvmlInit()
    current_pid = os.getpid()
    
    used_gpu_memory = -1
    pynvml.nvmlInit()
    for dev_id in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if proc.pid == current_pid:
                used_gpu_memory = proc.usedGpuMemory
            
    #         print(
    #             "pid %d using %d bytes of memory on device %d."
    #             % (proc.pid, proc.usedGpuMemory, dev_id)
    #         )
    
    # handle = pynvml.nvmlDeviceGetHandleByIndex(3)

    # # Get the list of processes using the GPU
    # processes = pynvml.nvmlDeviceGetGraphicsRunningProcesses_v2(handle)
    
    # for process in processes:
    #     if process.pid == current_pid:
    #         used_memory = process.usedGpuMemory / (1024 ** 2)  # Convert bytes to MB

    # process = psutil.Process(os.getpid())
    # # print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
    # memory_usage = process.memory_info().rss / 1024 ** 2
    return used_gpu_memory / 1024 ** 3


def construct_qrels(dataset_name, queries, cached_img_idx, img_idx_ls, query_count):
    qrels = {}
    # if query_count < 0:
    #     query_count = 
    
    for idx in range(len(queries)):
        curr_img_idx = img_idx_ls[idx]
        cached_idx = cached_img_idx.index(curr_img_idx)
        qrels[str(idx+1)] = {str(cached_idx+1): 2}
    q_idx_ls = list(range(len(queries)))
    if query_count > 0:
        if dataset_name == "crepe":
            subset_q_idx_ls = [q_idx_ls[idx] for idx in range(query_count)] 
        else:
            subset_q_idx_ls = random.sample(q_idx_ls, query_count)
        
        subset_q_idx_ls = sorted(subset_q_idx_ls)
        
        subset_qrels = {str(key_id + 1): qrels[str(subset_q_idx_ls[key_id] + 1)] for key_id in range(len(subset_q_idx_ls))}
    
        qrels = subset_qrels
        
        queries = [queries[idx] for idx in subset_q_idx_ls]
    else:
        subset_q_idx_ls = q_idx_ls #list(qrels.keys())
        
    return qrels, queries, subset_q_idx_ls



logging.basicConfig(format='%(asctime)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                level=logging.INFO,
                handlers=[LoggingHandler()])
used_memory0 = psutil.virtual_memory().used

args = parse_args()


args.is_img_retrieval = args.dataset_name in image_retrieval_datasets
set_rand_seed(args.random_seed)
query_path = os.path.dirname(os.path.realpath(__file__)) 

# args.query_concept = False
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" )


# processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224')
# model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224').to(device)
if args.is_img_retrieval:
    if args.model_name == "default":
       
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        # processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        raw_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        # processor =  lambda images: raw_processor(images=images, return_tensors="pt", padding=False, do_resize=False, do_center_crop=False)["pixel_values"]
        # processor = lambda images: safe_raw_processor(images)
        
        processor =  lambda images: raw_processor(images=images, return_tensors="pt")["pixel_values"]

        text_processor =  lambda text: raw_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        img_processor =  lambda images: raw_processor(images=images, return_tensors="pt")["pixel_values"]
        model = model.eval()
        
    elif args.model_name == "blip":
        print("start loading blip model")
        if os.path.exists("output/blip.pkl"):
            model,vis_processors_eval, txt_processors_eval = utils.load("output/blip.pkl")
        else:
            from lavis.models import load_model_and_preprocess
            model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
            utils.save((model,vis_processors["eval"], txt_processors["eval"]), "output/blip.pkl")
            txt_processors_eval = txt_processors["eval"]
            vis_processors_eval = vis_processors["eval"]
        text_processor = lambda text: txt_processors_eval(text)
        processor =  lambda images: torch.stack([vis_processors_eval(image) for image in images])
    
    model = model.eval()
    
    
    if args.add_sparse_index:
        # text_model = models.SentenceBERT("msmarco-distilbert-base-tas-b", prefix = sparse_prefix, suffix=sparse_suffix)
        # if args.model_name == "default":
        print("start loading distill-bert model")
        if not os.path.exists("output/msmarco-distilbert-base-tas-b.pkl"):
            text_model = models.SentenceBERT("msmarco-distilbert-base-tas-b", prefix = sparse_prefix, suffix=sparse_suffix) # offline model
            utils.save(text_model, "output/msmarco-distilbert-base-tas-b.pkl")
        else:
            text_model = utils.load("output/msmarco-distilbert-base-tas-b.pkl")
    
else:
    # if args.dataset_name not in image_retrieval_datasets:
    # if not args.is_img_retrieval:
    # text_model = models.clip_model(text_processor, model, device)
    if args.model_name == "default":
        print("start loading distill-bert model")
        if True: #not os.path.exists("output/msmarco-distilbert-base-tas-b.pkl"):
            text_model = models.SentenceBERT("msmarco-distilbert-base-tas-b", prefix = sparse_prefix, suffix=sparse_suffix) # offline model
            utils.save(text_model, "output/msmarco-distilbert-base-tas-b.pkl")
        # else:
        #     text_model = utils.load("output/msmarco-distilbert-base-tas-b.pkl")
    # elif args.model_name == "phi":
    #     text_model = models.ms_phi(prefix=sparse_prefix, suffix=sparse_suffix)
    elif args.model_name == "llm":
        # text_model = models.LlmtoVec(("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp","McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"),
        #     ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised","McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"))
        text_model = models.LlmtoVec(("McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp"),
            ("McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised", "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised"))
    else:
        raise ValueError("Invalid model name")
    # text_model = AutoModelForCausalLM.from_pretrained(
    #             "microsoft/Phi-3-mini-4k-instruct", 
    #             device_map="cuda", 
    #             torch_dtype="auto", 
    #             trust_remote_code=True, 
    #         )
    text_retrieval_model = DRES(batch_size=16)
    # retriever = EvaluateRetrieval(text_model, score_function="cos_sim") # or "cos_sim" for cosine similarity

    # text_processor = AutoProcessor.from_pretrained("sentence-transformers/msmarco-distilbert-base-tas-b")
    # model = models.SentenceBERT("msmarco-distilbert-base-tas-b")
    # model = model.eval()


full_data_path = os.path.join(args.data_path, args.dataset_name)
if args.dataset_name.startswith("crepe"):
    full_data_path = os.path.join(args.data_path, "crepe")

# change 1
# query_path = os.path.dirname(os.path.realpath(__file__)) 
query = 'What color is the Santa Anita Park logo?'



if not os.path.exists(full_data_path):
    os.makedirs(full_data_path)

pipe, generation_args= None, None
if args.use_phi:
    pipe, generation_args = init_phi_utils()
    args.cached_file_suffix="_phi"

# origin_corpus = None
bboxes_ls = None
grouped_sub_q_ids_ls = None
bboxes_overlap_ls = None
img_file_name_ls = None

# , bboxes_ls=bboxes_ls, img_file_name_ls=img_file_name_ls, bboxes_overlap_ls=bboxes_overlap_ls, grouped_sub_q_ids_ls=grouped_sub_q_ids_ls


if args.dataset_name == "art":
    queries, img_file_name_ls, sub_queries_ls, img_idx_ls, grouped_sub_q_ids_ls = load_art_datasets(full_data_path, query, subset_img_id=args.subset_img_id, is_test=args.is_test)

elif args.dataset_name == "multiqa":
    queries, img_file_name_ls, sub_queries_ls, img_idx_ls, grouped_sub_q_ids_ls = load_art_datasets(full_data_path, query, subset_img_id=args.subset_img_id, is_test=args.is_test)

elif args.dataset_name == "multiqa_few":
    queries, img_file_name_ls, sub_queries_ls, img_idx_ls, grouped_sub_q_ids_ls = load_multiqafew_datasets(full_data_path, query_path, subset_img_id=args.subset_img_id, is_test=args.is_test)
    
elif args.dataset_name == "manyqa":
    queries, img_file_name_ls, sub_queries_ls, img_idx_ls, grouped_sub_q_ids_ls = load_manyqa_datasets(full_data_path, query_path, subset_img_id=args.subset_img_id, is_test=args.is_test)
    
elif args.dataset_name == "webqa":
    queries, img_file_name_ls, sub_queries_ls, img_idx_ls, grouped_sub_q_ids_ls = load_webqa_datasets(full_data_path, query_path, subset_img_id=args.subset_img_id, is_test=args.is_test)



if args.is_img_retrieval:
    # if not args.query_concept:    
    #     patch_count_ls = [4, 8, 16, 32, 64, 128]
    # else:
        # patch_count_ls = [4, 8, 16, 32, 64, 128]
        if args.dataset_name.startswith("crepe"):
            patch_count_ls = [4,8,16, 64,128]
            # patch_count_ls = [32]
        elif args.dataset_name.startswith("mscoco"):
            # patch_count_ls = [4, 8, 16, 64, 128]
            patch_count_ls = [4, 8, 16, 64]
            # patch_count_ls = [4, 16, 64]
        else:
            patch_count_ls = [4, 8, 16, 64]   #use this
            
        if args.segmentation_method == "scene_graph":
            patch_count_ls = [1, 4, 8, 16, 32]
else:
    # patch_count_ls = [8, 24, 32]
    patch_count_ls = [1, 16, 8, 4, 32]
    # if not args.dataset_name == "fiqa":
    #     patch_count_ls = [1, 4, 8, 16, 32]
    # else:
    #     patch_count_ls = [4, 32, 128, 256]
    # patch_count_ls = [1]
    # patch_count_ls = [32]

if args.subset_patch_count > 0 and args.subset_patch_count < len(patch_count_ls):
    patch_count_ls = patch_count_ls[:args.subset_patch_count]

if args.is_img_retrieval:
    # print('to here~~~~~~~~1')
    samples_hash = obtain_sample_hash(img_idx_ls, img_file_name_ls)
    # cached_img_idx_ls, image_embs, patch_activations, masks, bboxes, img_for_patch
    # if args.save_mask_bbox:

    cached_img_ls, img_emb, patch_emb_ls, _, bboxes_ls, img_per_patch_ls = convert_samples_to_concepts_img(args, samples_hash, model, img_file_name_ls, img_idx_ls, processor, device, patch_count_ls=patch_count_ls,save_mask_bbox=args.save_mask_bbox)  # debug
    # else:
    #     cached_img_ls, img_emb, patch_emb_ls, img_per_patch_ls = convert_samples_to_concepts_img(args, samples_hash, model, img_file_name_ls, img_idx_ls, processor, device, patch_count_ls=patch_count_ls,save_mask_bbox=args.save_mask_bbox)
        
    if args.add_sparse_index:
        img_sparse_emb = construct_dense_or_sparse_encodings(args, corpus, text_model, samples_hash, is_sparse=True)
        store_sparse_index(samples_hash, img_sparse_emb, encoding_query = False)

elif args.dataset_name in text_retrieval_datasets:
    
    if (args.use_raptor):
        raptor_model = RaptorEmbeddingGenerator()
    else:
        raptor_model = None
    samples_hash,(img_emb, img_sparse_index), patch_emb_ls, bboxes_ls = convert_samples_to_concepts_txt(args, text_model, corpus, device, raptor_model=raptor_model, patch_count_ls=patch_count_ls)
    
    # img_emb = text_model.encode_corpus(corpus)
    # if args.img_concept:
    #     _,img_per_patch_ls, patch_emb_ls = generate_patch_ids_ls(patch_emb_ls)
else:
    print("Invalid dataset name, exit!")
    exit(1)



print("sample hash::", samples_hash)
patch_emb_by_img_ls = patch_emb_ls
if args.img_concept:
    # if args.is_img_retrieval:
    patch_emb_by_img_ls, bboxes_ls = reformat_patch_embeddings(patch_emb_ls, None, img_emb, bbox_ls=bboxes_ls)    # debug
    # else:
    #     patch_emb_by_img_ls, bboxes_ls = reformat_patch_embeddings(patch_emb_ls, img_per_patch_ls, img_emb, bbox_ls=bboxes_ls)
    
    # if args.is_img_retrieval:

    
    
sample_patch_ids_to_cluster_id_mappings = None
if args.search_by_cluster:
    if args.index_method == "dessert":
        import dessert_py_dependency
    if args.img_concept:
        
        # patch_clustering_info_cached_file = get_clustering_res_file_name(args, patch_count_ls)
        if type(patch_emb_by_img_ls) is list:
            patch_emb_by_img_ls = [torch.nn.functional.normalize(all_sub_corpus_embedding, p=2, dim=1) for all_sub_corpus_embedding in patch_emb_by_img_ls]
        else:
            patch_emb_by_img_ls = torch.nn.functional.normalize(patch_emb_by_img_ls, p=2, dim=1)

        
        patch_clustering_info_cached_file = get_dessert_clustering_res_file_name(args, samples_hash, patch_count_ls, clustering_number=args.clustering_number, index_method=args.index_method, typical_doclen=args.clustering_doc_count_factor, num_tables=args.num_tables, hashes_per_table=args.hashes_per_table)
        
        if not os.path.exists(patch_clustering_info_cached_file):
        
            centroid_file_name = get_clustering_res_file_name(args, samples_hash, patch_count_ls)
            if os.path.exists(centroid_file_name):
                centroids = torch.load(centroid_file_name)
            else:
                centroids =sampling_and_clustering(patch_emb_by_img_ls, dataset_name=args.dataset_name, clustering_number=args.clustering_number, typical_doclen=args.clustering_doc_count_factor)
                torch.save(centroids, centroid_file_name)
            # centroids = torch.zeros([1, patch_emb_by_img_ls[-1].shape[-1]])
            # hashes_per_table: int, num_tables
            max_patch_count = max([len(patch_emb_by_img_ls[idx]) for idx in range(len(patch_emb_by_img_ls))])
            if not args.index_method == "dessert":
                retrieval_method = DocRetrieval(max_patch_count, args.hashes_per_table, args.num_tables, patch_emb_by_img_ls[-1].shape[-1], centroids, device=device)
            else:
                retrieval_method = dessert_py_dependency.DocRetrieval(hashes_per_table = args.hashes_per_table, num_tables = args.num_tables, dense_input_dimension = patch_emb_by_img_ls[-1].shape[-1], nprobe_query=args.nprobe_query, centroids = centroids.detach().cpu().numpy().astype(np.float32));

            for idx in tqdm(range(len(patch_emb_by_img_ls)), desc="add doc"):
                if not args.index_method == "dessert":
                    retrieval_method.add_doc(patch_emb_by_img_ls[idx], idx, index_method=args.index_method)
                else:
                    retrieval_method.add_doc(patch_emb_by_img_ls[idx].detach().cpu().numpy().astype(np.float32), str(idx))
            
            # utils.save(retrieval_method, "output/retrieval_method.pkl")
            if not args.index_method == "dessert":
                utils.save(retrieval_method, patch_clustering_info_cached_file)
            else:
                retrieval_method.serialize_to_file(patch_clustering_info_cached_file)
        else:
            if not args.index_method == "dessert":
                retrieval_method = utils.load(patch_clustering_info_cached_file)
            else:
                retrieval_method = dessert_py_dependency.DocRetrieval.deserialize_from_file(patch_clustering_info_cached_file)
            
    else:
        patch_clustering_info_cached_file = get_dessert_clustering_res_file_name(samples_hash, [-1], clustering_number=args.clustering_number, index_method=args.index_method, typical_doclen=args.clustering_doc_count_factor)
        patch_emb_by_img_ls = [img_emb[idx].view(1,-1) for idx in range(len(img_emb))]
        if not os.path.exists(patch_clustering_info_cached_file):
        
            centroid_file_name = get_clustering_res_file_name(args, samples_hash, [-1])
            if os.path.exists(centroid_file_name):
                centroids = torch.load(centroid_file_name)
            else:
                centroids =sampling_and_clustering(patch_emb_by_img_ls, dataset_name=args.dataset_name, clustering_number=args.clustering_number, typical_doclen=args.clustering_doc_count_factor)
                torch.save(centroids, centroid_file_name)
            # centroids = torch.zeros([1, patch_emb_by_img_ls[-1].shape[-1]])
            # hashes_per_table: int, num_tables
            max_patch_count = max([len(patch_emb_by_img_ls[idx]) for idx in range(len(patch_emb_by_img_ls))])
            retrieval_method = DocRetrieval(max_patch_count, args.hashes_per_table, args.num_tables, patch_emb_by_img_ls[-1].shape[-1], centroids, device=device)

            for idx in tqdm(range(len(patch_emb_by_img_ls)), desc="add doc"):
                retrieval_method.add_doc(patch_emb_by_img_ls[idx], idx, index_method=args.index_method)
            
            # utils.save(retrieval_method, "output/retrieval_method.pkl")
            utils.save(retrieval_method, patch_clustering_info_cached_file)
        else:
            retrieval_method = utils.load(patch_clustering_info_cached_file)
        # cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls, cluster_sample_ids_ls, cluster_sub_X_patch_ids_ls, cluster_sub_X_granularity_ids_ls
        # cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls,cluster_sample_ids_ls, cluster_sub_X_patch_ids_ls, cluster_sub_X_granularity_ids_ls, cluster_sub_X_cat_patch_ids_ls, sample_patch_ids_to_cluster_id_mappings = clustering_img_patch_embeddings(patch_emb_by_img_ls, args.dataset_name + "_" + str(args.total_count), patch_emb_ls, closeness_threshold=args.closeness_threshold)
    if not args.index_method == "dessert":
        retrieval_method._centroids = torch.nn.functional.normalize(retrieval_method._centroids, p=2, dim=0)
        print("centroid shape::", retrieval_method._centroids.shape)
        
        
        # if False: #os.path.exists(patch_clustering_info_cached_file):
        #     cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls,cluster_sample_ids_ls, cluster_sub_X_patch_ids_ls, cluster_sub_X_granularity_ids_ls, cluster_sub_X_cat_patch_ids_ls = utils.load(patch_clustering_info_cached_file)
        # else: 
        #     utils.save((cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls,cluster_sample_ids_ls, cluster_sub_X_patch_ids_ls, cluster_sub_X_granularity_ids_ls, cluster_sub_X_cat_patch_ids_ls, sample_patch_ids_to_cluster_id_mappings), patch_clustering_info_cached_file)
    # else:
    #     cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_sample_ids_ls = clustering_img_embeddings(img_emb)

if args.img_concept:
    bboxes_overlap_ls, clustering_nbs_mappings = init_bbox_nbs(args, patch_count_ls, samples_hash, bboxes_ls, patch_emb_by_img_ls, sample_patch_ids_to_cluster_id_mappings)


def llava_vllm(img_ls, queries):  # 'You are a helpful assistant. Please answer the question according to the question and picture. You can only answer one word. '
    
    system ="You are a helpful assistant. You must answer the question strictly based on the image. If the image and question are unrelated, you must still derive your answer from the image."
    print(system)
    prompt = []
    for idx in range(len(img_ls)):
        image = PIL.Image.open('/home/icml01/images/manyqa/' + img_ls[idx])
        query = queries[idx]
        prompt.append({
                    "prompt": f"USER: <image>\n{system + query}\nASSISTANT:",
                    "multi_modal_data": {"image": image},
                })
    
    outputs = llm.generate(prompt, sampling_params)
    ans = []
    for o in outputs:
        generated_text = o.outputs[0].text
        ans.append(generated_text)
    return ans

def wo_llava_vllm(queries):  # 'You are a helpful assistant. Please answer the question according to the question and picture. You can only answer one word. '
    
    system ="You are a helpful assistant. You must answer the question strictly based on the image. If the image and question are unrelated, you must still derive your answer from the image."
    prompt = []
    for idx in range(len(queries)):
        query = queries[idx]
        prompt.append({
                    "prompt": f"USER: {system + query}\nASSISTANT:",
                })
    
    outputs = llm.generate(prompt, sampling_params)
    ans = []
    for o in outputs:
        generated_text = o.outputs[0].text
        ans.append(generated_text)
    return ans

def get_eval_answer_llava(query, sub_query_str_l, true_img_ls, patch_emb_by_img_ls = patch_emb_by_img_ls):
    # sub_query_str_l双层列表 [	['man| wearing| on his head| in this poster| for The Blacklist (TV series)'], []]
    queries = query
    sub_queries_ls = sub_query_str_l
    grouped_sub_q_ids_ls = None
    sparse_sim_scores = None

    assert len(query) == len(sub_query_str_l)


    if args.is_img_retrieval:
        qrels, queries, subset_q_idx = construct_qrels(args.dataset_name, queries, cached_img_ls, img_idx_ls, query_count=args.query_count)
        if args.query_count > 0:
            sub_queries_ls = [sub_queries_ls[idx] for idx in subset_q_idx]
            grouped_sub_q_ids_ls = [grouped_sub_q_ids_ls[idx] for idx in subset_q_idx]
        
        
    if args.is_img_retrieval:
        if args.query_concept:
            full_sub_queries_ls = [sub_queries_ls[idx] + [[queries[idx]]] for idx in range(len(sub_queries_ls))] 
            text_emb_ls = embed_queries_ls(args.model_name, full_sub_queries_ls, text_processor, model, device)

            print(f"======full_sub_queries_ls max len sub+query: {max(len(full_sub_queries_ls[idx][0]) for idx in range(len(full_sub_queries_ls)))}======")
            # print(f"======embedding shape: {max(len(full_sub_queries_ls[idx]) for idx in range(len(full_sub_queries_ls)))}======")
        else:
            text_emb_ls = embed_queries_with_input_queries(args.model_name, queries, text_processor, model, device)
            full_sub_queries_ls = []  # change-1
    else:
        if not args.query_concept:
            # text_emb_ls = text_model.encode_queries(queries, convert_to_tensor=True)
            text_emb_ls, _ = construct_dense_or_sparse_encodings_queries(queries, text_model, args.add_sparse_index)
                    
        else:
            
            full_sub_queries_ls = sub_queries_ls
            # full_sub_queries_ls = [sub_queries_ls[idx] + [[reformated_queries[idx]]] for idx in range(len(sub_queries_ls))]
            text_emb_ls = encode_sub_queries_ls(full_sub_queries_ls, text_model)
            text_emb_dense = text_model.encode_queries(queries, convert_to_tensor=True)
            text_emb_ls = [text_emb_ls[idx] + [text_emb_dense[idx].unsqueeze(0)] for idx in range(len(text_emb_ls))]
                
            
    if args.add_sparse_index:
        _, query_sparse_index = construct_dense_or_sparse_encodings_queries(queries, text_model, args.add_sparse_index)
        store_sparse_index(samples_hash, query_sparse_index, encoding_query = True)
        # text_emb_ls = text_retrieval_model.model.encode_queries(queries, convert_to_tensor=True)

        run_search_with_sparse_index(samples_hash)
        sparse_sim_scores = read_trec_run(samples_hash, len(queries), len(corpus))
    
    # retrieve_by_full_query(img_emb, text_emb_ls)
    
    # if args.is_img_retrieval:
    retrieval_model = DRES(batch_size=16, algebra_method=args.algebra_method, is_img_retrieval=(args.is_img_retrieval or not args.dataset_name == "webis-touche2020"), prob_agg=args.prob_agg, dependency_topk=args.dependency_topk)
    # retrieval_model = DRES(batch_size=16, algebra_method=args.algebra_method, is_img_retrieval=True, prob_agg=args.prob_agg, dependency_topk=args.dependency_topk)
    # else:
    #     retrieval_model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=16, algebra_method=one)
    retriever = EvaluateRetrieval(retrieval_model, score_function='cos_sim') # or "cos_sim" for cosine similarity
    
    if args.query_concept:
        if args.is_img_retrieval:
            text_emb_ls = [[torch.cat(item) if item else torch.tensor([]) for item in items] for items in text_emb_ls]

            # text_emb_ls = [[torch.cat(item) for item in items] for items in text_emb_ls]
    
    # print(text_emb_ls)
    # print(f"======final text_emb_ls: {max(len(full_sub_queries_ls[idx][0]) for idx in range(len(full_sub_queries_ls)))}======")
    # print(f"======final text_emb_ls: {max(len(full_sub_queries_ls[idx]) for idx in range(len(full_sub_queries_ls)))}======")

    # if args.query_concept:
    perc_method = "two"
    if args.retrieval_method == "ours":
        if not args.img_concept:
            if not args.search_by_cluster:
                if args.query_concept:
                    patch_emb_by_img_ls = [img_emb[idx].view(1,-1) for idx in range(len(img_emb))]
                    results=retrieve_by_embeddings(perc_method, full_sub_queries_ls, queries, retriever, patch_emb_by_img_ls, text_emb_ls, qrels, query_count=args.query_count, parallel=args.parallel, bboxes_ls=bboxes_ls, img_file_name_ls=img_file_name_ls, bboxes_overlap_ls=None, grouped_sub_q_ids_ls=None, clustering_topk=args.clustering_topk, sparse_sim_scores=sparse_sim_scores, dataset_name=args.dataset_name)
                else:
                    results=retrieve_by_embeddings(perc_method, full_sub_queries_ls, queries, retriever, img_emb, text_emb_ls, qrels, query_count=args.query_count, parallel=args.parallel, bboxes_ls=bboxes_ls, img_file_name_ls=img_file_name_ls, bboxes_overlap_ls=None, grouped_sub_q_ids_ls=None, clustering_topk=args.clustering_topk, sparse_sim_scores=sparse_sim_scores, dataset_name=args.dataset_name)
            else:
                # results=retrieve_by_embeddings(retriever, img_emb, text_emb_ls, qrels, query_count=args.query_count, parallel=args.parallel, use_clustering=args.search_by_cluster, clustering_info=(cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls, cluster_sample_ids_ls, cluster_sub_X_cat_patch_ids_ls, clustering_nbs_mappings), bboxes_ls=bboxes_ls, img_file_name_ls=img_file_name_ls, bboxes_overlap_ls=bboxes_overlap_ls, grouped_sub_q_ids_ls=grouped_sub_q_ids_ls, clustering_topk=args.clustering_topk, sparse_sim_scores=sparse_sim_scores)
                results=retrieve_by_embeddings(perc_method, full_sub_queries_ls, queries, retriever, patch_emb_by_img_ls, text_emb_ls, qrels, query_count=args.query_count, parallel=args.parallel, use_clustering=args.search_by_cluster, bboxes_ls=bboxes_ls, img_file_name_ls=img_file_name_ls, bboxes_overlap_ls=bboxes_overlap_ls, grouped_sub_q_ids_ls=grouped_sub_q_ids_ls,doc_retrieval=retrieval_method, dataset_name=args.dataset_name)
        else:
            # if args.dataset_name == "webis-touche2020":
            args.is_img_retrieval = True
                
            if not args.search_by_cluster:
                results=retrieve_by_embeddings(perc_method, full_sub_queries_ls, queries, retriever, patch_emb_by_img_ls, text_emb_ls, qrels, query_count=args.query_count, parallel=args.parallel, bboxes_ls=bboxes_ls, img_file_name_ls=img_file_name_ls, bboxes_overlap_ls=bboxes_overlap_ls, grouped_sub_q_ids_ls=grouped_sub_q_ids_ls, clustering_topk=args.clustering_topk, sparse_sim_scores=sparse_sim_scores, dataset_name=args.dataset_name, is_img_retrieval=args.is_img_retrieval)
            else:
                # results=retrieve_by_embeddings(retriever, patch_emb_by_img_ls, text_emb_ls, qrels, query_count=args.query_count, parallel=args.parallel, use_clustering=args.search_by_cluster, clustering_info=(cluster_sub_X_tensor_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls, cluster_sample_ids_ls, cluster_sub_X_cat_patch_ids_ls, clustering_nbs_mappings), bboxes_ls=bboxes_ls, img_file_name_ls=img_file_name_ls, bboxes_overlap_ls=bboxes_overlap_ls, grouped_sub_q_ids_ls=grouped_sub_q_ids_ls, clustering_topk=args.clustering_topk, sparse_sim_scores=sparse_sim_scores)
                # grouped_sub_q_ids_ls, bboxes_overlap_ls, dependency_topk, device, prob_agg, is_img_retrieval
                results=retrieve_by_embeddings(perc_method, full_sub_queries_ls, queries, retriever, patch_emb_by_img_ls, text_emb_ls, qrels, query_count=args.query_count, parallel=args.parallel, use_clustering=args.search_by_cluster, bboxes_ls=bboxes_ls, img_file_name_ls=img_file_name_ls, bboxes_overlap_ls=bboxes_overlap_ls, clustering_topk=args.clustering_topk, grouped_sub_q_ids_ls=grouped_sub_q_ids_ls,doc_retrieval=retrieval_method, prob_agg=args.prob_agg, dependency_topk=args.dependency_topk, device=device, is_img_retrieval=args.is_img_retrieval, method=args.algebra_method, index_method=args.index_method, _nprobe_query=args.nprobe_query, dataset_name=args.dataset_name)
    # elif args.retrieval_method == "llm_ranker":
    #     ranker = LLM_ranker(corpus)
    #     results = ranker.retrieval(queries)
    #     ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values, ignore_identical_ids=False)
    elif args.retrieval_method == "bm25":
        ranker = BuildIndex(samples_hash, corpus)
        t1 = time.time()

        results=ranker.retrieval(queries)
        t2 = time.time()
        print("retrieval time::", t2 - t1)
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values, ignore_identical_ids=False)
    else:
        raise ValueError("Invalid retrieval method")

    img_folder = '/home/icml01/images/manyqa/'
    img_list = os.listdir(img_folder)
    
    
    re_img_100 = []
    for idx in tqdm(range(len(queries))):
        tmp = []
        for j in range(100):
            i = int(list(results[str(idx + 1)].keys())[j])
            img = img_list[i - 1]
            tmp.append(img)
        re_img_100.append(tmp)
        
#     print(re_img_100[:3])
#     print(true_img_ls[:3])
    
    
    print(f"hit@20 :{hit_score(re_img_100, true_img_ls, 20)}")
    print(f"hit@100 :{hit_score(re_img_100, true_img_ls, 100)}")

    re_imgs = []
    for idx in tqdm(range(len(queries))):
        i = int(list(results[str(idx + 1)].keys())[0])
        img = img_list[i - 1]
        re_imgs.append(img)

        
    return llava_vllm(re_imgs, queries), re_imgs



def hit_score(passages_ls, anspids, k):
    assert len(passages_ls) == len(anspids)

    cnt = 0
    for i in range(len(passages_ls)):
        retrieved_100_topk = passages_ls[i][:k]
        ans_raw = anspids[i]
        if ans_raw in retrieved_100_topk:
            cnt += 1
           
    return cnt / len(passages_ls)
