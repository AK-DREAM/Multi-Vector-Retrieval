import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.append('/home/keli/Decompose_Retrieval/raptor')

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
import pickle
import pandas as pd

image_retrieval_datasets = ["flickr", "AToMiC", "crepe", "crepe_full", "mscoco", "mscoco_40k", "multiqa" ,"multiqa_few", "manyqa"]
text_retrieval_datasets = ["trec-covid", "nq", "climate-fever", "hotpotqa", "msmarco", "webis-touche2020", "scifact", "fiqa", "strategyqa"]

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
    # print(text_emb_ls)
    return text_emb_ls
    
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
    parser.add_argument('--nprobe_query', type=int, default=10, help='config file')
    parser.add_argument('--subset_patch_count', type=int, default=-1, help='config file')
    parser.add_argument('--cached_file_suffix', type=str, default="", help='config file')
    parser.add_argument("--is_test", action="store_true", help="config file")
    parser.add_argument("--store_res", action="store_true", help="config file")
    parser.add_argument("--use_phi", action="store_true", help="config file")
    parser.add_argument('--use_raptor', action="store_true", help='config file')
    
    args = parser.parse_args()
    return args

import psutil
import os

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
    return used_gpu_memory / 1024 ** 3


def construct_q_qrels(queries, dataset_name='strategyqa'):
    q = {}
    qrels = {}
    
    for query in queries:
        q.update({hashlib.md5(query.encode('utf-8')).hexdigest() : query})
    
    q_key_list = list(q.keys())
   
    for idx in range(len(queries)):
        q_key = q_key_list[idx]
    
        qrels[q_key] = {str(idx+1): 2}
        
    return q, qrels


def construct_qrels(dataset_name, queries, cached_img_idx, img_idx_ls, query_count):
    qrels = {}
    
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


if __name__ == "__main__":       
    logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
    used_memory0 = psutil.virtual_memory().used
    
    args = parse_args()
    print(args)
    args.is_img_retrieval = args.dataset_name in image_retrieval_datasets
    set_rand_seed(args.random_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载 dataset embedding 模型
    if args.is_img_retrieval:
        if args.model_name == "default":
            model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
            raw_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
            processor =  lambda images: raw_processor(images=images, return_tensors="pt")["pixel_values"]
            text_processor =  lambda text: raw_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            img_processor =  lambda images: raw_processor(images=images, return_tensors="pt")["pixel_values"]
            model = model.eval()
        model = model.eval()
    else:
        if args.model_name == "default":
            print("start loading distill-bert model")
            if not os.path.exists("output/msmarco-distilbert-base-tas-b.pkl"):
                text_model = models.SentenceBERT("msmarco-distilbert-base-tas-b", prefix = sparse_prefix, suffix=sparse_suffix) # offline model
                utils.save(text_model, "output/msmarco-distilbert-base-tas-b.pkl")
            else:
                text_model = utils.load("output/msmarco-distilbert-base-tas-b.pkl")
        elif args.model_name == "llm":
            text_model = models.LlmtoVec(("McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp"),
                ("McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised", "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised"))
        else:
            raise ValueError("Invalid model name")
        
        text_retrieval_model = DRES(batch_size=16)

    full_data_path = os.path.join(args.data_path, args.dataset_name)
    if args.dataset_name.startswith("crepe"):
        full_data_path = os.path.join(args.data_path, "crepe")

    query_path = os.path.dirname(os.path.realpath(__file__)) 
    
    if not os.path.exists(full_data_path):
        os.makedirs(full_data_path)
    
    pipe, generation_args= None, None
    
    bboxes_ls = None
    grouped_sub_q_ids_ls = None
    bboxes_overlap_ls = None
    img_file_name_ls = None
    
    # 加载 data 和 query 文件
    if args.dataset_name == "crepe":
        queries, img_file_name_ls, sub_queries_ls, img_idx_ls, grouped_sub_q_ids_ls = load_crepe_datasets(full_data_path, query_path, subset_img_id=args.subset_img_id, is_test=args.is_test)
        img_idx_ls, img_file_name_ls = load_other_crepe_images(full_data_path, query_path, img_idx_ls, img_file_name_ls, total_count = args.total_count)
        # grouped_sub_q_ids_ls = group_dependent_segments_seq_all(queries, sub_queries_ls, full_data_path, None, query_key_ls=None) # [None for _ in range(len(queries))]
        args.algebra_method=two
        if  args.retrieval_method == "bm25" or args.add_sparse_index:
            corpus = load_crepe_text_datasets(full_data_path, query_path, img_idx_ls)
    elif args.dataset_name == "trec-covid":
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset_name)
        data_path = util.download_and_unzip(url, full_data_path)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")       
        full_query_key_ls = [str(idx + 1) for idx in range(len(queries))]
        query_key_ls = full_query_key_ls
        sub_queries_ls, idx_to_rid = decompose_queries_into_sub_queries(queries, data_path, query_key_ls=query_key_ls, cached_file_suffix=args.cached_file_suffix, use_phi=args.use_phi, pipe=pipe, generation_args=generation_args, dataset_name="trec-covid")
        # print(sub_queries_ls)
        qrels = {key: qrels[idx_to_rid[key]] for key in sub_queries_ls if not check_empty_mappings(qrels[idx_to_rid[key]])}
        if len(qrels) == 0:
            print("no valid queries, exit!")
            exit(1)
        
        origin_corpus = None 
        corpus, qrels = convert_corpus_to_concepts_txt(corpus, qrels)
        query_key_idx_ls = [full_query_key_ls.index(key) for key in query_key_ls]
        args.algebra_method=three
        queries = [queries[key] for key in query_key_ls]

    if args.is_img_retrieval:
        patch_count_ls = [16] 
    else:
        patch_count_ls = [1]

    # 对 dataset 进行 embedding
    print("embedding dataset")
    if args.is_img_retrieval:
        samples_hash = obtain_sample_hash(img_idx_ls, img_file_name_ls)
        cached_img_ls, img_emb, patch_emb_ls, _, bboxes_ls, img_per_patch_ls = convert_samples_to_concepts_img(args, samples_hash, model, img_file_name_ls, img_idx_ls, processor, device, patch_count_ls=patch_count_ls,save_mask_bbox=args.save_mask_bbox)  # debug
    elif args.dataset_name in text_retrieval_datasets:
        samples_hash,(img_emb, img_sparse_index), patch_emb_ls, bboxes_ls = convert_samples_to_concepts_txt(args, text_model, corpus, device, raptor_model=None, patch_count_ls=patch_count_ls)
    else:
        print("Invalid dataset name, exit!")
        exit(1)

    patch_emb_by_img_ls = patch_emb_ls
    if args.img_concept:
        patch_emb_by_img_ls, bboxes_ls = reformat_patch_embeddings(patch_emb_ls, None, img_emb, bbox_ls=bboxes_ls)    # debug

    if args.search_by_cluster:
        print("building indexes")
        if args.img_concept:
            patch_emb_by_img_ls = [torch.nn.functional.normalize(all_sub_corpus_embedding, p=2, dim=1) for all_sub_corpus_embedding in patch_emb_by_img_ls]
            
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
                retrieval_method = DocRetrieval(max_patch_count, args.hashes_per_table, args.num_tables, patch_emb_by_img_ls[-1].shape[-1], centroids, device=device)

                for idx in tqdm(range(len(patch_emb_by_img_ls)), desc="add doc"):
                    retrieval_method.add_doc(patch_emb_by_img_ls[idx], idx, index_method=args.index_method)

                utils.save(retrieval_method, patch_clustering_info_cached_file)
            else:
                retrieval_method = utils.load(patch_clustering_info_cached_file)           
        
        retrieval_method._centroids = torch.nn.functional.normalize(retrieval_method._centroids, p=2, dim=0)
        print("centroid shape::", retrieval_method._centroids.shape)

    sparse_sim_scores = None
    if args.is_img_retrieval:
        qrels, queries, subset_q_idx = construct_qrels(args.dataset_name, queries, cached_img_ls, img_idx_ls, query_count=args.query_count)
        if args.query_count > 0:
            sub_queries_ls = [sub_queries_ls[idx] for idx in subset_q_idx]

    print("embedding queries")
    if args.is_img_retrieval:
        full_sub_queries_ls = [sub_queries_ls[idx] + [[queries[idx]]] for idx in range(len(sub_queries_ls))]   #change
        text_emb_ls = embed_queries_ls(args.model_name, full_sub_queries_ls, text_processor, model, device)
        text_emb_ls = [[torch.cat(item) for item in items] for items in text_emb_ls]
    else:
        full_sub_queries_ls = sub_queries_ls
        text_emb_ls = encode_sub_queries_ls(full_sub_queries_ls, text_model)
        text_emb_dense = text_model.encode_queries(queries, convert_to_tensor=True)
        text_emb_ls = [text_emb_ls[idx] + [text_emb_dense[idx].unsqueeze(0)] for idx in range(len(text_emb_ls))]
            
    retrieval_model = DRES(batch_size=16, algebra_method=args.algebra_method, is_img_retrieval=(args.is_img_retrieval or not args.dataset_name == "webis-touche2020"), prob_agg=args.prob_agg, dependency_topk=args.dependency_topk)
     
    retriever = EvaluateRetrieval(retriever=retrieval_model, score_function='cos_sim') # or "cos_sim" for cosine similarity      

    perc_method = "two"
    if args.retrieval_method == "ours":
        if not args.search_by_cluster: 
            results=retrieve_by_embeddings(perc_method, full_sub_queries_ls, queries, retriever, img_emb, patch_emb_by_img_ls, text_emb_ls, qrels, query_count=args.query_count, parallel=args.parallel, img_file_name_ls=img_file_name_ls, clustering_topk=args.clustering_topk, sparse_sim_scores=sparse_sim_scores, dataset_name=args.dataset_name, is_img_retrieval=args.is_img_retrieval)
        else:
            results=retrieve_by_embeddings(perc_method, full_sub_queries_ls, queries, retriever, img_emb, patch_emb_by_img_ls, text_emb_ls, qrels, query_count=args.query_count, parallel=args.parallel, use_clustering=args.search_by_cluster, img_file_name_ls=img_file_name_ls, clustering_topk=args.clustering_topk, doc_retrieval=retrieval_method, dependency_topk=args.dependency_topk, device=device, is_img_retrieval=args.is_img_retrieval, _nprobe_query=args.nprobe_query, dataset_name=args.dataset_name)
    else:
        raise ValueError("Invalid retrieval method")
        
    print("finish")
        
    used_memory = obtain_memory_usage()
    used_gpu_memory = obtain_gpu_memory_usage()
    
    print("used CPU memory::", used_memory)
    print("used GPU memory::", used_gpu_memory)
    
    final_res_file_name = utils.get_final_res_file_name(args, patch_count_ls)
    if args.store_res:
        print("The results are stored at ", final_res_file_name)
        utils.save(results, final_res_file_name)
    
    # print(results_without_decomposition)