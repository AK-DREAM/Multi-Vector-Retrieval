import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.append('/home/keli/Decompose_Retrieval/raptor')

import asyncio

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
import torch.cuda.nvtx as nvtx

image_retrieval_datasets = ["flickr", "AToMiC", "crepe", "crepe_full", "mscoco", "mscoco_40k", "multiqa" ,"multiqa_few", "manyqa", "webqa"]
text_retrieval_datasets = ["trec-covid", "nq", "climate-fever", "hotpotqa", "msmarco", "webis-touche2020", "scifact", "fiqa", "strategyqa"]
    
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
    parser.add_argument("--search_by_cluster", action="store_true", help="config file")
    parser.add_argument("--patch_count", type=int, default=16, help="config file")
    parser.add_argument("--agg_arg", type=float, default=0.5, help="config file")
    
    parser.add_argument('--retrieval_method', type=str, default="ours", help='config file')
    parser.add_argument('--index_method', type=str, default="default", choices=["default", "dessert", "dessert0"], help='config file')
    parser.add_argument('--hashes_per_table', type=int, default=5, help='config file')
    parser.add_argument('--num_tables', type=int, default=100, help='config file')
    parser.add_argument('--clustering_doc_count_factor', type=int, default=1, help='config file')
    parser.add_argument('--clustering_number', type=float, default=0.01, help='config file')
    parser.add_argument('--clustering_topk', type=int, default=10000, help='config file')
    parser.add_argument('--nprobe_query', type=int, default=50, help='config file')

    # useless args
    parser.add_argument("--store_res", action="store_true", help="config file")
    parser.add_argument('--segmentation_method', type=str, default="default", choices=["default", "scene_graph"], help='config file')
    parser.add_argument('--use_raptor', action="store_true", help='config file')
    parser.add_argument('--algebra_method', type=str, default=one, help='config file')
    parser.add_argument("--add_sparse_index", action="store_true", help="config file")

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

def construct_qrels(dataset_name, queries, cached_img_idx, img_idx_ls, query_count):
    qrels = {}
    
    for idx in range(len(queries)):
        curr_img_idx = img_idx_ls[idx]
        cached_idx = cached_img_idx.index(curr_img_idx)
        qrels[str(idx+1)] = {str(cached_idx+1): 2}
        
    q_idx_ls = list(range(len(queries)))
    if query_count > 0:
        # subset_q_idx_ls = list(range(query_count))
        subset_q_idx_ls = random.sample(q_idx_ls, query_count)
        
        subset_q_idx_ls = sorted(subset_q_idx_ls)
        
        subset_qrels = {str(key_id + 1): qrels[str(subset_q_idx_ls[key_id] + 1)] for key_id in range(len(subset_q_idx_ls))}
    
        qrels = subset_qrels
        
        queries = [queries[idx] for idx in subset_q_idx_ls]
    else:
        subset_q_idx_ls = q_idx_ls #list(qrels.keys())
        
    return qrels, queries, subset_q_idx_ls

def embed_img_queries_ls(model_name, queries, processor, model, device):
    text_emb_ls = []
    with torch.no_grad():
        for subquery in queries:
            inputs = processor(subquery)
            if model_name == "default":
                inputs = {key: val.to(device) for key, val in inputs.items()}
                text_features = model.get_text_features(**inputs)
            else:
                raise ValueError("Invalid model name")
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
            text_emb_ls.append(text_features)
    return torch.cat(text_emb_ls, dim=0)

def embed_text_queries_ls(text_model, sub_queries_ls):
    return text_model.encode_queries(sub_queries_ls, convert_to_tensor=True)

def cos_sim(a: torch.Tensor, b: torch.Tensor):
    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1)) #TODO: this keeps allocating GPU memory

def calc_sim_score(args, query_emb, subquery_embs, doc_embs, subdoc_embs, subdoc_cnt):
    subdoc_idx = torch.repeat_interleave(
        torch.arange(len(subdoc_cnt), device=device),
        torch.tensor(subdoc_cnt, device=device)
    )
    score1 = cos_sim(query_emb, doc_embs).squeeze(0) # [doc_cnt,]
    score2 = cos_sim(subquery_embs, subdoc_embs) # [subquery_cnt, sum_subdoc_cnt]

    sf = score2.transpose(0, 1)  # [sum_subdoc_cnt, subquery_cnt]
    c_batch = len(subdoc_cnt)
    q_tokens = sf.shape[1]
    maxed = torch.zeros(c_batch, q_tokens, device=device)
    maxed.scatter_reduce_(0,
        subdoc_idx[:, None].expand(-1, q_tokens),
        sf,
        reduce='amax', include_self=True
    ) # [subquery_cnt, doc_cnt]

    maxed[maxed < 0] = 0
    score2 = torch.prod(maxed.transpose(0, 1), dim=0) #[doc_cnt,]
    
    score = torch.stack([score1, score2], dim=0)
    score = score/torch.sum(score, dim=-1, keepdim=True)
    # weights = torch.tensor([1.0,0.0]).view(2,1).to('cuda')
    weights = torch.tensor([args.agg_arg, 1-args.agg_arg]).view(2,1).to('cuda')
    score = (weights*score).sum(dim=0)

    return score

def my_decompose_query(full_query, dataset_name):
    if dataset_name in ['crepe']:
        prompt = prompt_crepe(full_query)
    elif dataset_name == 'mscoco':
        prompt = prompt_mscoco(full_query)
    elif dataset_name == 'trec-covid':
        prompt = prompt_trec_covid(full_query)
    else: 
        print("unknown dataset")
        exit(1)
    outputs = obtain_response_from_llama3_utils(prompt)
    return decompose_single_query(outputs, reg_pattern="\|")

def get_subquery_embs(curr_query, args, sub_queries_ls, model, text_processor, device):
    nvtx.range_push("get subquery embs")
    # sub_queries = sub_queries_ls[idx]
    sub_queries = my_decompose_query(curr_query, args.dataset_name)
    print(curr_query, sub_queries)
    emb = embed_img_queries_ls(args.model_name, sub_queries, text_processor, model, device)
    nvtx.range_pop()
    return emb


def search_and_load_data(
    fullquery_emb, args, retrieval_method,
    all_doc_embs, all_subdoc_embs, device
):
    nvtx.range_push("search and load data")
    sample_num = args.clustering_topk

    if args.search_by_cluster: 
        sample_ids_gpu = retrieval_method.query(fullquery_emb, sample_num, sample_num, args.nprobe_query)
    else:
        sample_ids_gpu = torch.tensor(range(len(all_doc_embs)), dtype=torch.long, device=device)

    sample_ids = sample_ids_gpu.cpu()

    sample_doc_embs = all_doc_embs[sample_ids].to(device)
    sample_subdoc_cnt = [len(all_subdoc_embs[idx]) for idx in sample_ids]
    sample_subdoc_embs = [all_subdoc_embs[idx] for idx in sample_ids]
    sample_subdoc_embs = torch.cat(sample_subdoc_embs, dim=0).to(device)

    nvtx.range_pop()

    return sample_ids_gpu, sample_doc_embs, sample_subdoc_cnt, sample_subdoc_embs

async def work_pipeline(curr_query, sub_queries_ls, model, text_processor, fullquery_emb, retrieval_method, all_doc_embs, all_subdoc_embs, args, device):
    task1 = asyncio.to_thread(
        get_subquery_embs, curr_query, args, sub_queries_ls, model, text_processor, device
    )
    task2 = asyncio.to_thread(
        search_and_load_data, fullquery_emb, args, retrieval_method, 
        all_doc_embs, all_subdoc_embs, device
    )
    subquery_embs, (sample_ids_gpu, sample_doc_embs, sample_subdoc_cnt, sample_subdoc_embs) = await asyncio.gather(
        task1, task2
    )
    return subquery_embs, sample_ids_gpu, sample_doc_embs, sample_subdoc_cnt, sample_subdoc_embs

# 可以调整的一些超参数：
# patch_count = [16]
# weights = torch.tensor([0.7,0.3]) 表示粗粒度和细粒度分别的权重
# 和 indexing 有关的：
# clustering_number：聚成 clustering_number*N 类
# nprobe_query：搜索时找多少个 centroids
# clustering_topk：一共保留多少 documents

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

    # 加载 dataset embedding 模型 (CLIP)
    if args.model_name == "default":
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        raw_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        processor =  lambda images: raw_processor(images=images, return_tensors="pt")["pixel_values"]
        text_processor =  lambda text: raw_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        img_processor =  lambda images: raw_processor(images=images, return_tensors="pt")["pixel_values"]
        model = model.eval()
    else:
        print("invalid model name")
        exit(1)

    full_data_path = os.path.join(args.data_path, args.dataset_name)

    query_path = os.path.dirname(os.path.realpath(__file__)) 
    
    if not os.path.exists(full_data_path):
        os.makedirs(full_data_path)
    
    # 加载 data 和 query 文件
    if args.dataset_name == "crepe":
        queries, img_file_name_ls, sub_queries_ls, img_idx_ls, grouped_sub_q_ids_ls = load_crepe_datasets(full_data_path, query_path)
        img_idx_ls, img_file_name_ls = load_other_crepe_images(full_data_path, query_path, img_idx_ls, img_file_name_ls, total_count = args.total_count)
        
        sub_queries_ls = [t[0] for t in sub_queries_ls]

    elif args.dataset_name == "mscoco":
        queries, img_file_name_ls, sub_queries_ls, img_idx_ls, grouped_sub_q_ids_ls = load_mscoco_datasets_new(full_data_path, args.total_count)

    elif args.dataset_name == "webqa":
        pass

    else:
        print("Invalid dataset!")
        exit

    patch_count_ls = [args.patch_count]

    print("Img count: ", len(img_idx_ls))

    # 对 dataset 进行 embedding
    print("embedding dataset")
    samples_hash = obtain_sample_hash(img_idx_ls, img_file_name_ls)
    cached_img_ls, img_emb, patch_emb_ls, _, bboxes_ls, img_per_patch_ls = convert_samples_to_concepts_img(args, samples_hash, model, img_file_name_ls, img_idx_ls, processor, device, patch_count_ls=patch_count_ls)  
    patch_emb_by_img_ls, bboxes_ls = reformat_patch_embeddings(patch_emb_ls, None, img_emb, bbox_ls=bboxes_ls)  

    # clustering
    if args.search_by_cluster:
        print("building indexes")
        patch_emb_by_img_ls = [torch.nn.functional.normalize(all_sub_corpus_embedding, p=2, dim=1) for all_sub_corpus_embedding in patch_emb_by_img_ls]
        
        patch_clustering_info_cached_file = get_dessert_clustering_res_file_name(args, samples_hash, patch_count_ls, clustering_number=args.clustering_number, index_method=args.index_method, typical_doclen=args.clustering_doc_count_factor, num_tables=args.num_tables, hashes_per_table=args.hashes_per_table)
        
        if not os.path.exists(patch_clustering_info_cached_file):
            centroid_file_name = get_clustering_res_file_name(args, samples_hash, patch_count_ls)
            if os.path.exists(centroid_file_name):
                centroids = torch.load(centroid_file_name)
            else:
                centroids = sampling_and_clustering(patch_emb_by_img_ls, dataset_name=args.dataset_name, clustering_number=args.clustering_number, typical_doclen=args.clustering_doc_count_factor)
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
    else:
        retrieval_method = None

    # 建立 qrels (评测得分)
    if args.is_img_retrieval:
        qrels, queries, subset_q_idx = construct_qrels(args.dataset_name, queries, cached_img_ls, img_idx_ls, query_count=args.query_count)
        if args.query_count > 0 and (not sub_queries_ls == None):
            sub_queries_ls = [sub_queries_ls[idx] for idx in subset_q_idx]  

    all_doc_embs = img_emb
    all_subdoc_embs = patch_emb_by_img_ls

    print("preprocess ok, start processing queries")

    query_ids = [str(idx+1) for idx in list(range(len(queries)))]
    corpus_ids = [str(idx+1) for idx in list(range(len(all_doc_embs)))]
    results = {qid: {} for qid in query_ids}
    query_count = len(queries)
    k_values = [1,3,5,10,20,100]
    cos_scores_top_k_values = []
    cos_scores_top_k_idx = []

    # embedding original queries
    query_embs = embed_img_queries_ls(args.model_name, queries, text_processor, model, device)

    
    for idx in tqdm(range(query_count), desc="processing queries"):
        nvtx.range_push("single query")
        curr_query = queries[idx]
        fullquery_emb = query_embs[idx].unsqueeze(0)

        if not args.parallel:
            # parse query & embedding
            subquery_embs = get_subquery_embs(curr_query, args, sub_queries_ls, model, text_processor, device)

            # search indexes & load data
            sample_ids_gpu, sample_doc_embs, sample_subdoc_cnt, sample_subdoc_embs = search_and_load_data(
                fullquery_emb, args, retrieval_method,
                all_doc_embs, all_subdoc_embs, device
            )
        else:
            # async version
            subquery_embs, sample_ids_gpu, sample_doc_embs, sample_subdoc_cnt, sample_subdoc_embs = asyncio.run(work_pipeline(
                curr_query, sub_queries_ls, model, text_processor,
                fullquery_emb, retrieval_method, all_doc_embs, all_subdoc_embs, 
                args, device
            ))

        # reranking
        nvtx.range_push("reranking")
        sim_score = calc_sim_score(args, fullquery_emb, subquery_embs, sample_doc_embs, sample_subdoc_embs, sample_subdoc_cnt)
        cos_scores_top_k_values_raw, cos_scores_top_k_idx_raw = torch.topk(sim_score, min(k_values[-1]+1, len(sim_score)), largest=True)
        
        cos_scores_top_k_values.append(cos_scores_top_k_values_raw)
        cos_scores_top_k_idx.append(sample_ids_gpu[cos_scores_top_k_idx_raw])
        nvtx.range_pop()
        
        nvtx.range_pop()

    cos_scores_top_k_values = [t.cpu().tolist() for t in cos_scores_top_k_values]
    cos_scores_top_k_idx = [t.cpu().tolist() for t in cos_scores_top_k_idx]
    for query_itr in range(len(queries)):
        query_id = query_ids[query_itr]                  
        for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
            corpus_id = corpus_ids[sub_corpus_id]
            results[query_id][corpus_id] = score

    evaluator = EvaluateRetrieval(k_values=k_values) 
    print("Overall scores:")
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, evaluator.k_values, ignore_identical_ids=False)
        

    print("finish")
        
    used_memory = obtain_memory_usage()
    used_gpu_memory = obtain_gpu_memory_usage()
    
    print("used CPU memory::", used_memory)
    print("used GPU memory::", used_gpu_memory)
    
    # final_res_file_name = utils.get_final_res_file_name(args, patch_count_ls)
    # if args.store_res:
    #     print("The results are stored at ", final_res_file_name)
    #     utils.save(results, final_res_file_name)
    
    # print(results_without_decomposition)