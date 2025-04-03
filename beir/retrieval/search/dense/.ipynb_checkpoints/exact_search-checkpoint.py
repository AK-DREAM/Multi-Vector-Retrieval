from .util import cos_sim, dot_score, load_single_embeddings
import logging
import sys
import torch
from typing import Dict, List
from tqdm import tqdm
logger = logging.getLogger(__name__)
import concurrent.futures
import multiprocessing
from joblib import Parallel, delayed
import threading
import os
import cv2
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import gc
#Parent class for any dense model

one="one"
two="two"
three="three"
four="four"

def draw_bbox_on_single_image(img_file_name, bbox_ls):
    img = cv2.imread(img_file_name)
    for bbox in bbox_ls:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imwrite("img_with_bbox.jpg", img)
    


# def raw_score_fnc_batch(final_tensor, sub_corpus_embeddings, device):
#     """
#     Computes the cosine similarity scores for all query embeddings in a batch.
#     :param final_tensor: Tensor of shape [batch_size, num_queries, embedding_dim]
#     :return: Tensor of scores for all queries in the batch
#     """
#     batch_size, num_queries, embedding_dim = final_tensor.shape
#     # Convert sub_corpus_embeddings to a tensor and move to the same device
#     sub_corpus_embeddings_tensor = sub_corpus_embeddings[:-1]  # Shape: [num_corpus_embeddings, embedding_dim]

#     # Normalize sub_corpus_embeddings
#     sub_corpus_embeddings_norm = torch.nn.functional.normalize(sub_corpus_embeddings_tensor, p=2, dim=1).to(device)

#     # Normalize final_tensor embeddings
#     final_tensor_norm = torch.nn.functional.normalize(final_tensor, p=2, dim=-1).to(device)

#     # Compute cosine similarities in one go
#     # Shape: [batch_size, num_queries, num_corpus_embeddings]
#     cos_sim_matrix = torch.matmul(final_tensor_norm, sub_corpus_embeddings_norm.T)

#     # Compute max cosine similarity for each query across corpus embeddings
#     max_cos_sim, _ = torch.max(cos_sim_matrix, dim=-1)  # Shape: [batch_size, num_queries]

#     # Handle the special case for the last embedding
#     last_query_embeddings = final_tensor[:, -1, :].to(device)  # Shape: [batch_size, embedding_dim]
#     last_corpus_embedding = sub_corpus_embeddings[-1].to(device).unsqueeze(0)  # Shape: [1, embedding_dim]
#     last_query_cos_sim = cos_sim(last_query_embeddings, last_corpus_embedding)[:, 0]  # Shape: [batch_size]

#     # Combine scores
#     curr_scores_ls = max_cos_sim[:, :-1]  # Exclude the last query from max similarity scores
#     curr_scores_ls[curr_scores_ls == 0] = 1
#     curr_scores_ls[curr_scores_ls < 0] = 0

#     # Compute product for each batch
#     prod_scores = torch.prod(curr_scores_ls, dim=1)  # Shape: [batch_size]

#     # Concatenate product scores with last query similarity
#     final_scores = torch.cat([prod_scores.unsqueeze(1), last_query_cos_sim.unsqueeze(1)], dim=1)  # Shape: [batch_size, 2]

#     del sub_corpus_embeddings_norm, final_tensor_norm, cos_sim_matrix, max_cos_sim
#     del last_query_embeddings, last_corpus_embedding, curr_scores_ls, prod_scores
#     # gc.collect()

#     return final_scores

def raw_score_text(final_tensor, sub_corpus_embeddings, device):
    sub_corpus_embeddings = sub_corpus_embeddings.to(device) 

    sub_corpus_embeddings_tensor = sub_corpus_embeddings # [:-1]

    sub_corpus_embeddings_norm = torch.nn.functional.normalize(
        sub_corpus_embeddings_tensor, p=2, dim=1
    ).to(device)

    final_tensor_norm = final_tensor.to(device) 

    cos_sim_matrix = torch.matmul(final_tensor_norm, sub_corpus_embeddings_norm.T)

    # Compute max cosine similarity for each query across corpus embeddings
    max_cos_sim, _ = torch.max(cos_sim_matrix, dim=-1)  # Shape: [batch_size, num_queries]

    # Handle the special case for the last embedding
    last_query_embeddings = final_tensor[:, -1, :].to(device)  # Shape: [batch_size, embedding_dim]
    last_query_cos_sim = cos_sim(last_query_embeddings, sub_corpus_embeddings)[:, 0]  # Shape: [batch_size]

    # Combine scores
    curr_scores_ls = max_cos_sim[:, :-1]  # Exclude the last query from max similarity scores
    curr_scores_ls[curr_scores_ls == 0] = 1
    curr_scores_ls[curr_scores_ls < 0] = 0

    # Compute product for each batch
    prod_scores = torch.prod(curr_scores_ls, dim=1)

    # Concatenate product scores with last query similarity
    final_scores = torch.cat([prod_scores.unsqueeze(1), last_query_cos_sim.unsqueeze(1)], dim=1)  # Shape:
    
    return final_scores



# def raw_score_batch_text(final_tensor, sub_corpus_embeddings, device):
    
#     sub_corpus_embeddings = sub_corpus_embeddings.reshape(sub_corpus_embeddings.shape[0], sub_corpus_embeddings.shape[-1]).to(device)
#     sub_corpus_embeddings_norm = torch.nn.functional.normalize(
#         sub_corpus_embeddings, p=2, dim=1
#     )
  
#     cos_sim_matrix = torch.matmul(final_tensor, sub_corpus_embeddings_norm.T)
#     # cos_sim_matrix = 1 - torch.cdist(final_tensor_norm, sub_corpus_embeddings_norm, p=2)

#     last_query_embeddings = final_tensor[:, -1, :]
#     last_query_cos_sim = torch.matmul(last_query_embeddings, sub_corpus_embeddings_norm.T)  # [:, 0]

#     curr_scores_ls = cos_sim_matrix[:, :-1]

#     curr_scores_ls[curr_scores_ls == 0] = 1
#     curr_scores_ls[curr_scores_ls < 0] = 0

#     prod_scores = torch.prod(curr_scores_ls, dim=1)
   
#     combined_column_tensor = torch.stack((prod_scores, last_query_cos_sim), dim=1)
    
#     del cos_sim_matrix
#     del last_query_embeddings, last_query_cos_sim, curr_scores_ls, prod_scores
    
#     return combined_column_tensor




def process_all_embeddings(final_tensor, all_sub_corpus_embedding_ls, device):

    final_tensor_norm = torch.nn.functional.normalize(final_tensor, p=2, dim=-1).to(device)
    all_sub_corpus_embeddings_tensor = torch.stack(all_sub_corpus_embedding_ls)           #

    # 批处理大小
    batch_size = 64  # 根据你的 GPU 内存调整批次大小
    num_batches = len(all_sub_corpus_embedding_ls) // batch_size + 1

    all_cos_scores = []

    # 使用 DataLoader 进行批量处理
    dataset = TensorDataset(all_sub_corpus_embeddings_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch_idx, (sub_corpus_embeddings_batch,) in enumerate(tqdm(dataloader)):
        sub_corpus_embeddings_batch = sub_corpus_embeddings_batch.to(device)
        cos_scores_batch = raw_score_batch_text(
            final_tensor_norm, sub_corpus_embeddings_batch, device
        )
        # print(cos_scores_batch.shape)
        all_cos_scores.append(cos_scores_batch.cpu())  # 将结果从 GPU 移动到 CPU，防止内存溢出

    return torch.cat(all_cos_scores, dim=-1)




def raw_score_fnc_batch(final_tensor, sub_corpus_embeddings, device):
    """
    Computes the cosine similarity scores for all query embeddings in a batch.
    :param final_tensor: Tensor of shape [batch_size, num_queries, embedding_dim]
    :param sub_corpus_embeddings: Tensor of shape [num_corpus_embeddings, embedding_dim]
    :param device: The device (e.g., 'cuda:0')
    :return: Tensor of scores for all queries in the batch
    """
    # Ensure all inputs are on the correct device
    final_tensor = final_tensor.to(device)  # Move final_tensor to device
    sub_corpus_embeddings = sub_corpus_embeddings.to(device)  # Move sub_corpus_embeddings to device
    
    batch_size, num_queries, embedding_dim = final_tensor.shape
    # Convert sub_corpus_embeddings to a tensor and move to the same device
    sub_corpus_embeddings_tensor = sub_corpus_embeddings[:-1]  # Shape: [num_corpus_embeddings, embedding_dim]

    # Normalize sub_corpus_embeddings
    sub_corpus_embeddings_norm = torch.nn.functional.normalize(sub_corpus_embeddings_tensor, p=2, dim=1).to(device)

    # Normalize final_tensor embeddings
    final_tensor_norm = torch.nn.functional.normalize(final_tensor, p=2, dim=-1).to(device)

    # Compute cosine similarities in one go
    # Shape: [batch_size, num_queries, num_corpus_embeddings]
    cos_sim_matrix = torch.matmul(final_tensor_norm, sub_corpus_embeddings_norm.T)
    
    # print(cos_sim_matrix)

    # Compute max cosine similarity for each query across corpus embeddings
    max_cos_sim, _ = torch.max(cos_sim_matrix, dim=-1)  # Shape: [batch_size, num_queries]

    # Handle the special case for the last embedding
    last_query_embeddings = final_tensor[:, -1, :].to(device)  # Shape: [batch_size, embedding_dim]
    last_corpus_embedding = sub_corpus_embeddings[-1].to(device).unsqueeze(0)  # Shape: [1, embedding_dim]
    last_query_cos_sim = cos_sim(last_query_embeddings, last_corpus_embedding)[:, 0]  # Shape: [batch_size]

    # Combine scores
    curr_scores_ls = max_cos_sim[:, :-1]  # Exclude the last query from max similarity scores
    
    
    
    curr_scores_ls[curr_scores_ls == 0] = 1
    curr_scores_ls[curr_scores_ls < 0] = 0

    # Compute product for each batch
    prod_scores = torch.prod(curr_scores_ls, dim=1)  # Shape: [batch_size]

    # Concatenate product scores with last query similarity
    final_scores = torch.cat([prod_scores.unsqueeze(1), last_query_cos_sim.unsqueeze(1)], dim=1)  # Shape: [batch_size, 2]

    # Clean up unnecessary variables to free up memory
    del sub_corpus_embeddings_norm, final_tensor_norm, cos_sim_matrix, max_cos_sim
    del last_query_embeddings, last_corpus_embedding, curr_scores_ls, prod_scores
    # Optional: You can manually clear unused GPU memory if needed
    # torch.cuda.empty_cache()

    return final_scores




class DenseRetrievalExactSearch:
    
    def __init__(self, batch_size: int = 128, corpus_chunk_size: int = 50000, algebra_method="one", is_img_retrieval=False, prob_agg="prod", dependency_topk=20, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        # self.model = model
        self.batch_size = batch_size
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = True #TODO: implement no progress bar if false
        self.convert_to_tensor = True
        self.algebra_method = algebra_method
        self.results = {}
        self.is_img_retrieval = is_img_retrieval
        self.prob_agg = prob_agg
        self.dependency_topk=dependency_topk
    
    def check_nbs_info(self, curr_query_embedding, bboxes_overlap_ls, grouped_sub_q_ids, common_sample_ids, sample_to_cat_patch_idx_ls):
        
        valid_sample_count = 0
        
        valid_samples_to_patch_ids_mappings = dict()
        
        for sample_id in common_sample_ids:
            curr_bbox_overlaps = bboxes_overlap_ls[sample_id]
            is_valid=True
            curr_sample_all_overlapped_patch_ids = set()
            # if grouped_sub_q_ids is None:
            #     grouped_sub_q_ids = [list(range(curr_query_embedding.shape[0]))]
            
            
            for sub_q_ids in grouped_sub_q_ids:
                # for sub_q_ids in sub_q_ids_ls:
                    local_q_count = 0
                    overlapped_patch_ids = set()
                    for q_id in sub_q_ids:
                        
                        patch_ids = sample_to_cat_patch_idx_ls[q_id][sample_id]
                        
                        if local_q_count == 0:
                            overlapped_patch_ids.update(patch_ids)
                        else:
                            remaining_patch_ids = []
                            for patch_id in patch_ids:
                                if self.is_img_retrieval and patch_id >= len(curr_bbox_overlaps):
                                    continue
                                    
                                
                                curr_bbox_overlap = curr_bbox_overlaps[patch_id]
                                curr_overlapped_patch_ids = set(curr_bbox_overlap).intersection(set(overlapped_patch_ids))
                                if len(curr_overlapped_patch_ids) > 0:
                                    remaining_patch_ids.append(patch_id)
                                    
                            if len(remaining_patch_ids) <= 0:
                                is_valid=False
                                break      
                            else:
                                overlapped_patch_ids.update(remaining_patch_ids)                     
                        
                        local_q_count += 1
                    if not is_valid:
                        break
                    else:
                        curr_sample_all_overlapped_patch_ids.update(overlapped_patch_ids)
                        
                # if not is_valid:
                #     break
            if is_valid:
                valid_samples_to_patch_ids_mappings[sample_id] = curr_sample_all_overlapped_patch_ids
                valid_sample_count += 1
            
        return valid_sample_count, valid_samples_to_patch_ids_mappings
    
    def compute_dependency_aware_sim_score0(self, curr_query_embedding, sub_corpus_embeddings, corpus_idx, score_function, grouped_sub_q_ids_ls, sub_q_ls_idx, device, bboxes_overlap_ls, query_itr, valid_patch_ids=None):
        if grouped_sub_q_ids_ls[query_itr] is not None:
            curr_grouped_sub_q_ids_ls = grouped_sub_q_ids_ls[query_itr][sub_q_ls_idx]
        else:
            curr_grouped_sub_q_ids_ls = [list(range(curr_query_embedding.shape[0]))]
        
        if self.prob_agg == "prod":
            curr_scores_ls= 1
        else:
            curr_scores_ls = 0
        # curr_grouped_sub_q_ids_ls = [list(range(curr_query_embedding.shape[0]))]
        if self.is_img_retrieval:
            curr_sub_corpus_embeddings = sub_corpus_embeddings[0:-1]
        else:
            curr_sub_corpus_embeddings = sub_corpus_embeddings
        
        if len(curr_sub_corpus_embeddings) <= 0:
            return torch.tensor([0], device=device)
            
        for curr_grouped_sub_q_ids in curr_grouped_sub_q_ids_ls:
            
            selected_embedding_idx = torch.arange(curr_sub_corpus_embeddings.shape[0])
            beam_search_topk=min(self.dependency_topk, curr_sub_corpus_embeddings.shape[0])
            if self.prob_agg == "prod":
                sub_curr_scores = torch.ones(1).to(device)
            else:
                sub_curr_scores = torch.zeros(1).to(device)
            selected_patch_ids_ls = None
            for sub_query_idx in range(len(curr_grouped_sub_q_ids)): #range(curr_query_embedding.shape[0]):
                # print(curr_grouped_sub_q_ids, sub_query_idx)
                if valid_patch_ids is not None:
                    selected_embedding_idx = torch.tensor(list(set(selected_embedding_idx.tolist()).intersection(valid_patch_ids)))
                
                curr_prod_mat = self.score_functions[score_function](curr_query_embedding[curr_grouped_sub_q_ids[sub_query_idx]].to(device), curr_sub_corpus_embeddings[selected_embedding_idx].to(device)).view(-1,1)
                if self.prob_agg == "prod":
                    curr_prod_mat[curr_prod_mat < 0] = 0
                    prod_mat = curr_prod_mat*sub_curr_scores.view(1,-1)
                else:
                    prod_mat = curr_prod_mat+sub_curr_scores.view(1,-1)

                # beam_search_topk=max(20, int(torch.numel(prod_mat)*0.05) + 1)
                # print("beam_search_topk::", beam_search_topk)
                sub_curr_scores_ls, topk_ids = torch.topk(prod_mat.view(-1), k=min(beam_search_topk, torch.numel(prod_mat)), dim=-1)
                topk_emb_ids = topk_ids // prod_mat.shape[1]
                
                topk_emb_ids = selected_embedding_idx.to(device)[topk_emb_ids].tolist()
                # topk_emb_ids = list(set(topk_emb_ids.tolist()))
                if sub_query_idx == 0:
                    selected_patch_ids_ls = [[emb_id] for emb_id in topk_emb_ids]
                    # selected_embedding_idx = torch.cat([torch.tensor(bboxes_overlap_ls[corpus_idx][topk_id]).view(-1) for topk_id in topk_emb_ids])
                else:
                    selected_seq_ids = topk_ids%prod_mat.shape[1]
                    curr_selected_patch_ids_ls = [selected_patch_ids_ls[selected_seq_ids[selected_seq_id_idx]]+ [topk_emb_ids[selected_seq_id_idx]] for selected_seq_id_idx in range(len(selected_seq_ids))]
                    selected_patch_ids_ls = curr_selected_patch_ids_ls
                    # curr_selected_embedding_idx = torch.cat([torch.tensor(bboxes_overlap_ls[corpus_idx][topk_id]).view(-1) for topk_id in topk_emb_ids])
                    # selected_embedding_idx = torch.tensor(list(set(torch.cat([selected_embedding_idx, curr_selected_embedding_idx]).tolist())))
                existing_topk_emb_ids = set()
                for selected_patch_ids in selected_patch_ids_ls:
                    existing_topk_emb_ids.update(selected_patch_ids)
                # selected_embedding_idx = torch.cat([torch.tensor(bboxes_overlap_ls[corpus_idx][topk_id]).view(-1) for topk_id in existing_topk_emb_ids])
                selected_embedding_idx = set()
                for topk_id in existing_topk_emb_ids:
                    selected_embedding_idx.update(bboxes_overlap_ls[corpus_idx][topk_id])
                selected_embedding_idx = torch.tensor(list(selected_embedding_idx))
                sub_curr_scores = sub_curr_scores_ls
            if self.prob_agg == "prod":
                sub_curr_scores[sub_curr_scores <= 0] = 0
                curr_scores_ls *= torch.max(sub_curr_scores)
                assert torch.all(curr_scores_ls >= 0).item()
            else:
                curr_scores_ls += torch.max(sub_curr_scores)
                
        return curr_scores_ls
    
    # @profile
    def compute_dependency_aware_sim_score(self, curr_query_embedding, sub_corpus_embeddings, corpus_idx, score_function, grouped_sub_q_ids_ls, sub_q_ls_idx, device, bboxes_overlap_ls, query_itr, valid_patch_ids=None):
        if grouped_sub_q_ids_ls[query_itr] is not None:
            curr_grouped_sub_q_ids_ls = grouped_sub_q_ids_ls[query_itr][sub_q_ls_idx]
        else:
            curr_grouped_sub_q_ids_ls = [list(range(curr_query_embedding.shape[0]))]
        
        if self.prob_agg == "prod":
            curr_scores_ls= 1
        else:
            curr_scores_ls = 0
        # curr_grouped_sub_q_ids_ls = [list(range(curr_query_embedding.shape[0]))]
        if self.is_img_retrieval:
            curr_sub_corpus_embeddings = sub_corpus_embeddings[0:-1]
        else:
            curr_sub_corpus_embeddings = sub_corpus_embeddings
        
        full_scores =  self.score_functions[score_function](curr_query_embedding.to(device), curr_sub_corpus_embeddings.to(device))
        
        for curr_grouped_sub_q_ids in curr_grouped_sub_q_ids_ls:
            
            full_selected_embedding_idx = torch.zeros(curr_sub_corpus_embeddings.shape[0],device=device).bool()
            selected_embedding_idx = torch.arange(curr_sub_corpus_embeddings.shape[0])
            beam_search_topk=min(self.dependency_topk, curr_sub_corpus_embeddings.shape[0])
            if self.prob_agg == "prod":
                sub_curr_scores = torch.ones(1).to(device)
            else:
                sub_curr_scores = torch.zeros(1).to(device)
            selected_patch_ids_ls = None
            for sub_query_idx in range(len(curr_grouped_sub_q_ids)): #range(curr_query_embedding.shape[0]):
                # print(curr_grouped_sub_q_ids, sub_query_idx)
                if valid_patch_ids is not None:
                    selected_embedding_idx = torch.tensor(list(set(selected_embedding_idx.tolist()).intersection(valid_patch_ids)))
                
                # curr_prod_mat =  self.score_functions[score_function](curr_query_embedding[curr_grouped_sub_q_ids[sub_query_idx]].to(device), curr_sub_corpus_embeddings.to(device)[selected_embedding_idx]).view(-1,1)
                curr_prod_mat =  full_scores[curr_grouped_sub_q_ids[sub_query_idx], selected_embedding_idx].view(-1,1)
                if self.prob_agg == "prod":
                    curr_prod_mat[curr_prod_mat < 0] = 0
                    prod_mat = curr_prod_mat*sub_curr_scores.view(1,-1)
                else:
                    prod_mat = curr_prod_mat+sub_curr_scores.view(1,-1)

                # beam_search_topk=max(20, int(torch.numel(prod_mat)*0.05) + 1)
                # print("beam_search_topk::", beam_search_topk)
                sub_curr_scores_ls, topk_ids = torch.topk(prod_mat.view(-1), k=min(beam_search_topk, torch.numel(prod_mat)), dim=-1)
                topk_emb_ids = topk_ids // prod_mat.shape[1]
                
                topk_emb_ids_tensor = selected_embedding_idx.to(device)[topk_emb_ids]
                # topk_emb_ids = selected_embedding_idx.to(device)[topk_emb_ids].tolist()
                # topk_emb_ids = list(set(topk_emb_ids.tolist()))
                if sub_query_idx == 0:
                    # selected_patch_ids_ls = [[emb_id] for emb_id in topk_emb_ids]
                    selected_patch_ids_ls_tensor = topk_emb_ids_tensor.view(-1,1)
                    # selected_embedding_idx = torch.cat([torch.tensor(bboxes_overlap_ls[corpus_idx][topk_id]).view(-1) for topk_id in topk_emb_ids])
                else:
                    # selected_seq_ids = topk_ids%prod_mat.shape[1]
                    selected_seq_ids = torch.remainder(topk_ids, prod_mat.shape[1])
                    # curr_selected_patch_ids_ls = [selected_patch_ids_ls[selected_seq_ids[selected_seq_id_idx]]+ [topk_emb_ids[selected_seq_id_idx]] for selected_seq_id_idx in range(len(selected_seq_ids))]
                    curr_selected_patch_ids_ls_tensor = torch.cat([selected_patch_ids_ls_tensor[selected_seq_ids], topk_emb_ids_tensor.view(-1,1)], dim=-1)
                    # print(torch.max(torch.abs(curr_selected_patch_ids_ls_tensor.cpu() - torch.tensor(curr_selected_patch_ids_ls))))
                    # selected_patch_ids_ls = curr_selected_patch_ids_ls
                    selected_patch_ids_ls_tensor = curr_selected_patch_ids_ls_tensor
                    # curr_selected_embedding_idx = torch.cat([torch.tensor(bboxes_overlap_ls[corpus_idx][topk_id]).view(-1) for topk_id in topk_emb_ids])
                    # selected_embedding_idx = torch.tensor(list(set(torch.cat([selected_embedding_idx, curr_selected_embedding_idx]).tolist())))
                existing_topk_emb_ids = set()
                for selected_patch_ids in selected_patch_ids_ls_tensor.tolist():
                    existing_topk_emb_ids.update(selected_patch_ids)
                # existing_topk_emb_ids_tensor = selected_patch_ids_ls_tensor.view(-1).cpu().unique().tolist()
                # selected_embedding_idx = torch.cat([torch.tensor(bboxes_overlap_ls[corpus_idx][topk_id]).view(-1) for topk_id in existing_topk_emb_ids])
                selected_embedding_idx = set()
                for topk_id in existing_topk_emb_ids:
                    selected_embedding_idx.update(bboxes_overlap_ls[corpus_idx][topk_id])
                    # full_selected_embedding_idx[bboxes_overlap_ls[corpus_idx][topk_id]] = True
                selected_embedding_idx = torch.tensor(list(selected_embedding_idx))
                # selected_embedding_idx = full_selected_embedding_idx.nonzero().view(-1)
                sub_curr_scores = sub_curr_scores_ls
            if self.prob_agg == "prod":
                sub_curr_scores[sub_curr_scores <= 0] = 0
                curr_scores_ls *= torch.max(sub_curr_scores)
                assert torch.all(curr_scores_ls >= 0).item()
            else:
                curr_scores_ls += torch.max(sub_curr_scores)
                
        return curr_scores_ls

    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict, 
               top_k: List[int], 
               score_function: str,
               return_sorted: bool = False, 
               query_negations: List=None, all_sub_corpus_embedding_ls=None, query_embeddings=None, query_count=10, device = 'cuda', bboxes_ls=None, img_file_name_ls=None, bboxes_overlap_ls=None,grouped_sub_q_ids_ls=None,
               sparse_sim_scores=None, dataset_name="", **kwargs) -> Dict[str, Dict[str, float]]:
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
        
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        logger.info("Encoding Queries...")
        # if queries is not None:
        #     query_ids = list(queries.keys())
        # else:
        assert query_embeddings is  not None
        if query_embeddings is not None:
            query_ids = [str(idx+1) for idx in list(range(len(query_embeddings)))]
        else:
            raise ValueError("Either queries or query_embeddings must be provided!")
        self.results = {qid: {} for qid in query_ids}
        # if queries is not None:
        #     queries = [queries[qid] for qid in query_ids]
        # if query_negations is not None:
        #     query_negations = [query_negations[qid] if qid in query_negations else None for qid in query_ids]
        
        # if query_embeddings is None:
        #     query_embeddings=[]
        #     for idx in range(len(queries)):
        #         curr_query = queries[idx]
        #         if type(curr_query) is str:
        #             curr_query_embedding_ls = self.model.encode_queries(
        #                 curr_query, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
        #         elif type(curr_query) is list:
        #             curr_query_embedding_ls = []
        #             for k in range(len(curr_query)):
        #                 curr_conjunct = []
        #                 for j in range(len(curr_query[k])):
        #                     qe = self.model.encode_queries(
        #                         [curr_query[k][j]], batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
        #                     curr_conjunct.append(qe)
        #                 curr_query_embedding_ls.append(torch.cat(curr_conjunct))
        #         query_embeddings.append(curr_query_embedding_ls)
          
        logger.info("Sorting Corpus by document length (Longest first)...")

        # if corpus is not None:
        #     corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        #     corpus = [corpus[cid] for cid in corpus_ids]
        # else:
        assert all_sub_corpus_embedding_ls is not None
        if all_sub_corpus_embedding_ls is not None:
            corpus_ids = [str(idx+1) for idx in list(range(len(all_sub_corpus_embedding_ls)))]
        else:
            raise ValueError("Either corpus or all_sub_corpus_embedding_ls must be provided!")

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        if query_count < 0:
            query_count = len(query_embeddings)
            
        # query_embeddings = query_embeddings.to(device) # change

        assert all_sub_corpus_embedding_ls is not None
        all_cos_scores = []
       
        corpus_idx = 0
        
        max_len = max(len(row) for row in all_sub_corpus_embedding_ls)
        print(f"len:: {max_len}")
        
        # import pickle
        # with open('/home/icml01/multi_rag/RAG/Search-in-the-Chain/data/subcor.pkl', 'wb') as file:
        #     pickle.dump(all_sub_corpus_embedding_ls, file)

        if type(all_sub_corpus_embedding_ls) is list or type(query_embeddings[0]) is list:
            if dataset_name in {"multiqa", "manyqa", "webqa"}:
                max_len = max(len(tensor) for row in query_embeddings for tensor in row) + 1
                processed_tensors = []
                for row in query_embeddings:
                    concatenated_tensor = torch.cat(row, dim=0)
                    padding_size = max_len - concatenated_tensor.size(0)  # 计算需要填充的长度
                    # pad 参数: (left, right) -> (前向填充, 后向填充)，我们在第一维填充
                    padded_tensor = F.pad(concatenated_tensor, (0, 0, padding_size, 0))  # 填充到 maxlen
                    processed_tensors.append(padded_tensor)  # Stack to create a tensor of shape (maxlen, 768)
                final_tensor = torch.stack(processed_tensors).to(device)  # Shape: [85, maxlen, 768]

                # print(f"======final_tensor Shape: {final_tensor.shape}======")

                for sub_corpus_embeddings in tqdm(all_sub_corpus_embedding_ls):    
                    sub_corpus_embeddings = sub_corpus_embeddings.to(device)
                    cos_scores = raw_score_fnc_batch(final_tensor, sub_corpus_embeddings, device)
                    all_cos_scores.append(cos_scores)
                    
                all_cos_scores_tensor = torch.stack(all_cos_scores, dim=-1)
                
                print("=====Matrix-acc=====")
                
            elif dataset_name in {"strategyqa"}:
                
                #  query_embeddings 结构[[tensor1(3,768), tensor2(1,768)], [tensor1(2,768)， tnesor2(1,768)], [....]]
                # 一个[tensor1(3,768), tensor2(1,768)]代表一个query的分解，tensor1(3,768)代表一个query被分解为3个子查询， tensor2(1,768)代表原本的query整体编码，这里正常情况下只有2个Tensor
                # len(query_embeddings) 代表数据集 query的item数量
                max_len = max(len(tensor) for row in query_embeddings for tensor in row) + 1
                processed_tensors = []
                for row in query_embeddings:
                    row = [tensor.to(device) for tensor in row]
                    
                    concatenated_tensor = torch.cat(row, dim=0)
                    padding_size = max_len - concatenated_tensor.size(0)  # 计算需要填充的长度
                    # pad 参数: (left, right) -> (前向填充, 后向填充)，我们在第一维填充
                    padded_tensor = F.pad(concatenated_tensor, (0, 0, padding_size, 0))  # 填充到 maxlen
                    processed_tensors.append(padded_tensor)  # Stack to create a tensor of shape (maxlen, 768)
                final_tensor = torch.stack(processed_tensors).to(device)  
                
                final_tensor_norm = torch.nn.functional.normalize(final_tensor, p=2, dim=-1).to(device)

                # all_sub_corpus_embedding_ls代表按照不同patch分解的corpus，数据类型为list
                # 其中每一个sub_corpus_embeddings 类型为Tensor，形状为[n, 768]这个n代表的是按句子分解的数量
                # raw_score_text 计算的是所有query的子向量 final_tensor_norm 和 一个文档的子向量sub_corpus_embeddings的分数
                for sub_corpus_embeddings in tqdm(all_sub_corpus_embedding_ls):    
                    sub_corpus_embeddings = sub_corpus_embeddings.to(device)
                    cos_scores = raw_score_text(final_tensor_norm, sub_corpus_embeddings, device)
                    all_cos_scores.append(cos_scores)
                    
                all_cos_scores_tensor = torch.stack(all_cos_scores, dim=-1)
                
                print("=====Matrix-acc=====")

                
            else:
                for sub_corpus_embeddings in tqdm(all_sub_corpus_embedding_ls):    
                    
                    print(sub_corpus_embeddings.shape)
                    
                    #Compute similarites using either cosine-similarity or dot product
                    cos_scores = []
                    # for query_itr in range(len(query_embeddings)):
                    for query_itr in range(query_count):
                        # orin： curr_query_embedding_ls = query_embeddings[query_itr]
                        curr_query_embedding_ls = query_embeddings[query_itr][:2]
                        
                        if type(curr_query_embedding_ls) is list:
                            full_curr_scores_ls = []
                            
                            for sub_q_ls_idx in range(len(curr_query_embedding_ls)):
                                curr_query_embedding = curr_query_embedding_ls[sub_q_ls_idx]
                                curr_scores = 1
                                
                                # if len(sub_corpus_embeddings.shape) == 2 and sub_corpus_embeddings.shape[0] > 1:
                                if curr_query_embedding.shape[0] == 1:
                                    if self.is_img_retrieval:
                                        curr_scores_ls = self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings[-1].to(device))
                                    else:
                                        curr_scores_ls = torch.max(self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device)), dim=-1)[0]    
                                    curr_scores = curr_scores_ls
                                    full_curr_scores_ls.append(curr_scores.item())
                                    continue
                                # else:
                                    # curr_scores_ls = torch.max(self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device)), dim=-1)[0]
                                if self.algebra_method == one or self.algebra_method == three:
                                    curr_scores_ls = self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device))#, dim=-1)
                                elif self.algebra_method == two:
                                    if self.is_img_retrieval:
                                        if len(sub_corpus_embeddings) <= 1:
                                            curr_scores_ls = torch.tensor([0], device= device)
                                        else:
                                            curr_scores_ls = torch.max(self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings[0:-1].to(device)), dim=-1)[0]
                                    else:
                                        curr_scores_ls = torch.max(self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device)), dim=-1)[0]
                                    
                                    # curr_scores_ls_max_id = torch.argmax(self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device)), dim=-1)
                                else:
                                    curr_scores_ls = self.compute_dependency_aware_sim_score0(curr_query_embedding, sub_corpus_embeddings, corpus_idx, score_function, grouped_sub_q_ids_ls, sub_q_ls_idx, device, bboxes_overlap_ls, query_itr)
                                        # curr_scores_ls2 = torch.max(self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings[0:-1].to(device)), dim=-1)[0]
                                        
                                # else:    
                                #     curr_scores_ls = self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device))
                                
                                # whole_img_sim = self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings[-1].to(device)).view(-1)
                                # curr_scores = torch.prod(curr_scores_ls, dim=0)
                                # for conj_id in range(len(curr_scores_ls)):
                                #     curr_scores *= curr_scores_ls[conj_id]
                                # full_curr_scores += curr_scores
                                if self.algebra_method == one:
                                    curr_scores = torch.max(torch.prod(curr_scores_ls, dim=0))
                                    full_curr_scores_ls.append(curr_scores.item())
                                elif self.algebra_method == three:
                                    curr_scores = torch.max(torch.sum(curr_scores_ls, dim=0))
                                    # curr_scores = torch.max(torch.max(curr_scores_ls, dim=0))
                                    full_curr_scores_ls.append(curr_scores.item())
                                elif self.algebra_method == two:
                                    if self.prob_agg == "prod":
                                        curr_scores_ls[curr_scores_ls < 0] = 0
                                        curr_scores = torch.prod(curr_scores_ls)
                                    else:
                                        curr_scores = torch.sum(curr_scores_ls)
                                    # curr_scores = torch.sum(curr_scores_ls)
                                    # curr_scores = torch.sum(curr_scores_ls)
                                    full_curr_scores_ls.append(curr_scores.item())
                                else:
                                    curr_scores = curr_scores_ls
                                    full_curr_scores_ls.append(curr_scores.item())
                            
                            curr_scores = torch.tensor(full_curr_scores_ls)
                            
                        else:
                            # hotpot not 
                            curr_cos_scores = self.score_functions[score_function](curr_query_embedding_ls.to(device), sub_corpus_embeddings.to(device))
                            curr_cos_scores[torch.isnan(curr_cos_scores)] = -1
                            curr_scores = curr_cos_scores.squeeze(0)
                            if len(curr_scores) > 1:
                                curr_scores = torch.max(curr_scores, dim=-1)[0]
                            curr_scores = curr_scores.view(-1)
                            
                        cos_scores.append(curr_scores)
                        
                    cos_scores = torch.stack(cos_scores)
                    all_cos_scores.append(cos_scores)
                    
                    corpus_idx += 1
                all_cos_scores_tensor = torch.stack(all_cos_scores, dim=-1)

           
            
            # del all_cos_scores
            # gc.collect()
            

            # if dataset_name == "webis-touche2020":
            #     torch.save(all_cos_scores_tensor,"/data2/wuyinjun/output/all_cos_scores_tensor.pkl")
            if sparse_sim_scores is not None:
                all_cos_scores_tensor = torch.cat([all_cos_scores_tensor, sparse_sim_scores.to(device).unsqueeze(1)], dim=1)
            
            # all_cos_scores_tensor = torch.max(all_cos_scores_tensor, dim=1)[0]
            if self.prob_agg == "prod":
                
                all_cos_scores_tensor = all_cos_scores_tensor/torch.sum(all_cos_scores_tensor, dim=-1, keepdim=True)
                all_cos_scores_tensor = torch.mean(all_cos_scores_tensor, dim=1)
            else:
                if dataset_name == "trec-covid":
                    all_cos_scores_tensor = all_cos_scores_tensor/torch.sum(all_cos_scores_tensor, dim=-1, keepdim=True)
                    all_cos_scores_tensor = torch.mean(all_cos_scores_tensor, dim=1)
                else:
                    all_cos_scores_tensor = all_cos_scores_tensor/torch.sum(all_cos_scores_tensor, dim=-1, keepdim=True)
                    all_cos_scores_tensor = torch.max(all_cos_scores_tensor, dim=1)[0]
            # print(all_cos_scores_tensor)
            
           
        
        else:
            all_cos_scores_tensor_ls = []
            for query_itr in tqdm(range(query_count)):
                curr_query_embedding = query_embeddings[query_itr].view(1,-1)
                curr_score = self.score_functions[score_function](curr_query_embedding.to(device), all_sub_corpus_embedding_ls.to(device)).cpu()
                all_cos_scores_tensor_ls.append(curr_score)
            
            all_cos_scores_tensor = torch.cat(all_cos_scores_tensor_ls)
            if sparse_sim_scores is not None:
                all_cos_scores_tensor = torch.stack([all_cos_scores_tensor, sparse_sim_scores], dim=1)
                all_cos_scores_tensor = all_cos_scores_tensor/torch.sum(all_cos_scores_tensor, dim=-1, keepdim=True)
                all_cos_scores_tensor = torch.mean(all_cos_scores_tensor, dim=1)
            # print(all_cos_scores_tensor.shape)
        #Get top-k values
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(all_cos_scores_tensor, min(top_k+1, len(all_cos_scores_tensor[0])), dim=1, largest=True)#, sorted=return_sorted)
        cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
        cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
        
        # for query_itr in range(len(query_embeddings)):
        for query_itr in range(query_count):
            query_id = query_ids[query_itr]                  
            for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                corpus_id = corpus_ids[sub_corpus_id]
                # if corpus_id != query_id:
                self.results[query_id][corpus_id] = score
        
        return self.results, all_sub_corpus_embedding_ls

    # @profile
    def search_by_clusters(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict, 
               top_k: List[int], 
               score_function: str,
               return_sorted: bool = False, 
               query_negations: List=None, all_sub_corpus_embedding_ls=None, query_embeddings=None, query_count=10, device = 'cuda', clustering_info=None, topk_embs = 500,
               bboxes_overlap_ls=None,grouped_sub_q_ids_ls=None,
               **kwargs) -> Dict[str, Dict[str, float]]:
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
        
        
        assert clustering_info is not None
        cluster_sub_X_ls, cluster_centroid_tensor, cluster_sample_count_ls, cluster_unique_sample_ids_ls, cluster_sample_ids_ls, cluster_sub_X_cat_patch_ids_ls, clustering_nbs_mappings = clustering_info
        
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        # logger.info("Encoding Queries...")
        # if queries is not None:
        #     query_ids = list(queries.keys())
        # else:
        
        assert query_embeddings is not None
        if query_embeddings is not None:
            query_ids = [str(idx+1) for idx in list(range(len(query_embeddings)))]
        else:
            raise ValueError("Either queries or query_embeddings must be provided!")
        self.results = {qid: {} for qid in query_ids}
        # if queries is not None:
        #     queries = [queries[qid] for qid in query_ids]
        # if query_negations is not None:
        #     query_negations = [query_negations[qid] if qid in query_negations else None for qid in query_ids]
        
        
        # if query_embeddings is None:
        #     query_embeddings=[]
        #     for idx in range(len(queries)):
        #         curr_query = queries[idx]
        #         if type(curr_query) is str:
        #             curr_query_embedding_ls = self.model.encode_queries(
        #                 curr_query, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
        #         elif type(curr_query) is list:
        #             curr_query_embedding_ls = []
        #             for k in range(len(curr_query)):
        #                 curr_conjunct = []
        #                 for j in range(len(curr_query[k])):
        #                     qe = self.model.encode_queries(
        #                         curr_query[k][j], batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
        #                     curr_conjunct.append(qe)
        #                 curr_query_embedding_ls.append(curr_conjunct)
        #         query_embeddings.append(curr_query_embedding_ls)
          
        logger.info("Sorting Corpus by document length (Longest first)...")

        # if corpus is not None:
        #     corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        #     corpus = [corpus[cid] for cid in corpus_ids]
        # else:
        assert all_sub_corpus_embedding_ls is not None
        if all_sub_corpus_embedding_ls is not None:
            corpus_ids = [str(idx+1) for idx in list(range(len(all_sub_corpus_embedding_ls)))]
        else:
            raise ValueError("Either corpus or all_sub_corpus_embedding_ls must be provided!")

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        if query_count < 0:
            query_count = len(query_embeddings)

        # all_cos_scores = []
        # if all_sub_corpus_embedding_ls is None:
        #     itr = range(0, len(corpus), self.corpus_chunk_size)
        #     all_sub_corpus_embedding_ls = []
        #     for batch_num, corpus_start_idx in enumerate(itr):
        #         logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
        #         corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

        #         #Encode chunk of corpus    
        #         sub_corpus_embeddings = self.model.encode_corpus(
        #             corpus[corpus_start_idx:corpus_end_idx],
        #             batch_size=self.batch_size,
        #             show_progress_bar=self.show_progress_bar, 
        #             convert_to_tensor = self.convert_to_tensor
        #             )
                
        #         all_sub_corpus_embedding_ls.append(sub_corpus_embeddings)
        
        all_cos_scores_tensor = torch.zeros((len(all_sub_corpus_embedding_ls), len(query_embeddings[0]),  query_count), device=device)
        
        topk_embs = min(topk_embs, len(all_sub_corpus_embedding_ls))
        
        for query_itr in tqdm(range(query_count)):
            curr_query_embedding_ls = query_embeddings[query_itr]
            if type(curr_query_embedding_ls) is list:                                
                for sub_query_itr in range(len(curr_query_embedding_ls)):
                    curr_query_embedding = curr_query_embedding_ls[sub_query_itr]
                    
                    curr_scores_mat = self.score_functions[score_function](curr_query_embedding.to(device), cluster_centroid_tensor.to(device))
                    
                    # for sub_q_idx in range(curr_scores_mat.shape[0]):
                    #     curr_scores *= curr_scores_mat[sub_q_idx]
                    # 
                    if self.algebra_method == one or self.algebra_method == three:
                        if self.algebra_method == one:
                            curr_scores = torch.prod(curr_scores_mat, dim=0)
                        else:
                            curr_scores = torch.sum(curr_scores_mat, dim=0)
                        
                        
                        topk_cluster_ids = torch.argsort(curr_scores, descending=True)
                        
                        # cluster_sample_cum_count = torch.cumsum(cluster_sample_count_ls[topk_cluster_ids])
                        
                        # covered_sample_id_set = set()
                        
                        # # if self.algebra_method == two:
                        
                        # for cluster_id in topk_cluster_ids:
                        #     curr_cluster_sample_ids = cluster_unique_sample_ids_ls[cluster_id]
                            
                        #     # if query_itr == 0 and 0 in curr_cluster_sample_ids:
                        #     #     print()
                        
                        #     curr_scores_mat_curr_cluster = self.score_functions[score_function](curr_query_embedding.to(device), cluster_sub_X_ls[cluster_id].to(device))
                            
                        #     curr_scores_mat_curr_cluster = torch.prod(curr_scores_mat_curr_cluster, dim=0)
                            
                        #     all_cos_scores_tensor[curr_cluster_sample_ids, sub_query_itr, query_itr] = (curr_scores_mat_curr_cluster > all_cos_scores_tensor[curr_cluster_sample_ids, sub_query_itr, query_itr])*curr_scores_mat_curr_cluster \
                        #         + (curr_scores_mat_curr_cluster <= all_cos_scores_tensor[curr_cluster_sample_ids, sub_query_itr, query_itr])*all_cos_scores_tensor[curr_cluster_sample_ids, sub_query_itr, query_itr]
                            
                        #     covered_sample_id_set.update(curr_cluster_sample_ids.tolist())
                            
                        #     # curr_cluster_sample_ids = torch.tensor(list(set(curr_cluster_sample_ids.tolist()).difference(covered_sample_id_set)))
                        #     # if len(curr_cluster_sample_ids) > 0:
                        #     #     all_cos_scores_tensor[curr_cluster_sample_ids, sub_query_itr, query_itr] = curr_scores[cluster_id]
                                
                        #     #     covered_sample_id_set.update(curr_cluster_sample_ids.tolist())
                        #     if len(covered_sample_id_set) >= topk_embs:
                        #         break
                        
                        sample_to_cat_patch_idx_mappings = dict()
                        for cluster_id in topk_cluster_ids:
                            curr_cat_patch_ids_mappings = cluster_sub_X_cat_patch_ids_ls[cluster_id]
                            for sample_id in curr_cat_patch_ids_mappings:
                                if sample_id not in sample_to_cat_patch_idx_mappings:
                                    sample_to_cat_patch_idx_mappings[sample_id] = []
                                sample_to_cat_patch_idx_mappings[sample_id].extend(curr_cat_patch_ids_mappings[sample_id])
                            
                            if len(sample_to_cat_patch_idx_mappings) >= topk_embs:
                                break
                                    
                        for sample_id in sample_to_cat_patch_idx_mappings:
                                # sample_id = int(sample_id)
                            # curr_sample_sub_x = []
                            # for cluster_idx in merged_sample_to_cluster_idx_mappings[sample_id]:
                            #     curr_sample_sub_x.append(cluster_sub_X_ls[cluster_idx][sample_id])
                            
                            patch_ids = torch.tensor(list(sample_to_cat_patch_idx_mappings[sample_id]))
                            curr_sample_sub_X_tensor = all_sub_corpus_embedding_ls[sample_id][patch_ids].to(device)
                            
                            # curr_sample_sub_X_tensor2 = torch.cat(curr_sample_sub_x).to(device)
                            cos_scores = torch.max(torch.prod(self.score_functions[score_function](curr_sample_sub_X_tensor, curr_query_embedding.to(device)), dim=-1))
                            all_cos_scores_tensor[sample_id, sub_query_itr, query_itr] = cos_scores
                            
                        
                        
                    elif self.algebra_method == two or self.algebra_method == four:   
                        sorted_scores, sorted_indices = torch.sort(curr_scores_mat, dim=-1, descending=True)
                        # local_sim_array = torch.ones(len(all_sub_corpus_embedding_ls), device=device)
                        # local_visited_times_tensor = torch.zeros([len(all_sub_corpus_embedding_ls), sorted_scores.shape[0]], device=device)
                        # for cluster_id in range(sorted_indices.shape[1]):
                        #     curr_scores = sorted_scores[:,cluster_id]
                        #     for sub_q_idx in range(len(curr_scores)):
                        #         curr_cluster_idx = sorted_indices[sub_q_idx,cluster_id]
                        #         curr_cluster_sample_ids = cluster_unique_sample_ids_ls[curr_cluster_idx].to(device)
                        #         # if 0 in curr_cluster_sample_ids and query_itr == 0 and sub_query_itr == 0:
                        #         #     print()
                        #         selected_rid = (local_visited_times_tensor[curr_cluster_sample_ids,sub_q_idx] == 0)
                        #         curr_cluster_sample_ids = curr_cluster_sample_ids[selected_rid]
                        #         curr_sub_X  = cluster_sub_X_ls[curr_cluster_idx].to(device)[selected_rid]
                        #         curr_sim_score=self.score_functions[score_function](curr_query_embedding[sub_q_idx].to(device), curr_sub_X)
                                
                                
                        #         local_sim_array[curr_cluster_sample_ids] *= curr_sim_score.view(-1) #curr_scores[sub_q_idx]
                        #         local_visited_times_tensor[curr_cluster_sample_ids, sub_q_idx] = 1
                        #     if torch.sum(torch.sum(local_visited_times_tensor, dim=-1) >= len(curr_scores)) >= topk_embs:
                        #         break
                        sample_to_cat_patch_idx_ls = [dict()]*curr_query_embedding.shape[0]
                        full_sample_to_cluster_idx_mappings = dict()
                        # sample_to_cluster_idx_ls = [dict()]*curr_query_embedding.shape[0]
                        # if self.algebra_method == two:
                        for cluster_id in range(sorted_indices.shape[1]):
                            common_sample_ids = set()
                            for sub_q_idx in range(curr_query_embedding.shape[0]):
                                curr_cluster_idx = sorted_indices[sub_q_idx,cluster_id].item()
                                # curr_cluster_sample_ids = cluster_sample_ids_ls[curr_cluster_idx]
                                # curr_sample_idx_sub_X_mappings  = cluster_sub_X_ls[curr_cluster_idx]
                                curr_cat_patch_ids_mappings = cluster_sub_X_cat_patch_ids_ls[curr_cluster_idx]
                                for sample_id in curr_cat_patch_ids_mappings:
                                    if sample_id not in sample_to_cat_patch_idx_ls[sub_q_idx]:
                                        # sample_to_cluster_idx_ls[sub_q_idx][sample_id] = []
                                        sample_to_cat_patch_idx_ls[sub_q_idx][sample_id] = []# sample_to_sub_X_mappings_ls[sub_q_idx][sample_id].append(curr_sample_idx_sub_X_mappings[sample_id])
                                    # sample_to_cluster_idx_ls[sub_q_idx][sample_id].append(curr_cluster_idx)
                                    sample_to_cat_patch_idx_ls[sub_q_idx][sample_id].extend(curr_cat_patch_ids_mappings[sample_id])
                                if sub_q_idx == 0:
                                    common_sample_ids = set(sample_to_cat_patch_idx_ls[sub_q_idx].keys())
                                else:
                                    common_sample_ids = common_sample_ids.intersection(set(sample_to_cat_patch_idx_ls[sub_q_idx].keys()))
                                    
                            if self.algebra_method == four and curr_query_embedding.shape[0] > 1:
                                if grouped_sub_q_ids_ls[query_itr] is None:
                                    curr_grouped_sub_q_ids_ls = [list(range(curr_query_embedding.shape[0]))]
                                else:
                                    curr_grouped_sub_q_ids_ls = grouped_sub_q_ids_ls[query_itr][sub_query_itr]
                                
                                
                                valid_count, valid_samples_to_patch_ids_mappings = self.check_nbs_info(curr_query_embedding, bboxes_overlap_ls, curr_grouped_sub_q_ids_ls, common_sample_ids, sample_to_cat_patch_idx_ls)
                            else:
                                valid_count = len(common_sample_ids)
                            
                            if valid_count >= topk_embs:
                                break
                        # else:
                        #     common_sample_ids = set()
                        #     for cluster_id in range(sorted_indices.shape[1]):
                        #         for sub_q_g_idx_ls in range(len(grouped_sub_q_ids_ls)):
                        #             for sub_q_idx_idx in range(len(sub_q_g_idx_ls)):
                        #                 sub_q_idx = sub_q_g_idx_ls[sub_q_idx_idx]
                        #                 if sub_q_idx_idx == 0:
                        #                     curr_cluster_idx = sorted_indices[sub_q_idx,cluster_id].item()
                        #                     curr_cat_patch_ids_mappings = cluster_sub_X_cat_patch_ids_ls[curr_cluster_idx]
                        #                     # for sample_id in curr_cat_patch_ids_mappings:
                        #                     #     if sample_id not in sample_to_cat_patch_idx_ls[sub_q_idx]:
                        #                     #         # sample_to_cluster_idx_ls[sub_q_idx][sample_id] = []
                        #                     #         sample_to_cat_patch_idx_ls[sub_q_idx][sample_id] = []# sample_to_sub_X_mappings_ls[sub_q_idx][sample_id].append(curr_sample_idx_sub_X_mappings[sample_id])
                        #                     #     # sample_to_cluster_idx_ls[sub_q_idx][sample_id].append(curr_cluster_idx)
                        #                     #     sample_to_cat_patch_idx_ls[sub_q_idx][sample_id].extend(curr_cat_patch_ids_mappings[sample_id])
                        #                 else:
                        #                     curr_cluster_idx = clustering_nbs_mappings[]
                                            
                                        
                                            
                                            
                                            
                                    # cluster_id_ls = sorted_indices[0:topk_embs]
                            
                        # merged_sample_to_cluster_idx_mappings = dict()
                        
                        merged_sample_to_cat_patch_idx_mappings = dict()
                        for sample_id in common_sample_ids:
                            for sub_q_idx in range(len(sample_to_cat_patch_idx_ls)):
                                if sub_q_idx == 0:
                                    # merged_sample_to_cluster_idx_mappings[sample_id] = set(sample_to_cluster_idx_ls[sub_q_idx][sample_id])
                                    merged_sample_to_cat_patch_idx_mappings[sample_id] = set(sample_to_cat_patch_idx_ls[sub_q_idx][sample_id])
                                else:
                                    # merged_sample_to_cluster_idx_mappings[sample_id] = merged_sample_to_cluster_idx_mappings[sample_id].union(set(sample_to_cluster_idx_ls[sub_q_idx][sample_id]))
                                    merged_sample_to_cat_patch_idx_mappings[sample_id] = merged_sample_to_cat_patch_idx_mappings[sample_id].union(set(sample_to_cat_patch_idx_ls[sub_q_idx][sample_id]))
                        if self.algebra_method == two:    
                            for sample_id in common_sample_ids:
                                # sample_id = int(sample_id)
                                # curr_sample_sub_x = []
                                # for cluster_idx in merged_sample_to_cluster_idx_mappings[sample_id]:
                                #     curr_sample_sub_x.append(cluster_sub_X_ls[cluster_idx][sample_id])
                                
                                patch_ids = torch.tensor(list(merged_sample_to_cat_patch_idx_mappings[sample_id]))
                                curr_sample_sub_X_tensor = all_sub_corpus_embedding_ls[sample_id][patch_ids].to(device)
                                
                                # curr_sample_sub_X_tensor2 = torch.cat(curr_sample_sub_x).to(device)
                                if self.prob_agg == "prod":
                                    # cos_scores = torch.prod(self.score_functions[score_function](curr_sample_sub_X_tensor, curr_query_embedding.to(device)), dim=0)
                                    curr_cos_scores0 = torch.max(self.score_functions[score_function](curr_sample_sub_X_tensor, curr_query_embedding.to(device)), dim=0)[0]
                                    curr_cos_scores0[curr_cos_scores0 < 0] = 0
                                    cos_scores = torch.prod(curr_cos_scores0)
                                else:
                                    cos_scores = torch.sum(torch.max(self.score_functions[score_function](curr_sample_sub_X_tensor, curr_query_embedding.to(device)), dim=0)[0])
                                all_cos_scores_tensor[sample_id, sub_query_itr, query_itr] = cos_scores
                        else:
                            # for sub_q_ls_idx in range(len(curr_query_embedding)):
                            # if grouped_sub_q_ids_ls[query_itr] is not None:
                            #     curr_grouped_sub_q_ids_ls = grouped_sub_q_ids_ls[query_itr][sub_query_itr]
                            # else:
                            #     curr_grouped_sub_q_ids_ls = [list(range(curr_query_embedding.shape[0]))]
                            
                            for sample_id in common_sample_ids:
                                if curr_query_embedding.shape[0] == 1:
                                    if self.is_img_retrieval:
                                        curr_scores_ls = self.score_functions[score_function](curr_query_embedding.to(device), all_sub_corpus_embedding_ls[sample_id][-1].to(device))
                                    else:
                                        curr_scores_ls = torch.max(self.score_functions[score_function](curr_query_embedding.to(device), all_sub_corpus_embedding_ls[sample_id][-1].to(device)), dim=-1)[0]   

                                        # curr_scores_ls = self.score_functions[score_function](curr_query_embedding.to(device), all_sub_corpus_embedding_ls[sample_id][-1].to(device))
                                    curr_scores = curr_scores_ls
                                    all_cos_scores_tensor[sample_id, sub_query_itr, query_itr] = curr_scores
                                    continue
                                if sample_id not in valid_samples_to_patch_ids_mappings:
                                    continue
                                valid_patch_ids = valid_samples_to_patch_ids_mappings[sample_id]
                                # patch_ids = torch.tensor(list(merged_sample_to_cat_patch_idx_mappings[sample_id]))
                                cos_scores = self.compute_dependency_aware_sim_score(curr_query_embedding, all_sub_corpus_embedding_ls[sample_id], sample_id, score_function, grouped_sub_q_ids_ls, sub_query_itr, device, bboxes_overlap_ls, query_itr, valid_patch_ids=valid_patch_ids)
                                all_cos_scores_tensor[sample_id, sub_query_itr, query_itr] = cos_scores
                            
                            
                            # for curr_grouped_sub_q_ids in curr_grouped_sub_q_ids_ls:
                                
                                
                                # for sample_id in common_sample_ids:
                                #     # for sub_q_idx in range(sample_to_sub_X_mappings_ls):
                                #         cos_scores = torch.prod(torch.max(self.score_functions(torch.cat(sample_to_cluster_idx_ls[0][sample_id]), curr_query_embedding.to(device)), dim=0))
                                #         all_cos_scores_tensor[sample_id, sub_q_idx, query_itr] = cos_scores
                    else:
                        common_sample_ids = set()
                        for sub_q_idx in range(len(curr_query_embedding)):
                            sub_curr_scores_mat = curr_scores_mat[sub_q_idx].view(-1)
                            sub_sorted_scores, sub_sorted_indices = torch.sort(sub_curr_scores_mat, descending=True)
                            for cluster_id in range(sub_sorted_indices.shape[1]):
                                
                                # for sub_q_idx in range(curr_query_embedding.shape[0]):
                                curr_cluster_idx = sub_sorted_indices[cluster_id].item()
                                curr_cat_patch_ids_mappings = cluster_sub_X_cat_patch_ids_ls[curr_cluster_idx]
                                
                                for sample_id in curr_cat_patch_ids_mappings:
                                    if sample_id not in sample_to_cat_patch_idx_ls[sub_q_idx]:
                                        # sample_to_cluster_idx_ls[sub_q_idx][sample_id] = []
                                        sample_to_cat_patch_idx_ls[sub_q_idx][sample_id] = []# sample_to_sub_X_mappings_ls[sub_q_idx][sample_id].append(curr_sample_idx_sub_X_mappings[sample_id])
                                    # sample_to_cluster_idx_ls[sub_q_idx][sample_id].append(curr_cluster_idx)
                                    sample_to_cat_patch_idx_ls[sub_q_idx][sample_id].extend(curr_cat_patch_ids_mappings[sample_id])
                        # all_cos_scores_tensor[torch.sum(local_visited_times_tensor, dim=-1) >= len(curr_scores), sub_query_itr, query_itr] = local_sim_array

        all_cos_scores_tensor = all_cos_scores_tensor/torch.sum(all_cos_scores_tensor, dim=0, keepdim=True)
        # all_cos_scores_tensor = torch.max(all_cos_scores_tensor, dim=1)[0]
        all_cos_scores_tensor = torch.mean(all_cos_scores_tensor, dim=1)
        #Get top-k values
        all_cos_scores_tensor = all_cos_scores_tensor.t()
        cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(all_cos_scores_tensor, min(top_k+1, len(all_cos_scores_tensor[0])), dim=1, largest=True)#, sorted=return_sorted)
        cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
        cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
        
        # for query_itr in range(len(query_embeddings)):
        for query_itr in range(query_count):
            query_id = query_ids[query_itr]                  
            for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                corpus_id = corpus_ids[sub_corpus_id]
                # if corpus_id != query_id:
                self.results[query_id][corpus_id] = score
        
        return self.results, all_sub_corpus_embedding_ls
    
    # def search_in_disk(self, 
    #            corpus: Dict[str, Dict[str, str]], store_path: str,
    #            queries: Dict, 
    #            top_k: List[int], 
    #            score_function: str,
    #            return_sorted: bool = False, 
    #            query_negations: List=None, all_sub_corpus_embedding_ls=None, query_embeddings=None, query_count=10, device = 'cuda',corpus_count=1,
    #            **kwargs) -> Dict[str, Dict[str, float]]:
    #     #Create embeddings for all queries using model.encode_queries()
    #     #Runs semantic search against the corpus embeddings
    #     #Returns a ranked list with the corpus ids
        
    #     if score_function not in self.score_functions:
    #         raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
    #     logger.info("Encoding Queries...")
    #     if queries is not None:
    #         query_ids = list(queries.keys())
    #     else:
    #         if query_embeddings is not None:
    #             query_ids = [str(idx+1) for idx in list(range(len(query_embeddings)))]
    #         else:
    #             raise ValueError("Either queries or query_embeddings must be provided!")
    #     self.results = {qid: {} for qid in query_ids}
    #     if queries is not None:
    #         queries = [queries[qid] for qid in query_ids]
    #     if query_negations is not None:
    #         query_negations = [query_negations[qid] if qid in query_negations else None for qid in query_ids]
        
    #     if query_embeddings is None:
    #         query_embeddings=[]
    #         for idx in range(len(queries)):
    #             curr_query = queries[idx]
    #             if type(curr_query) is str:
    #                 curr_query_embedding_ls = self.model.encode_queries(
    #                     curr_query, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
    #             elif type(curr_query) is list:
    #                 curr_query_embedding_ls = []
    #                 for k in range(len(curr_query)):
    #                     curr_conjunct = []
    #                     for j in range(len(curr_query[k])):
    #                         qe = self.model.encode_queries(
    #                             curr_query[k][j], batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
    #                         curr_conjunct.append(qe)
    #                     curr_query_embedding_ls.append(curr_conjunct)
    #             query_embeddings.append(curr_query_embedding_ls)
          
    #     logger.info("Sorting Corpus by document length (Longest first)...")

        

    #     corpus_ids = [str(idx + 1) for idx in range(corpus_count)]

    #     # if corpus is not None:
    #     #     corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
    #     #     corpus = [corpus[cid] for cid in corpus_ids]
    #     # else:
    #     #     if all_sub_corpus_embedding_ls is not None:
    #     #         corpus_ids = [str(idx+1) for idx in list(range(len(all_sub_corpus_embedding_ls)))]
    #     #     else:
    #     #         raise ValueError("Either corpus or all_sub_corpus_embedding_ls must be provided!")

    #     logger.info("Encoding Corpus in batches... Warning: This might take a while!")
    #     logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

    #     if query_count < 0:
    #         query_count = len(query_embeddings)

    #     all_cos_scores = []
    #     # if all_sub_corpus_embedding_ls is None:
    #     #     itr = range(0, len(corpus), self.corpus_chunk_size)
    #     #     all_sub_corpus_embedding_ls = []
    #     #     for batch_num, corpus_start_idx in enumerate(itr):
    #     #         logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
    #     #         corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

    #     #         #Encode chunk of corpus    
    #     #         sub_corpus_embeddings = self.model.encode_corpus(
    #     #             corpus[corpus_start_idx:corpus_end_idx],
    #     #             batch_size=self.batch_size,
    #     #             show_progress_bar=self.show_progress_bar, 
    #     #             convert_to_tensor = self.convert_to_tensor
    #     #             )
                
    #     #         all_sub_corpus_embedding_ls.append(sub_corpus_embeddings)
        
    #     # all_sub_corpus_embedding_ls = [item.to(device) for item in all_sub_corpus_embedding_ls]
        
    #     # for sub_corpus_embeddings in tqdm(all_sub_corpus_embedding_ls):
    #     for corpus_id in tqdm(corpus_ids):
    #         # sub_folder = os.path.join(store_path, corpus_id)
    #         sub_corpus_embeddings = load_single_embeddings(store_path, int(corpus_id)-1)

    #         #Compute similarites using either cosine-similarity or dot product
    #         cos_scores = []
    #         # for query_itr in range(len(query_embeddings)):
    #         for query_itr in range(query_count):
    #             curr_query_embedding_ls = query_embeddings[query_itr]
    #             if type(curr_query_embedding_ls) is list:
    #                 full_curr_scores_ls = []
    #                 # for conj_id in range(len(curr_query_embedding)):
    #                 #     curr_cos_scores_ls = self.score_functions[score_function](torch.stack(curr_query_embedding[conj_id]), sub_corpus_embeddings)
    #                 #     if query_negations is not None and query_negations[query_itr] is not None:
    #                 #         curr_query_negations = torch.tensor(query_negations[query_itr])
    #                 #         curr_cos_scores_ls[curr_query_negations == 1] =  - curr_cos_scores_ls[curr_query_negations == 1]

    #                 #     curr_cos_scores_ls[torch.isnan(curr_cos_scores_ls)] = -1

    #                 #     curr_cos_scores = 1
    #                 #     for idx in range(len(curr_cos_scores_ls)):
    #                 #         curr_cos_scores *= curr_cos_scores_ls[idx]
    #                 #     curr_scores += curr_cos_scores
    #                 for curr_query_embedding in curr_query_embedding_ls:
    #                     curr_scores = 1
    #                     if len(sub_corpus_embeddings.shape) == 2 and sub_corpus_embeddings.shape[0] > 1:
    #                         curr_scores_ls = torch.max(self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device)), dim=-1)[0]
    #                     else:    
    #                         curr_scores_ls = self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device))
    #                     for conj_id in range(len(curr_scores_ls)):
    #                         curr_scores *= curr_scores_ls[conj_id]
    #                     # full_curr_scores += curr_scores
    #                     full_curr_scores_ls.append(curr_scores)
                    
    #                 curr_scores = torch.tensor(full_curr_scores_ls)
                    
    #             else:
    #                 # curr_cos_scores = self.score_functions[score_function](curr_query_embedding.unsqueeze(0), sub_corpus_embeddings)
    #                 curr_cos_scores = self.score_functions[score_function](curr_query_embedding_ls, sub_corpus_embeddings)
    #                 curr_cos_scores[torch.isnan(curr_cos_scores)] = -1
    #                 curr_scores = curr_cos_scores.squeeze(0)
    #                 if len(curr_scores) > 1:
    #                     curr_scores = torch.max(curr_scores, dim=-1)[0]
    #                 curr_scores = curr_scores.view(-1)
    #             cos_scores.append(curr_scores)
    #         cos_scores = torch.stack(cos_scores)
    #         all_cos_scores.append(cos_scores)
        
    #     all_cos_scores_tensor = torch.stack(all_cos_scores, dim=-1)
    #     all_cos_scores_tensor = all_cos_scores_tensor/torch.sum(all_cos_scores_tensor, dim=-1, keepdim=True)
    #     all_cos_scores_tensor = torch.max(all_cos_scores_tensor, dim=1)[0]
    #     # all_cos_scores_tensor = torch.mean(all_cos_scores_tensor, dim=1)
    #     #Get top-k values
    #     cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(all_cos_scores_tensor, min(top_k+1, len(all_cos_scores_tensor[0])), dim=1, largest=True)#, sorted=return_sorted)
    #     cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
    #     cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
        
    #     # for query_itr in range(len(query_embeddings)):
    #     for query_itr in range(query_count):
    #         query_id = query_ids[query_itr]                  
    #         for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
    #             corpus_id = corpus_ids[sub_corpus_id]
    #             # if corpus_id != query_id:
    #             self.results[query_id][corpus_id] = score
        
    #     return self.results, all_sub_corpus_embedding_ls
    
    # def search_parallel(self, 
    #            corpus: Dict[str, Dict[str, str]], 
    #            queries: Dict, 
    #            top_k: List[int], 
    #            score_function: str,
    #            return_sorted: bool = False, 
    #            query_negations: List=None, all_sub_corpus_embedding_ls=None, query_embeddings=None, query_count=10, batch_size=4, device='cuda',
    #            **kwargs) -> Dict[str, Dict[str, float]]:
    #     #Create embeddings for all queries using model.encode_queries()
    #     #Runs semantic search against the corpus embeddings
    #     #Returns a ranked list with the corpus ids
        
    #     if score_function not in self.score_functions:
    #         raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
    #     logger.info("Encoding Queries...")
    #     if queries is not None:
    #         query_ids = list(queries.keys())
    #     else:
    #         if query_embeddings is not None:
    #             query_ids = [str(idx+1) for idx in list(range(len(query_embeddings)))]
    #         else:
    #             raise ValueError("Either queries or query_embeddings must be provided!")
    #     self.results = {qid: {} for qid in query_ids}
    #     if queries is not None:
    #         queries = [queries[qid] for qid in query_ids]
    #     if query_negations is not None:
    #         query_negations = [query_negations[qid] if qid in query_negations else None for qid in query_ids]
        
    #     if query_embeddings is None:
    #         query_embeddings=[]
    #         for idx in range(len(queries)):
    #             curr_query = queries[idx]
    #             if type(curr_query) is str:
    #                 curr_query_embedding_ls = self.model.encode_queries(
    #                     curr_query, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
    #             elif type(curr_query) is list:
    #                 curr_query_embedding_ls = []
    #                 for k in range(len(curr_query)):
    #                     curr_conjunct = []
    #                     for j in range(len(curr_query[k])):
    #                         qe = self.model.encode_queries(
    #                             curr_query[k][j], batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
    #                         curr_conjunct.append(qe)
    #                     curr_query_embedding_ls.append(curr_conjunct)
    #             query_embeddings.append(curr_query_embedding_ls)
          
    #     logger.info("Sorting Corpus by document length (Longest first)...")

    #     if corpus is not None:
    #         corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
    #         corpus = [corpus[cid] for cid in corpus_ids]
    #     else:
    #         if all_sub_corpus_embedding_ls is not None:
    #             corpus_ids = [str(idx+1) for idx in list(range(len(all_sub_corpus_embedding_ls)))]
    #         else:
    #             raise ValueError("Either corpus or all_sub_corpus_embedding_ls must be provided!")

    #     logger.info("Encoding Corpus in batches... Warning: This might take a while!")
    #     logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

    #     if query_count < 0:
    #         query_count = len(query_embeddings)

    #     all_cos_scores = []
    #     if all_sub_corpus_embedding_ls is None:
    #         itr = range(0, len(corpus), self.corpus_chunk_size)
    #         all_sub_corpus_embedding_ls = []
    #         for batch_num, corpus_start_idx in enumerate(itr):
    #             logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
    #             corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

    #             #Encode chunk of corpus    
    #             sub_corpus_embeddings = self.model.encode_corpus(
    #                 corpus[corpus_start_idx:corpus_end_idx],
    #                 batch_size=self.batch_size,
    #                 show_progress_bar=self.show_progress_bar, 
    #                 convert_to_tensor = self.convert_to_tensor
    #                 )
                
    #             all_sub_corpus_embedding_ls.append(sub_corpus_embeddings)
        
    #     # query_embeddings = [[torch.cat(item).to(device) for item in items] for items in query_embeddings]
        
    #     # for sub_corpus_embeddings in tqdm(all_sub_corpus_embedding_ls):

    #         #Compute similarites using either cosine-similarity or dot product
    #     def compute_cos_scores(sub_corpus_embeddings):
    #         cos_scores = []
    #         # for query_itr in range(len(query_embeddings)):
    #         for query_itr in range(query_count):
    #             curr_query_embedding_ls = query_embeddings[query_itr]
    #             if type(curr_query_embedding_ls) is list:
    #                 full_curr_scores_ls = []
    #                 # for conj_id in range(len(curr_query_embedding)):
    #                 #     curr_cos_scores_ls = self.score_functions[score_function](torch.stack(curr_query_embedding[conj_id]), sub_corpus_embeddings)
    #                 #     if query_negations is not None and query_negations[query_itr] is not None:
    #                 #         curr_query_negations = torch.tensor(query_negations[query_itr])
    #                 #         curr_cos_scores_ls[curr_query_negations == 1] =  - curr_cos_scores_ls[curr_query_negations == 1]

    #                 #     curr_cos_scores_ls[torch.isnan(curr_cos_scores_ls)] = -1

    #                 #     curr_cos_scores = 1
    #                 #     for idx in range(len(curr_cos_scores_ls)):
    #                 #         curr_cos_scores *= curr_cos_scores_ls[idx]
    #                 #     curr_scores += curr_cos_scores
    #                 for curr_query_embedding in curr_query_embedding_ls:
    #                     curr_scores = 1
    #                     if len(sub_corpus_embeddings.shape) == 2 and sub_corpus_embeddings.shape[0] > 1:
    #                         curr_scores_ls = torch.max(self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device)), dim=-1)[0]
    #                     else:    
    #                         curr_scores_ls = self.score_functions[score_function](curr_query_embedding.to(device), sub_corpus_embeddings.to(device))
    #                     for conj_id in range(len(curr_scores_ls)):
    #                         curr_scores *= curr_scores_ls[conj_id]
    #                     # full_curr_scores += curr_scores
    #                     full_curr_scores_ls.append(curr_scores)
                    
    #                 curr_scores = torch.tensor(full_curr_scores_ls)
                    
    #             else:
    #                 # curr_cos_scores = self.score_functions[score_function](curr_query_embedding.unsqueeze(0), sub_corpus_embeddings)
    #                 curr_cos_scores = self.score_functions[score_function](curr_query_embedding_ls, sub_corpus_embeddings)
    #                 curr_cos_scores[torch.isnan(curr_cos_scores)] = -1
    #                 curr_scores = curr_cos_scores.squeeze(0)
    #                 if len(curr_scores) > 1:
    #                     curr_scores = torch.max(curr_scores, dim=-1)[0]
    #                 curr_scores = curr_scores.view(-1)
    #             cos_scores.append(curr_scores.cpu())
    #         cos_scores = torch.stack(cos_scores)
    #         return cos_scores
        
    #     class CountThread(threading.Thread):
    #         def __init__(self, param):
    #             threading.Thread.__init__(self)
    #             self.param = param
    #             self.result = None

    #         def run(self):
    #             self.result = compute_cos_scores(self.param)
        
    #     # Create a ThreadPoolExecutor with a maximum of 4 worker threads
    #     # with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        
    #     # all_cos_scores = Parallel(n_jobs=16)(delayed(compute_cos_scores)(item) for item in all_sub_corpus_embedding_ls)
        
    #     all_cos_scores = []
    #     # all_sub_corpus_embedding_ls = [item.to(device) for item in all_sub_corpus_embedding_ls]
    #     for idx in range(0, len(all_sub_corpus_embedding_ls), batch_size):
    #         all_sub_corpus_embedding_tuple_ls = all_sub_corpus_embedding_ls[idx:idx+batch_size]
    #         threads = []
    #         for sub_idx in range(len(all_sub_corpus_embedding_tuple_ls)):    
    #             thread = CountThread(all_sub_corpus_embedding_tuple_ls[sub_idx])
    #             # thread = threading.Thread(target=compute_cos_scores, args=(all_sub_corpus_embedding_tuple_ls[sub_idx],))
    #             thread.start()
    #             threads.append(thread)
    #         for thread in threads:
    #             thread.join()
    #             all_cos_scores.append(thread.result)
    #     # with multiprocessing.Pool() as pool:
    #     #     # Submit the tasks to the executor
    #     #     # The map function will apply the process_item function to each item in parallel
    #     #     all_cos_scores = list(pool.map(compute_cos_scores, all_sub_corpus_embedding_ls))
    #     # all_cos_scores.append(cos_scores)
        
    #     all_cos_scores_tensor = torch.stack(all_cos_scores, dim=-1)
    #     all_cos_scores_tensor = all_cos_scores_tensor/torch.sum(all_cos_scores_tensor, dim=-1, keepdim=True)
    #     all_cos_scores_tensor = torch.max(all_cos_scores_tensor, dim=1)[0]
    #     # all_cos_scores_tensor = torch.mean(all_cos_scores_tensor, dim=1)
    #     #Get top-k values
    #     cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(all_cos_scores_tensor, min(top_k+1, len(all_cos_scores_tensor[0])), dim=1, largest=True)#, sorted=return_sorted)
    #     cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
    #     cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
        
    #     # for query_itr in range(len(query_embeddings)):
    #     for query_itr in range(query_count):
    #         query_id = query_ids[query_itr]                  
    #         for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
    #             corpus_id = corpus_ids[sub_corpus_id]
    #             # if corpus_id != query_id:
    #             self.results[query_id][corpus_id] = score
        
    #     return self.results, all_sub_corpus_embedding_ls

    # def search2(self, 
    #            corpus: Dict[str, Dict[str, str]], 
    #            queries: Dict, 
    #            top_k: List[int], 
    #            score_function: str,
    #            return_sorted: bool = False, 
    #            query_negations: List=None, all_sub_corpus_embedding_ls=None, query_embeddings=None, query_count=10,
    #            **kwargs) -> Dict[str, Dict[str, float]]:
    #     #Create embeddings for all queries using model.encode_queries()
    #     #Runs semantic search against the corpus embeddings
    #     #Returns a ranked list with the corpus ids
        
    #     if score_function not in self.score_functions:
    #         raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
    #     logger.info("Encoding Queries...")
    #     if queries is not None:
    #         query_ids = list(queries.keys())
    #     else:
    #         if query_embeddings is not None:
    #             query_ids = [str(idx+1) for idx in list(range(len(query_embeddings)))]
    #         else:
    #             raise ValueError("Either queries or query_embeddings must be provided!")
    #     self.results = {qid: {} for qid in query_ids}
    #     if queries is not None:
    #         queries = [queries[qid] for qid in query_ids]
    #     if query_negations is not None:
    #         query_negations = [query_negations[qid] if qid in query_negations else None for qid in query_ids]
        
    #     if query_embeddings is None:
    #         query_embeddings=[]
    #         for idx in range(len(queries)):
    #             curr_query = queries[idx]
    #             if type(curr_query) is str:
    #                 curr_query_embedding_ls = self.model.encode_queries(
    #                     curr_query, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
    #             elif type(curr_query) is list:
    #                 curr_query_embedding_ls = []
    #                 for k in range(len(curr_query)):
    #                     curr_conjunct = []
    #                     for j in range(len(curr_query[k])):
    #                         qe = self.model.encode_queries(
    #                             curr_query[k][j], batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
    #                         curr_conjunct.append(qe)
    #                     curr_query_embedding_ls.append(curr_conjunct)
    #             query_embeddings.append(curr_query_embedding_ls)
          
    #     logger.info("Sorting Corpus by document length (Longest first)...")

    #     if corpus is not None:
    #         corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
    #         corpus = [corpus[cid] for cid in corpus_ids]
    #     else:
    #         if all_sub_corpus_embedding_ls is not None:
    #             corpus_ids = [str(idx+1) for idx in list(range(len(all_sub_corpus_embedding_ls)))]
    #         else:
    #             raise ValueError("Either corpus or all_sub_corpus_embedding_ls must be provided!")

    #     logger.info("Encoding Corpus in batches... Warning: This might take a while!")
    #     logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

    #     if query_count < 0:
    #         query_count = len(query_embeddings)

    #     all_cos_scores = []
    #     if all_sub_corpus_embedding_ls is None:
    #         itr = range(0, len(corpus), self.corpus_chunk_size)
    #         all_sub_corpus_embedding_ls = []
    #         for batch_num, corpus_start_idx in enumerate(itr):
    #             logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
    #             corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

    #             #Encode chunk of corpus    
    #             sub_corpus_embeddings = self.model.encode_corpus(
    #                 corpus[corpus_start_idx:corpus_end_idx],
    #                 batch_size=self.batch_size,
    #                 show_progress_bar=self.show_progress_bar, 
    #                 convert_to_tensor = self.convert_to_tensor
    #                 )
                
    #             all_sub_corpus_embedding_ls.append(sub_corpus_embeddings)
        
    #     for sub_corpus_embeddings in tqdm(all_sub_corpus_embedding_ls):

    #         #Compute similarites using either cosine-similarity or dot product
    #         cos_scores = []
    #         # for query_itr in range(len(query_embeddings)):
    #         for query_itr in range(query_count):
    #             curr_query_embedding_ls = query_embeddings[query_itr]
    #             if type(curr_query_embedding_ls) is list:
    #                 full_curr_scores_ls = []
    #                 # for conj_id in range(len(curr_query_embedding)):
    #                 #     curr_cos_scores_ls = self.score_functions[score_function](torch.stack(curr_query_embedding[conj_id]), sub_corpus_embeddings)
    #                 #     if query_negations is not None and query_negations[query_itr] is not None:
    #                 #         curr_query_negations = torch.tensor(query_negations[query_itr])
    #                 #         curr_cos_scores_ls[curr_query_negations == 1] =  - curr_cos_scores_ls[curr_query_negations == 1]

    #                 #     curr_cos_scores_ls[torch.isnan(curr_cos_scores_ls)] = -1

    #                 #     curr_cos_scores = 1
    #                 #     for idx in range(len(curr_cos_scores_ls)):
    #                 #         curr_cos_scores *= curr_cos_scores_ls[idx]
    #                 #     curr_scores += curr_cos_scores
    #                 for curr_query_embedding in curr_query_embedding_ls:
    #                     curr_scores = 1
    #                     if len(sub_corpus_embeddings.shape) == 2 and sub_corpus_embeddings.shape[0] > 1:
    #                         curr_scores_ls = torch.max(self.score_functions[score_function](torch.cat(curr_query_embedding), sub_corpus_embeddings), dim=-1)[0]
    #                     else:    
    #                         curr_scores_ls = self.score_functions[score_function](torch.cat(curr_query_embedding), sub_corpus_embeddings)
    #                     for conj_id in range(len(curr_scores_ls)):
    #                         curr_scores *= curr_scores_ls[conj_id]
    #                     # full_curr_scores += curr_scores
    #                     full_curr_scores_ls.append(curr_scores)
                    
    #                 curr_scores = torch.tensor(full_curr_scores_ls)
                    
    #             else:
    #                 # curr_cos_scores = self.score_functions[score_function](curr_query_embedding.unsqueeze(0), sub_corpus_embeddings)
    #                 curr_cos_scores = self.score_functions[score_function](curr_query_embedding_ls, sub_corpus_embeddings)
    #                 curr_cos_scores[torch.isnan(curr_cos_scores)] = -1
    #                 curr_scores = curr_cos_scores.squeeze(0)
    #                 if len(curr_scores) > 1:
    #                     curr_scores = torch.max(curr_scores, dim=-1)[0]
    #                 curr_scores = curr_scores.view(-1)
    #             cos_scores.append(curr_scores)
    #         cos_scores = torch.stack(cos_scores)
    #         all_cos_scores.append(cos_scores)
        
    #     all_cos_scores_tensor = torch.stack(all_cos_scores, dim=-1)
    #     all_cos_scores_tensor = all_cos_scores_tensor/torch.sum(all_cos_scores_tensor, dim=-1, keepdim=True)
    #     # all_cos_scores_tensor = torch.max(all_cos_scores_tensor, dim=1)[0]
    #     all_cos_scores_tensor = torch.mean(all_cos_scores_tensor, dim=1)
    #     #Get top-k values
    #     cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(all_cos_scores_tensor, min(top_k+1, len(all_cos_scores_tensor[0])), dim=1, largest=True)#, sorted=return_sorted)
    #     cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
    #     cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
        
    #     # for query_itr in range(len(query_embeddings)):
    #     for query_itr in range(query_count):
    #         query_id = query_ids[query_itr]                  
    #         for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
    #             corpus_id = corpus_ids[sub_corpus_id]
    #             # if corpus_id != query_id:
    #             self.results[query_id][corpus_id] = score
        
    #     return self.results, all_sub_corpus_embedding_ls


