import pytrec_eval
import logging
from typing import Type, List, Dict, Union, Tuple
from .search.dense import DenseRetrievalExactSearch as DRES
from .search.dense import DenseRetrievalFaissSearch as DRFS
from .search.lexical import BM25Search as BM25
from .search.sparse import SparseSearch as SS
from .custom_metrics import mrr, recall_cap, hole, top_k_accuracy

logger = logging.getLogger(__name__)

class EvaluateRetrieval:
    
    def __init__(self, retriever: Union[Type[DRES], Type[DRFS], Type[BM25], Type[SS]] = None, k_values: List[int] = [1,3,5,10,100,1000], score_function: str = "cos_sim"):
        self.k_values = k_values
        self.top_k = max(k_values)
        self.retriever = retriever
        self.score_function = score_function
            
    def retrieve(self, corpus: Dict[str, Dict[str, str]], queries: Dict, query_negations:Dict=None, all_corpus_embedding_ls=None, all_sub_corpus_embedding_ls=None,query_embeddings=None, query_count=10, parallel=False, in_disk=False, store_path=None, use_clustering=False,clustering_topk=500, **kwargs) -> Dict[str, Dict[str, float]]:
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")
        if not parallel:
            if not in_disk:
                if not use_clustering:
                    return self.retriever.search(corpus, queries, self.top_k, self.score_function, query_negations=query_negations, all_sub_corpus_embedding_ls=all_sub_corpus_embedding_ls, query_embeddings=query_embeddings, query_count=query_count,**kwargs)
                else:
                    return self.retriever.search_by_clusters(corpus, queries, self.top_k, self.score_function, query_negations=query_negations, all_sub_corpus_embedding_ls=all_sub_corpus_embedding_ls, query_embeddings=query_embeddings, query_count=query_count, topk_embs=clustering_topk, **kwargs)
            else:
                assert store_path is not None
                return self.retriever.search_in_disk(corpus, store_path, queries, self.top_k, self.score_function, query_negations=query_negations, all_sub_corpus_embedding_ls=None, query_embeddings=query_embeddings, query_count=query_count,**kwargs)
        else:
            if not in_disk:
                return self.retriever.search_parallel(corpus, queries, self.top_k, self.score_function, query_negations=query_negations, all_corpus_embedding_ls=all_corpus_embedding_ls, all_sub_corpus_embedding_ls=all_sub_corpus_embedding_ls, query_embeddings=query_embeddings, query_count=query_count,**kwargs)
            else:
                return self.retriever.search_in_disk_parallel(corpus, store_path, queries, self.top_k, self.score_function, query_negations=query_negations, all_sub_corpus_embedding_ls=None, query_embeddings=query_embeddings, query_count=query_count,**kwargs)
    def rerank(self, 
            corpus: Dict[str, Dict[str, str]], 
            queries: Dict[str, str],
            results: Dict[str, Dict[str, float]],
            top_k: int) -> Dict[str, Dict[str, float]]:
    
        new_corpus = {}
    
        for query_id in results:
            if len(results[query_id]) > top_k:
                for (doc_id, _) in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
                    new_corpus[doc_id] = corpus[doc_id]
            else:
                for doc_id in results[query_id]:
                    new_corpus[doc_id] = corpus[doc_id]
                    
        return self.retriever.search(new_corpus, queries, top_k, self.score_function)

    @staticmethod
    def evaluate(qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]], 
                 k_values: List[int],
                 ignore_identical_ids: bool=True) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        
        if ignore_identical_ids:
            logging.info('For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}
        
        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0
        
        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(results)
        
        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]
        
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)
        
        for eval in [ndcg, _map, recall, precision]:
            logging.info("\n")
            for k in eval.keys():
                logging.info("{}: {:.4f}".format(k, eval[k]))

        return ndcg, _map, recall, precision
    
    @staticmethod
    def evaluate_custom(qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]], 
                 k_values: List[int], metric: str) -> Tuple[Dict[str, float]]:
        
        if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
            return mrr(qrels, results, k_values)
        
        elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
            return recall_cap(qrels, results, k_values)
        
        elif metric.lower() in ["hole", "hole@k"]:
            return hole(qrels, results, k_values)
        
        elif metric.lower() in ["acc", "top_k_acc", "accuracy", "accuracy@k", "top_k_accuracy"]:
            return top_k_accuracy(qrels, results, k_values)