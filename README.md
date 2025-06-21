使用 `main_img.py` 或 `main_text.py` 来运行图像/文本任务。

示例：
```bash
python3 main_img.py --dataset_name mscoco --data_path ./data/ --query_count 100 --total_count 40000 --img_concept --query_concept --patch_count=32 --clustering_topk=5000 --parallel --search_by_cluster
```

`--query_count` 和 `--total_count` 是（抽取的）查询和文档总数；

`--img_concept` 和 `--query_concept` 默认加上就好；

`--search_by_cluster` 表示是否使用 indexing

`--clustering_topk` 表示从 clustering 中选取多少个文档来 rerank

`--parallel` 表示是否并行 query parsing 和 load data


从文件里加载数据集的部分可能需要自己改一改

vllm 的运行环境和主程序的环境似乎有冲突，可以另开一个环境，`./vllm` 文件夹里面有单独的 `requirements.txt`，运行 `./vllm/vllm.sh` 来启动服务，然后再运行主程序。