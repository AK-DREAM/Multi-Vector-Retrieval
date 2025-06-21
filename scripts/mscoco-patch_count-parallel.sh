#!/bin/bash

data_path="./data/"

dataset_name="mscoco"

cd /home/keli/Decompose_Retrieval/

# for val in 2500 5000 7500 10000 15000 20000
# do
# 	command="python main_img.py --dataset_name ${dataset_name} --data_path ${data_path} --query_count 100 --total_count 40000 --img_concept --query_concept --search_by_cluster --patch_count=16 --clustering_topk=$val --parallel --agg_arg=0.7"
#         echo "$command"
# 	$command > logs/mscoco/patch16/topk_${val}.txt 2>&1 
# done

for val in 2500 5000 7500 10000 15000 20000
do
	command="python main_img.py --dataset_name ${dataset_name} --data_path ${data_path} --query_count 100 --total_count 40000 --img_concept --query_concept --search_by_cluster --patch_count=32 --clustering_topk=$val --parallel --agg_arg=0.5"
        echo "$command"
	$command > logs/mscoco/patch32/topk_${val}.txt 2>&1 

done

# command="python main_img.py --dataset_name ${dataset_name} --data_path ${data_path} --query_count 100 --total_count 40000 --img_concept --query_concept --search_by_cluster --patch_count=32 --clustering_topk=$val --parallel --agg_arg=1.0"
# 	echo "$command"
# 	$command > logs/mscoco/baseline.txt 2>&1 