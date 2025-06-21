#!/bin/bash

# function handle_sigint {
#     echo
#     echo "Caught SIGINT (Ctrl+C)! Cleaning up..."
#     # Perform any necessary cleanup here
#     exit 1
# }

# Set the trap to call handle_sigint on SIGINT
# trap handle_sigint SIGINT


data_path="/home/keli/Decompose_Retrieval/data"

dataset_name="trec-covid"

cd /home/keli/Decompose_Retrieval/

for val in 1000 2000 5000 10000 15000 20000
do
	command="python main_text.py --dataset_name ${dataset_name} --data_path ${data_path} --query_count -1 --total_count -1 --img_concept --query_concept --search_by_cluster --clustering_topk=$val"
        echo "$command"
	$command > logs/trec-covid/seq/output_full_query_full_data_patch_1_seq_topk_${val}.txt 2>&1 

done
