#!/bin/bash

data_path="./data/"

dataset_name="crepe"

cd /home/keli/Decompose_Retrieval/

command="python main_img.py --dataset_name ${dataset_name} --data_path ${data_path} --query_count -1 --total_count 500 --img_concept --query_concept --patch_count=16 --parallel"
	echo "$command"
$command > logs/crepe/image500/patch_16.txt 2>&1 

command="python main_img.py --dataset_name ${dataset_name} --data_path ${data_path} --query_count -1 --total_count 500 --img_concept --query_concept --patch_count=32 --parallel"
	echo "$command"
$command > logs/crepe/image500/patch_32.txt 2>&1 

command="python main_img.py --dataset_name ${dataset_name} --data_path ${data_path} --query_count -1 --total_count 500 --img_concept --query_concept --patch_count=16 --parallel --agg_arg=1.0"
	echo "$command"
$command > logs/crepe/image500/baseline.txt 2>&1 
