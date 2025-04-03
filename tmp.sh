export CUDA_VISIBLE_DEVICES=1

python qa_squad.py | tee -a ./logs.txt

python qa_triviaqa.py | tee -a ./logs.txt