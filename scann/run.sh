# Annoy
#python3 split_and_run_sweep.py --dataset sift1m --program annoy --eval_split --num_split 1 --metric squared_l2 --topk 10
#python3 split_and_run_sweep.py --dataset glove --program annoy --eval_split --num_split 1 --metric dot_product --topk 10
python3 split_and_run_sweep.py --dataset sift1b --program annoy --eval_split --num_split 20 --metric squared_l2 --topk 10
