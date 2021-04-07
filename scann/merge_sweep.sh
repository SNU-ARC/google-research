# python3 split_and_run.py --program scann --dataset sift1m --metric squared_l2 --num_split 1 --topk 1000 --batch 128 --sweep
# python3 merge_sweep.py --program scann --dataset sift1m --metric squared_l2 --num_split 1 --topk 1000 --batch 128  --reorder -1

python3 split_and_run.py --program scann --dataset deep1m --metric squared_l2 --num_split 1 --topk 1000 --batch 128 --sweep --trace
python3 split_and_run.py --program scann --dataset deep1m --metric dot_product --num_split 1 --topk 1000 --batch 128 --sweep --trace
python3 split_and_run.py --program scann --dataset music1m --metric squared_l2 --num_split 1 --topk 1000 --batch 128 --sweep --trace
python3 split_and_run.py --program scann --dataset music1m --metric dot_product --num_split 1 --topk 1000 --batch 128 --sweep --trace
