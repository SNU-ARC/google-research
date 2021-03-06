# Annoy
# python3 split_and_run_sweep.py --dataset sift1m --program annoy --eval_split --num_split 1 --metric squared_l2 --topk 1000
# python3 split_and_run_sweep.py --dataset glove --program annoy --eval_split --num_split 1 --metric dot_product --topk 1000
# python3 split_and_run_sweep.py --dataset sift1b --program annoy --eval_split --num_split 200 --metric squared_l2 --topk 1000

# Faiss cpu
python3 split_and_run_faisscpu1b.py --program faiss --dataset sift1b --sweep --topk 1000 --metric squared_l2 --num_split 4 --batch 16
# python3 split_and_run_faisscpu1b.py --program faiss --dataset deep1b --sweep --topk 1000 --metric squared_l2 --num_split 4 --batch 16

# python3 split_and_run.py --program faiss --dataset sift1m --sweep --topk 1000 --metric squared_l2 --num_split 1 --batch 128
# python3 split_and_run.py --program faiss --dataset glove --sweep --topk 1000 --metric squared_l2 --num_split 1 --batch 128
# python3 split_and_run.py --program faiss --dataset sift1m --sweep --topk 1000 --metric dot_product --num_split 1 --batch 128
# python3 split_and_run.py --program faiss --dataset glove --sweep --topk 1000 --metric dot_product --num_split 1 --batch 128
# python3 split_and_run.py --program faiss --dataset deep1b --sweep --topk 1000 --metric squared_l2 --num_split 20 --batch 128
# python3 split_and_run.py --program faiss --dataset deep1b --sweep --topk 1000 --metric dot_product --num_split 20 --batch 128

# Faiss gpu
# python3 split_and_run.py --dataset glove --metric dot_product --num_split 1 --is_gpu --program faiss --sweep --topk 1000 --batch 128
# python3 split_and_run.py --dataset sift1m --metric dot_product --num_split 1 --is_gpu --program faiss --sweep --topk 1000 --batch 128
# python3 split_and_run.py --dataset glove --metric squared_l2 --num_split 1 --is_gpu --program faiss --sweep --topk 1000 --batch 128
# python3 split_and_run.py --dataset sift1m --metric squared_l2 --num_split 1 --is_gpu --program faiss --sweep --topk 1000 --batch 128
# python3 split_and_run.py --dataset deep1b --metric squared_l2 --num_split 20 --is_gpu --program faiss --sweep --topk 1000 --batch 128

# ScaNN
# python3 split_and_run.py --dataset sift1m --program scann --sweep --num_split 1 --metric squared_l2 --topk 1000 --reorder 1000 --batch 128
# python3 split_and_run.py --dataset glove --program scann --sweep --num_split 1 --metric squared_l2 --topk 1000 --reorder 1000 --batch 128
# python3 split_and_run.py --dataset sift1b --program scann --sweep --num_split 20 --metric squared_l2 --topk 1000 --reorder 1000 --batch 128
# python3 split_and_run.py --dataset sift1m --program scann --sweep --num_split 1 --metric dot_product --topk 1000 --reorder 1000 --batch 128
# python3 split_and_run.py --dataset glove --program scann --sweep --num_split 1 --metric dot_product --topk 1000 --reorder 1000 --batch 128
# python3 split_and_run.py --dataset sift1b --program scann --sweep --num_split 20 --metric dot_product --topk 1000 --reorder 1000 --batch 128

# python3 split_and_run.py --dataset sift1m --program scann --sweep --num_split 1 --metric squared_l2 --topk 1000 --batch 128
# python3 split_and_run.py --dataset glove --program scann --sweep --num_split 1 --metric dot_product --topk 1000 --batch 128
# python3 split_and_run.py --dataset sift1m --program scann --sweep --num_split 1 --metric dot_product --topk 1000 --batch 128
# python3 split_and_run.py --dataset glove --program scann --sweep --num_split 1 --metric squared_l2 --topk 1000 --batch 128
# python3 split_and_run.py --dataset sift1b --program scann --sweep --num_split 20 --metric squared_l2 --topk 1000 --batch 128
# python3 split_and_run.py --dataset sift1b --program scann --sweep --num_split 20 --metric dot_product --topk 1000 --batch 128
# python3 split_and_run.py --dataset deep1b --program scann --sweep --num_split 20 --metric squared_l2 --topk 1000 --batch 128
# python3 split_and_run.py --dataset deep1b --program scann --sweep --num_split 20 --metric dot_product --topk 1000 --batch 128
# python3 split_and_run.py --dataset music1m --program scann --sweep --num_split 1 --metric squared_l2 --topk 1000 --batch 128
# python3 split_and_run.py --dataset deep1m --program scann --sweep --num_split 1 --metric dot_product --topk 1000 --batch 128
# python3 split_and_run.py --dataset music1m --program scann --sweep --num_split 1 --metric dot_product --topk 1000 --batch 128
# python3 split_and_run.py --dataset deep1m --program scann --sweep --num_split 1 --metric squared_l2 --topk 1000 --batch 128

#Batch 1
# Faiss cpu
# python3 split_and_run.py --program faiss --dataset deep1b --sweep --topk 1000 --metric squared_l2 --num_split 20 --batch 1
# python3 split_and_run.py --program faiss --dataset sift1b --sweep --topk 1000 --metric squared_l2 --num_split 20 --batch 1
# python3 split_and_run.py --program faiss --dataset sift1m --sweep --topk 1000 --metric squared_l2 --num_split 1 --batch 1
# python3 split_and_run.py --program faiss --dataset deepm96 --sweep --topk 1000 --metric squared_l2 --num_split 1 --batch 1
# python3 split_and_run.py --program faiss --dataset glove --sweep --topk 1000 --metric dot_product --num_split 1 --batch 1
# python3 split_and_run.py --program faiss --dataset music1m --sweep --topk 1000 --metric dot_product --num_split 1 --batch 1

# Faiss gpu
# python3 split_and_run.py --dataset glove --metric dot_product --num_split 1 --is_gpu --program faiss --sweep --topk 1000 --batch 1
# python3 split_and_run.py --dataset sift1m --metric squared_l2 --num_split 1 --is_gpu --program faiss --sweep --topk 1000 --batch 1
# python3 split_and_run.py --dataset deepm96 --metric squared_l2 --num_split 1 --is_gpu --program faiss --sweep --topk 1000 --batch 1
# python3 split_and_run.py --dataset music1m --metric dot_product --num_split 1 --is_gpu --program faiss --sweep --topk 1000 --batch 1
# python3 split_and_run.py --dataset deep1b --metric squared_l2 --num_split 20 --is_gpu --program faiss --sweep --topk 1000 --batch 1
# python3 split_and_run.py --dataset sift1b --metric squared_l2 --num_split 20 --is_gpu --program faiss --sweep --topk 1000 --batch 1


# Scann
# python3 split_and_run.py --dataset sift1m --program scann --sweep --num_split 1 --metric squared_l2 --topk 1000 --batch 1
# python3 split_and_run.py --dataset glove --program scann --sweep --num_split 1 --metric dot_product --topk 1000 --batch 1
# python3 split_and_run.py --dataset deepm96 --program scann --sweep --num_split 1 --metric squared_l2 --topk 1000 --batch 1
# python3 split_and_run.py --dataset music1m --program scann --sweep --num_split 1 --metric dot_product --topk 1000 --batch 1
# python3 split_and_run.py --dataset sift1b --program scann --sweep --num_split 20 --metric squared_l2 --topk 1000 --batch 1
# python3 split_and_run.py --dataset deep1b --program scann --sweep --num_split 20 --metric squared_l2 --topk 1000 --batch 1
