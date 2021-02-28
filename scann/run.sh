# Annoy
python3 split_and_run.py --dataset sift1m --program annoy --num_split 1 --eval_split --topk 1000 --batch 1 --metric squared_l2 --sweep
python3 split_and_run.py --dataset glove --program annoy --num_split 1 --eval_split --topk 1000 --batch 1 --metric squared_l2 --sweep
python3 split_and_run.py --dataset glove --program annoy --num_split 1 --eval_split --topk 1000 --batch 1 --metric dot_product --sweep
python3 split_and_run.py --dataset sift1m --program annoy --num_split 1 --eval_split --topk 1000 --batch 1 --metric dot_product --sweep

# Faiss cpu
#python3 split_and_run.py --program faiss --dataset sift1m --sweep --topk 1000 --reorder 1000 --metric squared_l2 --num_split 1 --batch 128
python3 split_and_run.py --program faiss --dataset glove --sweep --topk 1000 --reorder 1000 --metric squared_l2 --num_split 1 --batch 128

# Faiss gpu
sudo python3 split_and_run.py  --dataset glove --metric dot_product --num_split 1 --is_gpu --program faiss --sweep --topk 1000 --batch 128
#sudo python3 split_and_run.py  --dataset sift1m --metric dot_product --num_split 1 --is_gpu --program faiss --sweep --topk 1000 --batch 128
sudo python3 split_and_run.py  --dataset glove --metric squared_l2 --num_split 1 --is_gpu --program faiss --sweep --topk 1000 --batch 128
#sudo python3 split_and_run.py  --dataset sift1m --metric squared_l2 --num_split 1 --is_gpu --program faiss --sweep --topk 1000 --batch 128


# ScaNN
# python3 split_and_run_sweep.py --dataset sift1m --program scann --eval_split --num_split 1 --metric squared_l2 --topk 10
#python3 split_and_run_sweep.py --dataset glove --program scann --eval_split --num_split 1 --metric dot_product --topk 10
# python3 split_and_run_sweep.py --dataset sift1b --program scann --eval_split --num_split 20 --metric squared_l2 --topk 10
