# Annoy
#python3 split_and_run_sweep.py --dataset sift1m --program annoy --eval_split --num_split 1 --metric squared_l2 --topk 10
#python3 split_and_run_sweep.py --dataset glove --program annoy --eval_split --num_split 1 --metric dot_product --topk 10
#python3 split_and_run_sweep.py --dataset sift1b --program annoy --eval_split --num_split 200 --metric squared_l2 --topk 10

# Faiss
# python3 split_and_run_sweep.py --dataset sift1m --program faiss --eval_split --num_split 1 --metric squared_l2 --topk 10
#python3 split_and_run_sweep.py --dataset glove --program faiss --eval_split --num_split 1 --metric dot_product --topk 10
# python3 split_and_run_sweep.py --dataset sift1b --program faiss --eval_split --num_split 20 --metric squared_l2 --topk 10

sudo python3 split_and_run.py  --dataset glove --metric dot_product --num_split 1 --is_gpu --program faiss --sweep --topk 1000 --batch 128
sudo python3 split_and_run.py  --dataset sift1m --metric dot_product --num_split 1 --is_gpu --program faiss --sweep --topk 1000 --batch 128
sudo python3 split_and_run.py  --dataset glove --metric squared_l2 --num_split 1 --is_gpu --program faiss --sweep --topk 1000 --batch 128
sudo python3 split_and_run.py  --dataset sift1m --metric squared_l2 --num_split 1 --is_gpu --program faiss --sweep --topk 1000 --batch 128
# ScaNN
# python3 split_and_run_sweep.py --dataset sift1m --program scann --eval_split --num_split 1 --metric squared_l2 --topk 10
#python3 split_and_run_sweep.py --dataset glove --program scann --eval_split --num_split 1 --metric dot_product --topk 10
# python3 split_and_run_sweep.py --dataset sift1b --program scann --eval_split --num_split 20 --metric squared_l2 --topk 10
