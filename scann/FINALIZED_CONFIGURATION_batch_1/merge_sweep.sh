python3 merge_sweep.py --program faiss --dataset sift1m --metric squared_l2
python3 merge_sweep.py --program faiss --dataset glove --metric dot_product
python3 merge_sweep.py --program faiss --dataset music1m --metric dot_product
python3 merge_sweep.py --program faiss --dataset deepm96 --metric squared_l2
python3 merge_sweep.py --program faiss --dataset sift1b --metric squared_l2
python3 merge_sweep.py --program faiss --dataset deep1b --metric squared_l2

python3 merge_sweep.py --program faissGPU --dataset sift1m --metric squared_l2
python3 merge_sweep.py --program faissGPU --dataset glove --metric dot_product
python3 merge_sweep.py --program faissGPU --dataset music1m --metric dot_product
python3 merge_sweep.py --program faissGPU --dataset deepm96 --metric squared_l2
python3 merge_sweep.py --program faissGPU --dataset sift1b --metric squared_l2
python3 merge_sweep.py --program faissGPU --dataset deep1b --metric squared_l2

python3 merge_sweep.py --program scann --dataset sift1m --metric squared_l2
python3 merge_sweep.py --program scann --dataset glove --metric dot_product
python3 merge_sweep.py --program scann --dataset music1m --metric dot_product
python3 merge_sweep.py --program scann --dataset deepm96 --metric squared_l2
python3 merge_sweep.py --program scann --dataset sift1b --metric squared_l2
python3 merge_sweep.py --program scann --dataset deep1b --metric squared_l2