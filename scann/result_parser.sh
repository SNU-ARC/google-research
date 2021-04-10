python3 result_parser.py --program scann --dataset sift1m --metric squared_l2 --recall 0.9 --target 100@1000
python3 result_parser.py --program scann --dataset glove --metric dot_product --recall 0.9 --target 100@1000
python3 result_parser.py --program scann --dataset music1m --metric dot_product --recall 0.9 --target 100@1000

python3 result_parser.py --program faiss --dataset sift1m --metric squared_l2 --recall 0.9 --target 100@1000
python3 result_parser.py --program faiss --dataset music1m --metric dot_product --recall 0.9 --target 100@1000

python3 result_parser.py --program faissGPU --dataset sift1m --metric squared_l2 --recall 0.9 --target 100@1000
python3 result_parser.py --program faissGPU --dataset glove --metric dot_product --recall 0.9 --target 100@1000
python3 result_parser.py --program faissGPU --dataset music1m --metric dot_product --recall 0.9 --target 100@1000
python3 result_parser.py --program faissGPU --dataset sift1b --metric squared_l2 --recall 0.9 --target 100@1000