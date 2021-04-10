python3 anna_plot_final_result.py --program scann --dataset sift1m --metric squared_l2 --reorder -1 --topk 1000 --build_config --target 100@1000
python3 anna_plot_final_result.py --program scann --dataset glove --metric dot_product --reorder -1 --topk 1000 --build_config --target 100@1000
python3 anna_plot_final_result.py --program scann --dataset music1m --metric dot_product --reorder -1 --topk 1000 --build_config --target 100@1000

python3 anna_plot_final_result.py --program faiss --dataset sift1m --metric squared_l2 --reorder -1 --topk 1000 --build_config --target 100@1000
python3 anna_plot_final_result.py --program faiss --dataset music1m --metric dot_product --reorder -1 --topk 1000 --build_config --target 100@1000

python3 anna_plot_final_result.py --program faissGPU --dataset sift1m --metric squared_l2 --reorder -1 --topk 1000 --build_config --target 100@1000
python3 anna_plot_final_result.py --program faissGPU --dataset glove --metric dot_product --reorder -1 --topk 1000 --build_config --target 100@1000
python3 anna_plot_final_result.py --program faissGPU --dataset music1m --metric dot_product --reorder -1 --topk 1000 --build_config --target 100@1000
python3 anna_plot_final_result.py --program faissGPU --dataset sift1b --metric squared_l2 --reorder -1 --topk 1000 --build_config --target 100@1000
