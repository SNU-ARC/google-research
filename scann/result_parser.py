import argparse
import os
import math

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--program', type=str, help='scann, faiss ...')
parser.add_argument('--dataset', type=str, default=None, help='sift1b, glove ...')
parser.add_argument('--num_split', type=int, default=-1, help='# of splits')
parser.add_argument('--metric', type=str, default=None, help='dot_product, squared_l2')
## Common algorithm parameters
# parser.add_argument('--L', type=int, default=-1, help='# of coarse codewords')
# parser.add_argument('--w', type=int, default=-1, help='# of clusters to search')
# parser.add_argument('--m', type=int, default=-1, help='# of dimension chunks')
parser.add_argument('--topk', type=int, default=-1, help='# of final result')
parser.add_argument('--batch', type=int, default=-1, help='# of final result')
## ScaNN parameters
# parser.add_argument('--threshold', type=float, default=0.2, help='anisotropic_quantization_threshold')
parser.add_argument('--reorder', type=int, default=-1, help='reorder size')
## Faiss parameters
# parser.add_argument('--k_star', type=int, default=-1, help='# of a single finegrained codewords')
# parser.add_argument('--is_gpu', action='store_true')
# parser.add_argument('--opq', type=int, default=-1, help='new desired dimension after applying OPQ')
parser.add_argument('--recall', type=float, default=0.8, help='lowerbound of recall')


args = parser.parse_args()

assert args.dataset != None and args.topk <= 1000
assert args.metric == "squared_l2" or args.metric == "dot_product"
assert args.program!=None and args.metric!=None and args.num_split!=-1 and args.topk!=-1

read_file_name = str(args.program)+"_"+str(args.dataset)+"_topk_"+str(args.topk)+"_num_split_"+str(args.num_split)+"_batch_"+str(args.batch)+"_"+str(args.metric)+"_reorder_"+str(args.reorder)+"_sweep_result.txt"
read_file_location = "./result/"

read_file = open(read_file_location+read_file_name, "r")
config_list = {}
read_lines = read_file.readlines()
read_line_number = 0

# scann
for line in read_lines:
	read_line_number += 1
	if read_line_number == 1 or read_line_number == 2:
		continue
	if read_line_number % 2 == 1:
		curr_config_preprocessed = line.split("\t")
		curr_L = int(curr_config_preprocessed[0])
		if args.program == "scann":
			curr_threshold = float(curr_config_preprocessed[1])
			curr_kstar = 16
			curr_m = int(curr_config_preprocessed[2])
		else:
			curr_threshold = None
			curr_kstar = int(curr_config_preprocessed[2])
			curr_m = int(curr_config_preprocessed[1])
		curr_mlog2kstar = curr_m * int(math.log(curr_kstar, 2))
		curr_config = (curr_L, curr_threshold, curr_m, curr_kstar, curr_mlog2kstar)
	else:
		curr_result_preprocessed = line.split("\t")
		curr_result = float(curr_result_preprocessed[3].split(" ")[0])
		if config_list.get(curr_config) == None:
			config_list[curr_config] = curr_result
			continue
		prev_highest_result = config_list[curr_config]
		if curr_result > prev_highest_result:
			config_list[curr_config] = curr_result

read_file.close()

# rule out recall < args.recall
selected_config_list = {}
for config in config_list:
	if config_list[config] >= args.recall:
		selected_config_list[config] = config_list[config]


print(selected_config_list)
