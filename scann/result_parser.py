import argparse
import os
import math
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--program', type=str, help='scann, faiss ...')
parser.add_argument('--dataset', type=str, default=None, help='sift1b, glove ...')
# parser.add_argument('--num_split', type=int, default=-1, help='# of splits')
parser.add_argument('--metric', type=str, default=None, help='dot_product, squared_l2')
## Common algorithm parameters
# parser.add_argument('--L', type=int, default=-1, help='# of coarse codewords')
# parser.add_argument('--w', type=int, default=-1, help='# of clusters to search')
# parser.add_argument('--m', type=int, default=-1, help='# of dimension chunks')
# parser.add_argument('--topk', type=int, default=-1, help='# of final result')
# parser.add_argument('--batch', type=int, default=-1, help='# of final result')
## ScaNN parameters
# parser.add_argument('--threshold', type=float, default=0.2, help='anisotropic_quantization_threshold')
# parser.add_argument('--reorder', type=int, default=-1, help='reorder size')
## Faiss parameters
# parser.add_argument('--k_star', type=int, default=-1, help='# of a single finegrained codewords')
# parser.add_argument('--is_gpu', action='store_true')
# parser.add_argument('--opq', type=int, default=-1, help='new desired dimension after applying OPQ')
parser.add_argument('--recall', type=float, default=0.8, help='lowerbound of recall')
parser.add_argument('--target', metavar="TARGET", default="1000@1000")


args = parser.parse_args()

assert args.dataset != None
assert args.metric == "squared_l2" or args.metric == "dot_product"
assert args.program!=None and args.metric!=None

read_file_name = str(args.program)+"_"+str(args.dataset)+"_topk_1000_num_split_"+(str(20) if args.dataset == "sift1b" or args.dataset == "deep1b" else str(1))+"_batch_128_"+str(args.metric)+"_reorder_-1_sweep_result_merged.txt"
read_file_location = "./final_result/"

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
		curr_threshold = float(curr_config_preprocessed[1]) if args.program == "scann" else None
		curr_kstar = 16 if args.program == "scann" else int(curr_config_preprocessed[2])
		curr_m = int(curr_config_preprocessed[2]) if args.program == "scann" else int(curr_config_preprocessed[1])
		curr_mlog2kstar = curr_m * int(math.log(curr_kstar, 2))
		curr_config = (curr_L, curr_threshold, curr_m, curr_kstar, curr_mlog2kstar)
	else:
		curr_result_preprocessed = line.split()
		if args.target == "1@1":
			curr_result = float(curr_result_preprocessed[0])
		elif args.target == "10@10":
			curr_result = float(curr_result_preprocessed[2])
		elif args.target == "100@100":
			curr_result = float(curr_result_preprocessed[4])
		elif args.target == "1000@1000":
			curr_result = float(curr_result_preprocessed[6])
		elif args.target == "1@10":
			curr_result = float(curr_result_preprocessed[9])
		elif args.target == "1@100":
			curr_result = float(curr_result_preprocessed[11])
		elif args.target == "10@100":
			curr_result = float(curr_result_preprocessed[13])
		elif args.target == "1@1000":
			curr_result = float(curr_result_preprocessed[15])
		elif args.target == "10@1000":
			curr_result = float(curr_result_preprocessed[17])
		elif args.target == "100@1000":
			curr_result = float(curr_result_preprocessed[19])
		else:
			assert False
		if config_list.get(curr_config) == None:
			config_list[curr_config] = curr_result
			continue
		prev_highest_result = config_list[curr_config]
		if curr_result > prev_highest_result:
			config_list[curr_config] = curr_result

read_file.close()

# rule out recall < args.recall
selected_config_list = {}
ordered_config_list = OrderedDict(sorted(config_list.items(), key = lambda x: (x[0][-1], x[0][0])))

for config in ordered_config_list:
	if ordered_config_list[config] >= args.recall:
		selected_config_list[config] = ordered_config_list[config]

result_path = "./final_config/"
result_fn = "config_"+str(args.program)+"_"+str(args.dataset)+"_"+str(args.metric)+"_"+"Recall"+str(args.target)+"_"+str(args.recall)

result_file = open(result_path+result_fn, "w")
result_file.write("L\tthreshold\tm\tkstar\tmlog2kstar\n")
for selected_config in selected_config_list.keys():
	L = selected_config[0]
	threshold = selected_config[1]
	m = selected_config[2]
	kstar = selected_config[3]
	mlog2kstar = selected_config[4]
	result_file.write(str(L)+"\t"+str(threshold)+"\t"+str(m)+"\t"+str(kstar)+"\t"+str(mlog2kstar)+"\n")

print(result_fn, "is successfully written at", result_path, ", closing ...")
result_file.close()