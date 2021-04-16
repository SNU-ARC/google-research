'''Usage: python3 merge_sweep.py --program scann --dataset sift1m --metric squared_l2 '''
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
# parser.add_argument('--recall', type=float, default=0.8, help='lowerbound of recall')

args = parser.parse_args()

assert args.dataset != None
assert args.metric == "squared_l2" or args.metric == "dot_product"
assert args.program!=None and args.metric!=None

if args.dataset == "sift1b" or args.dataset == "deep1b":
	if args.program == "scann":
		num_split = 20
	else:
		num_split = 4
else:
	num_split = 1

if args.program == "faiss":
	num_batch = 16
else:
	num_batch = 128

recall_file_location = "../FINALIZED_CONFIGURATION/"
recall_file_name = "final_eval_config_"+str(args.program)+"_"+str(args.dataset)+"_topk_1000"+"_num_split_"+str(num_split)+"_batch_"+str(num_batch)+"_"+str(args.metric)+"_reorder_-1_sweep_result.txt"
recall_read_file = open(recall_file_location+recall_file_name, "r")
recall_read_lines = recall_read_file.readlines()
recall_read_line_number = 0

recall_configs = OrderedDict()


def parse_odd_lines(line):
	tokens = line.split("\t")
	L = int(tokens[0])
	threshold = float(tokens[1]) if args.program == "scann" else None
	m = int(tokens[2]) if args.program == "scann" else int(tokens[1])
	kstar = 16 if args.program == "scann" else int(tokens[2])
	w = int(tokens[4])
	return L, threshold, m, kstar, w


def parse_even_lines_recall(line):
	tokens = line.split("\t")
	top1 = tokens[0]
	top10 = tokens[1]
	top100 = tokens[2]
	top1000 = tokens[3]
	top1_10 = tokens[5]
	top1_100 = tokens[6]
	top10_100 = tokens[7]
	top1_1000 = tokens[8]
	top10_1000 = tokens[9]
	top100_1000 = tokens[10]
	latency = tokens[11]
	return top1, top10, top100, top1000, top1_10, top1_100, top10_100, top1_1000, top10_1000, top100_1000, latency


def parse_even_lines_latency(line):
	tokens = line.split("\t")
	top1 = tokens[0]
	top10 = tokens[1]
	top100 = tokens[2]
	top1000 = tokens[3]
	latency = tokens[4]
	return top1, top10, top100, top1000, latency


for line in recall_read_lines:
	recall_read_line_number += 1
	if recall_read_line_number == 1 or recall_read_line_number == 2:
		continue
	elif recall_read_line_number % 2 == 1:
		L, threshold, m, kstar, w = parse_odd_lines(line)
		recall_config = tuple([L, threshold, m, kstar, w])
	else:
		top1, top10, top100, top1000, top1_10, top1_100, top10_100, top1_1000, top10_1000, top100_1000, latency = parse_even_lines_recall(line)
		default_latency = -1
		recalls_and_latency = tuple([top1, top10, top100, top1000, top1_10, top1_100, top10_100, top1_1000, top10_1000, top100_1000, default_latency])
		recall_configs[recall_config] = recalls_and_latency

recall_configs_length = len(recall_configs)

recall_read_file.close()

latency_file_location = "./"
latency_file_name = "batch1_"+str(args.program)+"_"+str(args.dataset)+"_topk_1000"+"_num_split_"+str(num_split)+"_batch_1_"+str(args.metric)+"_reorder_-1_sweep_result.txt"
latency_read_file = open(latency_file_location+latency_file_name, "r")
latency_read_lines = latency_read_file.readlines()
latency_read_line_number = 0


old_recall_configs = OrderedDict()

for line in latency_read_lines:
	latency_read_line_number += 1
	if latency_read_line_number == 1:
		new_file_first_line = line
	elif latency_read_line_number == 2:
		new_file_second_line = line
	elif latency_read_line_number %2 == 1:
		L, threshold, m, kstar, w = parse_odd_lines(line)
		old_recall_config = tuple([L, threshold, m, kstar, w])
	else:
		top1, top10, top100, top1000, top1_10, top1_100, top10_100, top1_1000, top10_1000, top100_1000, latency = parse_even_lines_recall(line)
		# default_latency = -1
		recalls_and_latency = tuple([top1, top10, top100, top1000, top1_10, top1_100, top10_100, top1_1000, top10_1000, top100_1000, latency])
		# recall_configs[recall_config] = recalls_and_latency

		# top1, top10, top100, top1000, latency = parse_even_lines_latency(line)
		# recalls_and_latency = tuple([top1, top10, top100, top1000, latency])
		old_recall_configs[old_recall_config] = recalls_and_latency

old_recall_configs_length = len(old_recall_configs)

latency_read_file.close()

new_file_location = latency_file_location
new_file_name = str(args.program)+"_"+str(args.dataset)+"_topk_1000"+"_num_split_"+str(num_split)+"_batch_1_"+str(args.metric)+"_reorder_-1_sweep_result.txt"
new_write_file = open(new_file_location+new_file_name, "w")
new_write_file.write(new_file_first_line)
new_write_file.write(new_file_second_line)

old_recall_configs_list = list(old_recall_configs.items())

for i in range(old_recall_configs_length):
	curr_config = old_recall_configs_list[i][0]
	# curr_many_recalls = old_recall_configs_list[i][1]
	if curr_config in recall_configs.keys():
		curr_many_recalls = recall_configs[curr_config]
		curr_latency = old_recall_configs[curr_config][-1]
		# print(curr_latency)
	else:
		continue
	L = int(curr_config[0])
	threshold = float(curr_config[1]) if args.program == "scann" else curr_config[1]
	m = int(curr_config[2])
	kstar = int(curr_config[3])
	w = int(curr_config[4])
	top1 = curr_many_recalls[0]
	top10 = curr_many_recalls[1]
	top100 = curr_many_recalls[2]
	top1000 = curr_many_recalls[3]
	top1_10 = curr_many_recalls[4]
	top1_100 = curr_many_recalls[5]
	top10_100 = curr_many_recalls[6]
	top1_1000 = curr_many_recalls[7]
	top10_1000 = curr_many_recalls[8]
	top100_1000 = curr_many_recalls[9]
	if args.program == "scann":
		new_write_file.write(str(L)+"\t"+str(threshold)+"\t"+str(m)+"\t|\t"+str(w)+"\t-1\t"+args.metric+"\n")
	else:
		new_write_file.write(str(L)+"\t"+str(m)+"\t"+str(kstar)+"\t|\t"+str(w)+"\t-1\t"+args.metric+"\n")
	new_write_file.write(top1+"\t"+top10+"\t"+top100+"\t"+top1000+"\t|\t"+top1_10+"\t"+top1_100+"\t"+top10_100+"\t"+top1_1000+"\t"+top10_1000+"\t"+top100_1000+"\t"+str(curr_latency))

# recall_configs_list = list(recall_configs.items())

# for i in range(recall_configs_length):
# 	curr_config = recall_configs_list[i][0]
# 	curr_many_recalls = recall_configs_list[i][1]
# 	if curr_config in old_recall_configs.keys():
# 		curr_latency = old_recall_configs[curr_config][-1]
# 	else:
# 		curr_latency = -1
# 		continue
# 	L = int(curr_config[0])
# 	threshold = float(curr_config[1]) if args.program == "scann" else curr_config[1]
# 	m = int(curr_config[2])
# 	kstar = int(curr_config[3])
# 	w = int(curr_config[4])
# 	top1 = curr_many_recalls[0]
# 	top10 = curr_many_recalls[1]
# 	top100 = curr_many_recalls[2]
# 	top1000 = curr_many_recalls[3]
# 	top1_10 = curr_many_recalls[4]
# 	top1_100 = curr_many_recalls[5]
# 	top10_100 = curr_many_recalls[6]
# 	top1_1000 = curr_many_recalls[7]
# 	top10_1000 = curr_many_recalls[8]
# 	top100_1000 = curr_many_recalls[9]
# 	if args.program == "scann":
# 		new_write_file.write(str(L)+"\t"+str(threshold)+"\t"+str(m)+"\t|\t"+str(w)+"\t-1\t"+args.metric+"\n")
# 	else:
# 		new_write_file.write(str(L)+"\t"+str(m)+"\t"+str(kstar)+"\t|\t"+str(w)+"\t-1\t"+args.metric+"\n")
# 	new_write_file.write(top1+"\t"+top10+"\t"+top100+"\t"+top1000+"\t|\t"+top1_10+"\t"+top1_100+"\t"+top10_100+"\t"+top1_1000+"\t"+top10_1000+"\t"+top100_1000+"\t"+curr_latency)

new_write_file.close()
