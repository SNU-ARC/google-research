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
# parser.add_argument('--recall', type=float, default=0.8, help='lowerbound of recall')

args = parser.parse_args()

assert args.dataset != None and args.topk <= 1000
assert args.metric == "squared_l2" or args.metric == "dot_product"
assert args.program!=None and args.metric!=None and args.num_split!=-1 and args.topk!=-1

original_file_name = str(args.program)+"_"+str(args.dataset)+"_topk_"+str(args.topk)+"_num_split_"+str(args.num_split)+"_batch_"+str(args.batch)+"_"+str(args.metric)+"_reorder_"+str(args.reorder)+"_sweep_result.txt"
original_file_location = "./result/"
new_file_location = "./final_result/"

original_read_file = open(original_file_location+original_file_name, "r")
original_read_lines = original_read_file.readlines()
original_read_line_number = 0

original_configs = list()
original_recalls = list()
original_latencies = list()

for line in original_read_lines:
	original_read_line_number += 1
	if original_read_line_number == 1:
		new_file_first_line = line
		continue
	elif original_read_line_number == 2:
		new_file_second_line = line
		continue
	if original_read_line_number % 2 == 1:
		original_configs.append(line)
	else:
		results = line.split("\t")
		top1 = float(results[0].split(" ")[0])
		top10 = float(results[1].split(" ")[0])
		top100 = float(results[2].split(" ")[0])
		top1000 = float(results[3].split(" ")[0])
		original_recall = list([top1, top10, top100, top1000])
		original_recalls.append(original_recall)
		latency = float(results[4].split(" ")[0])
		original_latencies.append(latency)

original_read_file.close()

new_read_file = open(new_file_location+original_file_name, "r")
new_read_lines = new_read_file.readlines()
new_read_line_number = 0

new_configs = list()
new_recalls = list()

for line in new_read_lines:
	new_read_line_number += 1
	if new_read_line_number == 1:
		assert line == new_file_first_line
		continue
	elif new_read_line_number == 2:
		assert line == new_file_second_line
		continue
	if new_read_line_number % 2 == 1:
		new_configs.append(line)
	else:
		results = line.split("\t")
		top1_10 = float(results[5].split(" ")[0])
		top1_100 = float(results[6].split(" ")[0])
		top10_100 = float(results[7].split(" ")[0])
		top1_1000 = float(results[8].split(" ")[0])
		top10_1000 = float(results[9].split(" ")[0])
		top100_1000 = float(results[10].split(" ")[0])
		new_recall = list([top1_10, top1_100, top10_100, top1_1000, top10_1000, top100_1000])
		new_recalls.append(new_recall)

new_read_file.close()

new_file_name = str(args.program)+"_"+str(args.dataset)+"_topk_"+str(args.topk)+"_num_split_"+str(args.num_split)+"_batch_"+str(args.batch)+"_"+str(args.metric)+"_reorder_"+str(args.reorder)+"_sweep_result_merged.txt"

new_write_file = open(new_file_location+new_file_name, "w")

new_write_file.write(new_file_first_line)
new_write_file.write(new_file_second_line)

assert original_read_line_number == new_read_line_number

configs_idx = 0
recalls_idx = 0

for i in range(original_read_line_number-2):
	if i % 2 == 0:
		assert original_configs[configs_idx] == new_configs[configs_idx]
		new_write_file.write(original_configs[configs_idx])
		configs_idx += 1
	else:
		top1 = original_recalls[recalls_idx][0]
		top10 = original_recalls[recalls_idx][1]
		top100 = original_recalls[recalls_idx][2]
		top1000 = original_recalls[recalls_idx][3]
		top1_10 = new_recalls[recalls_idx][0]
		top1_100 = new_recalls[recalls_idx][1]
		top10_100 = new_recalls[recalls_idx][2]
		top1_1000 = new_recalls[recalls_idx][3]
		top10_1000 = new_recalls[recalls_idx][4]
		top100_1000 = new_recalls[recalls_idx][5]
		latency = original_latencies[recalls_idx]
		new_write_file.write(str(top1)+" %\t"+str(top10)+" %\t"+str(top100)+" %\t"+str(top1000)+" %\t|\t"+str(top1_10)+" %\t"+str(top1_100)+" %\t"+str(top10_100)+" %\t"+str(top1_1000)+" %\t"+str(top10_1000)+" %\t"+str(top100_1000)+" %\t"+str(latency)+"\n")
		recalls_idx += 1

new_write_file.close()