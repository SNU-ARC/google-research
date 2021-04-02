import argparse
import os

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--program', type=str, help='scann, faiss ...')
parser.add_argument('--dataset', type=str, default=None, help='sift1b, glove ...')
parser.add_argument('--num_split', type=int, default=-1, help='# of splits')
parser.add_argument('--metric', type=str, default=None, help='dot_product, squared_l2')
## Common algorithm parameters
parser.add_argument('--L', type=int, default=-1, help='# of coarse codewords')
parser.add_argument('--w', type=int, default=-1, help='# of clusters to search')
parser.add_argument('--m', type=int, default=-1, help='# of dimension chunks')
parser.add_argument('--topk', type=int, default=-1, help='# of final result')
## ScaNN parameters
parser.add_argument('--threshold', type=float, default=0.2, help='anisotropic_quantization_threshold')
parser.add_argument('--reorder', type=int, default=-1, help='reorder size')
## Faiss parameters
parser.add_argument('--k_star', type=int, default=-1, help='# of a single finegrained codewords')
# parser.add_argument('--is_gpu', action='store_true')
# parser.add_argument('--opq', type=int, default=-1, help='new desired dimension after applying OPQ')

args = parser.parse_args()

assert args.dataset != None and args.topk <= 1000
assert args.metric == "squared_l2" or args.metric == "dot_product"
assert args.program!=None and args.metric!=None and args.num_split!=-1 and args.topk!=-1

if args.program =='scann':
	import scann
	assert args.topk <= args.reorder if args.reorder!=-1 else True
	assert args.L!=-1 and args.w!=-1 and args.topk!=-1 and args.k_star == -1 and args.m!=-1
	assert args.topk!=-1
elif args.program == "faiss":
	from runfaiss import build_faiss, faiss_search
	import math
	assert args.L!=-1 and args.k_star!=-1 and args.w!=-1 and args.m!=-1

read_file_location = "./"
write_file_location = "../../ANNA_chisel/simulator/trace/"
read_file_name = "temp.out"
if args.program == 'scann':
	write_file_name = "trace_scann_"+args.dataset+"_"+args.metric+"_"+str(args.L)+"L_"+str(args.w)+"w_"+str(args.threshold)+"thresholdTEMP"
	num_query = 10000
elif args.program == 'faiss':
	write_file_name = "trace_faiss_"+args.dataset+"_"+args.metric+"_"+str(args.L)+"L_"+str(args.w)+"wTEMP"
	num_query = 10000

def is_valid_format(filename):
	target_file = open(write_file_location + filename, "r")
	target_file_lines = target_file.readlines()
	target_file_line_number = 0
	for target_file_line in target_file_lines:
		target_file_line_number += 1
		elements = target_file_line.split("\t")
		assert len(elements) == args.w + 1
	return True

read_file = open(read_file_location+read_file_name, "r")
write_file = open(write_file_location+write_file_name, "w")

lines = read_file.readlines()
line_number = 0
for line in lines:
	line_number += 1
	if args.program == 'scann':
		if line_number <= 3:
			continue
		if line_number < num_query + 3:
			write_file.write(line)
		elif line_number == num_query + 3:
			parsed_line = line.split("Reading")[0]
		elif line_number > num_query + 3 and line_number < num_query + 26:
			continue
		elif line_number == num_query + 26:
			parsed_line += line
			write_file.write(parsed_line)
		else:
			assert False
	elif args.program == 'faiss':
		if line_number == 1:
			continue
		elif line_number <= num_query + 1:
			write_file.write(line)
		else:
			continue

write_file.close()
read_file.close()
if os.path.exists(read_file_location+read_file_name):
	os.remove(read_file_location+read_file_name)
	print(read_file_location, read_file_name, "is successfully deleted.")
else:
	print(read_file_location, read_file_name, "does not exist. Cannot delete it.")
if is_valid_format(write_file_name):
	print("Trace has valid format! Terminating..")


