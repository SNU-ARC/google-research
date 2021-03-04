'''
Usage 1: python3 split_and_run.py --dataset [dataset name] --num_split [# of split] --metric [distance measure] --num_leaves [num_leaves] --num_search [num_leaves_to_search] --coarse_training_size [coarse traing sample size] --fine_training_size [fine training sample size] --threshold [threshold] --reorder [reorder size] [--split] [--eval_split]
Usage 2: python3 split_and_run.py --dataset [dataset name] --groundtruth --metric [distance measure]
'''
import sys
import numpy as np
import time
import argparse
import os
import h5py
import math

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--program', type=str, help='scann, faiss ...')
parser.add_argument('--dataset', type=str, default=None, help='sift1b, glove ...')
parser.add_argument('--num_split', type=int, default=-1, help='# of splits')
parser.add_argument('--metric', type=str, default=None, help='dot_product, squared_l2')
## Common algorithm parameters
parser.add_argument('--L', type=int, default=-1, help='# of coarse codewords')
parser.add_argument('--w', type=int, default=-1, help='# of clusters to search')
parser.add_argument('--m', type=int, default=-1, help='# of dimension chunks')
parser.add_argument('--batch', type=int, default=1, help='query batch size')

## ScaNN parameters
parser.add_argument('--coarse_training_size', type=int, default=250000, help='coarse training sample size')
parser.add_argument('--fine_training_size', type=int, default=100000, help='fine training sample size')
parser.add_argument('--threshold', type=float, default=0.2, help='anisotropic_quantization_threshold')
parser.add_argument('--reorder', type=int, default=-1, help='reorder size')
## Faiss parameters
parser.add_argument('--k_star', type=int, default=-1, help='# of a single finegrained codewords')
parser.add_argument('--is_gpu', action='store_true')

## Annoy parameters
parser.add_argument('--n_trees', type=int, default=-1, help='# of trees')
## ScaNN & Annoy common parameters
parser.add_argument('--num_search', type=int, default=-1, help='# of searching leaves for ScaNN, # of searching datapoints for Annoy')
parser.add_argument('--topk', type=int, default=-1, help='# of final result')

## Run options
parser.add_argument('--split', action='store_true')
parser.add_argument('--eval_split', action='store_true')
parser.add_argument('--groundtruth', action='store_true')
parser.add_argument('--sweep', action='store_true')
args = parser.parse_args()

assert args.dataset != None and args.topk <= 1000
if args.split != True:
	assert args.metric == "squared_l2" or args.metric == "dot_product" or args.metric=="angular"

if args.eval_split or args.sweep:
	assert args.program!=None and args.metric!=None and args.num_split!=-1 and args.topk!=-1

if args.groundtruth:
	import ctypes
	assert args.metric!=None

if args.program=='scann':
	import scann
	if args.sweep == False:
		assert args.L!=-1 and args.w!=-1 and args.topk!=-1 and args.k_star == -1 and args.m!=-1 and (args.topk <= args.reorder if args.reorder!=-1 else True) and args.is_gpu==False
	assert args.topk!=-1
elif args.program == "faiss":
	#if os.environ.get('LD_PRELOAD') == None:
	#	assert False, "Please set LD_PRELOAD environment path and retry"
	# export LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_def.so:/opt/intel/mkl/lib/intel64/libmkl_avx2.so:/opt/intel/mkl/lib/intel64/libmkl_core.so:/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so:/opt/intel/mkl/lib/intel64/libmkl_intel_thread.so:/opt/intel/lib/intel64_lin/libiomp5.so
	from runfaiss import build_faiss, faiss_search
	import math
	if args.sweep == False:
		assert args.L!=-1 and args.k_star!=-1 and args.w!=-1 and args.m!=-1
elif args.program == "annoy":
	import annoy
	if args.batch > 1:
		from multiprocessing.pool import ThreadPool
	assert args.topk!=-1 and args.is_gpu==False and (args.num_search!=-1 and args.n_trees!=-1 if args.sweep!=True else True)

def compute_recall(neighbors, true_neighbors):
	total = 0
	for gt_row, row in zip(true_neighbors, neighbors):
		total += np.intersect1d(gt_row, row).shape[0]
	return total / true_neighbors.size

def ivecs_read(fname):
	a = np.fromfile(fname, dtype='int32')
	d = a[0]
	return a.reshape(-1, d + 1)[:, 1:].copy()

def ivecs_write(fname, m):
	n, d = m.shape
	dimension_arr = np.zeros((n, 1), dtype=np.int32)
	dimension_arr[:, 0] = d
	m = np.append(dimension_arr, m, axis=1)
	m.tofile(fname)

def bvecs_mmap(fname, offset_=None, shape_=None):
	if offset_!=None and shape_!=None:
		x = np.memmap(fname, dtype=np.uint8, mode='r', offset=offset_*132, shape=(shape_*132))
	else:
		x = np.memmap(fname, dtype=np.uint8, mode='r')

	d = x[:4].view('int32')[0]
	return x.reshape(-1, d + 4)[:, 4:]

def bvecs_write(fname, m):
	n, d = m.shape
	dimension_arr = np.zeros((n, 4), dtype=np.uint8)
	dimension_arr[:, 0] = d
	m = np.append(dimension_arr, m, axis=1)
	m.tofile(fname)

def bvecs_read(fname):
	b = np.fromfile(fname, dtype=np.uint8)
	d = b[:4].view('int32')[0]
	return b.reshape(-1, d+4)[:, 4:].copy()

def mmap_fvecs(fname):
	x = np.memmap(fname, dtype='int32', mode='r')
	d = x[0]
	return x.reshape(-1, d + 1)[:, 1:].copy().view('float32')

def fvecs_write(fname, m):
	m = m.astype('float32')
	n, d = m.shape
	m1 = np.empty((n, d + 1), dtype='int32')
	m1[:, 0] = d
	m1[:, 1:] = m.view('int32')
	m1.tofile(fname)

def read_data(dataset_path, offset_=None, shape_=None, base=True):
	if "sift1m" in args.dataset:
		file = dataset_path + "sift_base.fvecs" if base else dataset_path
		return mmap_fvecs(file)
	elif "sift1b" in args.dataset:
		file = dataset_path+"bigann_base.bvecs" if base else dataset_path
		return bvecs_mmap(file, offset_=offset_, shape_=shape_)
	elif "glove" in args.dataset:
		file = dataset_path+"glove-100-angular.hdf5" if base else dataset_path
		if base:
			dataset = h5py.File(file, "r")
			dataset = np.array(dataset['train'], dtype='float32')
			if args.metric == "dot_product":
				dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
			if offset_!=None and shape_!=None:
				return dataset[offset_:offset_+shape_]
			else:
				return dataset
		else:
			dataset = h5py.File(dataset_path, "r")
			return np.array(dataset['dataset'], dtype='float32')
	else:
		assert(false)

def write_split_data(split_data_path, split_data):
	if "sift1b" in args.dataset:
		bvecs_write(split_data_path, split_data)
	elif "sift1m" in args.dataset:
		fvecs_write(split_data_path, split_data)
	elif "glove" in args.dataset:
		hf = h5py.File(split_data_path, 'w')
		hf.create_dataset('dataset', data=split_data)
	print("Wrote to ", split_data_path, ", shape ", split_data.shape)

def write_gt_data(gt_data):
	if "sift1b" in args.dataset or "sift1m" in args.dataset:
		ivecs_write(groundtruth_path, gt_data)
	elif "glove" in args.dataset:
		hf = h5py.File(groundtruth_path, 'w')
		hf.create_dataset('dataset', data=gt_data)
	print("Wrote to ", groundtruth_path, ", shape ", gt_data.shape)

def write_split_gt_data(split_gt_path, gt_data):
	if "sift1b" in args.dataset or "sift1m" in args.dataset:
		ivecs_write(split_gt_path, gt_data)
	elif "glove" in args.dataset:
		hf = h5py.File(split_gt_path, 'w')
		hf.create_dataset('dataset', data=gt_data)
	print("Wrote to ", split_gt_path, ", shape ", gt_data.shape)

def split(filename, num_iter, N, D):
	num_per_split = int(N/args.num_split)
	dataset = np.empty((0, D), dtype=np.uint8 if 'sift1b' in args.dataset else np.float32)
	dataset_per_iter = int(N/num_iter)
	num_per_split = int(N/args.num_split)
	print("dataset_per_iter: ", dataset_per_iter, " / num_per_split: ", num_per_split)
	num_split_list=[]
	split = 0
	for it in range(num_iter):
		print("Iter: ", it)
		if it==num_iter-1:
			dataset = np.append(dataset, read_data(dataset_basedir, offset_=it*dataset_per_iter, shape_=(N-it*dataset_per_iter)), axis=0)
		else:
			dataset = np.append(dataset, read_data(dataset_basedir, offset_=it*dataset_per_iter, shape_=dataset_per_iter), axis=0)
		print("Reading dataset done")
		count=0
		while split<args.num_split:
			if (split+1)*num_per_split > dataset_per_iter*(it+1):
				if it!=num_iter-1:
					print("Entering next iter..")
					dataset = dataset[count*num_per_split:]
				else:
					split_size = dataset[count*num_per_split:].shape[0]
					write_split_data(split_dataset_path + str(args.num_split) + "_" + str(split), dataset[count*num_per_split:])
					trainset = np.random.choice(split_size, int(0.1*split_size), replace=False)
					write_split_data(split_dataset_path + "learn" + str(args.num_split) + "_" + str(split), dataset[count*num_per_split:][trainset])
					num_split_list.append(dataset[count*num_per_split:].shape[0])
					split = split+1
				break
			elif split < args.num_split:
				split_size = dataset[count*num_per_split:(count+1)*num_per_split].shape[0]
				write_split_data(split_dataset_path + str(args.num_split) + "_" + str(split), dataset[count*num_per_split:(count+1)*num_per_split])
				trainset = np.random.choice(split_size, int(0.1*split_size), replace=False)
				write_split_data(split_dataset_path + "learn" + str(args.num_split) + "_" + str(split), dataset[count*num_per_split:(count+1)*num_per_split][trainset])
				num_split_list.append(dataset[count*num_per_split:(count+1)*num_per_split].shape[0])
				split = split+1
			count = count+1
	print("num_split_lists: ", num_split_list)

def run_groundtruth():
	print("Making groudtruth file")
	import ctypes
	groundtruth_dir = dataset_basedir + "groundtruth/"
	if os.path.isdir(groundtruth_dir)!=True:
		os.mkdir(groundtruth_dir)
	queries = np.array(get_queries(), dtype='float32')
	groundtruth = np.empty([qN, 1000], dtype=np.int32)
	groundtruth_simil = np.empty([qN, 1000], dtype=np.float32)
	ypp_handles = [np.ctypeslib.as_ctypes(row) for row in queries]
	gpp_handles = [np.ctypeslib.as_ctypes(row) for row in groundtruth]
	gspp_handles = [np.ctypeslib.as_ctypes(row) for row in groundtruth_simil]
	ypp = (ctypes.POINTER(ctypes.c_float) * qN)(*ypp_handles)
	gpp = (ctypes.POINTER(ctypes.c_int) * qN)(*gpp_handles)
	gspp = (ctypes.POINTER(ctypes.c_float) * qN)(*gspp_handles)
	if(args.num_split == -1):
		dataset = read_data(dataset_basedir, base=True, offset_=0, shape_=None).astype('float32')
		xpp_handles = [np.ctypeslib.as_ctypes(row) for row in dataset]
		xpp = (ctypes.POINTER(ctypes.c_float) * N)(*xpp_handles)

		libc = ctypes.CDLL('./groundtruth.so')
		libc.compute_groundtruth.restype=None
		libc.compute_groundtruth(0, N, D, qN, xpp, ypp, gpp, gspp, True if args.metric=="dot_product" else False)
		write_gt_data(groundtruth)
	else:
		for num in range(args.num_split):
			print("Working on", str(num+1), "th out of", str(args.num_split), "splits...")
			num_per_split = int(N/args.num_split)
			partial_split_dataset_path = split_dataset_path+str(args.num_split)+"_"+str(num)
			dataset = read_data(partial_split_dataset_path, base=False, offset_=0, shape_=None).astype('float32')
			xpp_handles = [np.ctypeslib.as_ctypes(row) for row in dataset]
			xpp = (ctypes.POINTER(ctypes.c_float) * N)(*xpp_handles)

			libc = ctypes.CDLL('./groundtruth.so')
			libc.compute_groundtruth.restype=None
			libc.compute_groundtruth(num, num_per_split, D, qN, xpp, ypp, gpp, gspp, True if args.metric=="dot_product" else False)
		split_gt_path = groundtruth_path
		write_split_gt_data(split_gt_path, groundtruth)

def sort_neighbors(distances, neighbors):
	if "dot_product" == args.metric or "angular" == args.metric:
		return np.take_along_axis(neighbors, np.argsort(-distances, axis=-1), -1)[:,:,:args.topk], -np.sort(-distances, axis=-1)[:,:,:args.topk]
	elif "squared_l2" == args.metric:
		return np.take_along_axis(neighbors, np.argsort(distances, axis=-1), -1)[:,:,:args.topk], np.sort(distances, axis=-1)[:,:,:args.topk]
	else:
		assert False

def prepare_eval():
	gt = get_groundtruth()
	queries = get_queries()
	assert gt.shape[1] == 1000
	return gt, queries

def print_recall(final_neighbors, gt):
	print("final_neighbors :", final_neighbors.shape)
	print("gt :", gt.shape)
	top1 = compute_recall(final_neighbors[:,:1], gt[:, :1])
	top10 = compute_recall(final_neighbors[:,:10], gt[:, :10])
	top100 = compute_recall(final_neighbors[:,:100], gt[:, :100])
	top1000 = compute_recall(final_neighbors[:,:1000], gt[:, :1000])
	top1000_10000 = compute_recall(final_neighbors[:,:10000], gt[:, :1000])
	print("Recall 1@1:", top1)
	print("Recall 10@10:", top10)
	print("Recall 100@100:", top100)
	print("Recall 1000@1000:", top1000)
	print("Recall 1000@10000:", top1000_10000)
	return top1, top10, top100, top1000

def get_searcher_path(split):
	searcher_dir = basedir + args.program + ("GPU_" if args.is_gpu else "_") + 'searcher_' + args.metric + '/' + args.dataset + '/Split_' + str(args.num_split) + '/'
	searcher_path = searcher_dir + args.dataset + '_searcher_' + str(args.num_split)+'_'+str(split)
	return searcher_dir, searcher_path

def check_available_search_config(program, bc, search_config):
	sc_list = list()
	if program == "scann":
		num_leaves, threshold, dims, metric = bc
		for idx, sc in enumerate(search_config):
			leaves_to_search = sc[0]
			if leaves_to_search > num_leaves or (D%dims!=0 and args.sweep==True):
				continue
			else:
				sc_list.append(idx)
	elif program == "faiss":
		L, m, log2kstar, metric = bc
		for idx, sc in enumerate(search_config):
			nprobe, args.reorder = sc[0], sc[1]
			if nprobe > L or (D%m!=0 and args.sweep==True) or m > 96 or (not args.is_gpu and log2kstar>8) or (args.is_gpu and log2kstar != 8):
				continue
			else:
				sc_list.append(idx)
	else:
		assert False
	
	return sc_list

def run_scann():
	gt, queries = prepare_eval()
	if args.sweep:
		if "sift1b" in args.dataset:
			# For sift 1b
			# build_config = [[7000, 0.55, 2, args.metric], [7000, 0.2, 4, args.metric], [7000, 0.2, 2, args.metric], [7000, 0.2, 1, args.metric], \
			build_config = [[7000, 0.2, 4, args.metric], [7000, 0.2, 1, args.metric], \
							[8000, 0.55, 2, args.metric], [8000, 0.2, 4, args.metric], [8000, 0.2, 2, args.metric], [8000, 0.2, 1, args.metric], \
							[6000, 0.55, 2, args.metric], [6000, 0.2, 4 , args.metric], [6000, 0.2, 2, args.metric], [6000, 0.2, 1, args.metric]]
			search_config = [[1, args.reorder], [16, args.reorder], [32, args.reorder], [64, args.reorder], [128, args.reorder], \
							 [256, args.reorder], [320, args.reorder], [384, args.reorder], [448, args.reorder], [512, args.reorder], [576, args.reorder], [640, args.reorder], [704, args.reorder], [768, args.reorder], \
							 [1024, args.reorder], [1280, args.reorder], [1536, args.reorder], [2048, args.reorder], [2560, args.reorder], [3072, args.reorder], [4096, args.reorder], [4608, args.reorder], \
							 [5120, args.reorder], [5632, args.reorder], [6144, args.reorder], [6656, args.reorder], [7168, args.reorder], [7680, args.reorder], \
							 [8192, args.reorder], [16384, args.reorder]]
		else:
			build_config = [
							[800, 0.2, 1, args.metric], [800, 0.2, 2, args.metric], [800, 0.2, 3, args.metric], [800, 0.2, 4, args.metric], [800, 0.2, 5, args.metric], [800, 0.2, 8, args.metric], [800, 0.2, 10, args.metric], [800, 0.2, 16, args.metric], [800, 0.2, 25, args.metric], [800, 0.2, 32, args.metric], [800, 0.2, 50, args.metric], [800, 0.2, 64, args.metric], \
							[800, 0.4, 1, args.metric], [800, 0.4, 2, args.metric], [800, 0.4, 3, args.metric], [800, 0.4, 4, args.metric], [800, 0.4, 5, args.metric], [800, 0.4, 8, args.metric], [800, 0.4, 10, args.metric], [800, 0.4, 16, args.metric], [800, 0.4, 25, args.metric], [800, 0.4, 32, args.metric], [800, 0.4, 50, args.metric], [800, 0.4, 64, args.metric], \
							[800, 0.55, 1, args.metric], [800, 0.55, 2, args.metric], [800, 0.55, 3, args.metric], [800, 0.55, 4, args.metric], [800, 0.55, 5, args.metric], [800, 0.55, 8, args.metric], [800, 0.55, 10, args.metric], [800, 0.55, 16, args.metric], [800, 0.55, 25, args.metric], [800, 0.55, 32, args.metric], [800, 0.55, 50, args.metric], [800, 0.55, 64, args.metric], \
							[1000, 0.2, 1, args.metric], [1000, 0.2, 2, args.metric], [1000, 0.2, 3, args.metric], [1000, 0.2, 4, args.metric], [1000, 0.2, 5, args.metric], [1000, 0.2, 8, args.metric], [1000, 0.2, 10, args.metric], [1000, 0.2, 16, args.metric], [1000, 0.2, 25, args.metric], [1000, 0.2, 32, args.metric], [1000, 0.2, 50, args.metric], [1000, 0.2, 64, args.metric], \
							[1000, 0.4, 1, args.metric], [1000, 0.4, 2, args.metric], [1000, 0.4, 3, args.metric], [1000, 0.4, 4, args.metric], [1000, 0.4, 5, args.metric], [1000, 0.4, 8, args.metric], [1000, 0.4, 10, args.metric], [1000, 0.4, 16, args.metric], [1000, 0.4, 25, args.metric], [1000, 0.4, 32, args.metric], [1000, 0.4, 50, args.metric], [1000, 0.4, 64, args.metric], \
							[1000, 0.55, 1, args.metric], [1000, 0.55, 2, args.metric], [1000, 0.55, 3, args.metric], [1000, 0.55, 4, args.metric], [1000, 0.55, 5, args.metric], [1000, 0.55, 8, args.metric], [1000, 0.55, 10, args.metric], [1000, 0.55, 16, args.metric], [1000, 0.55, 25, args.metric], [1000, 0.55, 32, args.metric], [1000, 0.55, 50, args.metric], [1000, 0.55, 64, args.metric], \
							[1500, 0.2, 1, args.metric], [1500, 0.2, 2, args.metric], [1500, 0.2, 3, args.metric], [1500, 0.2, 4, args.metric], [1500, 0.2, 5, args.metric], [1500, 0.2, 8, args.metric], [1500, 0.2, 10, args.metric], [1500, 0.2, 16, args.metric], [1500, 0.2, 25, args.metric], [1500, 0.2, 32, args.metric], [1500, 0.2, 50, args.metric], [1500, 0.2, 64, args.metric], \
							[1500, 0.4, 1, args.metric], [1500, 0.4, 2, args.metric], [1500, 0.4, 3, args.metric], [1500, 0.4, 4, args.metric], [1500, 0.4, 5, args.metric], [1500, 0.4, 8, args.metric], [1500, 0.4, 10, args.metric], [1500, 0.4, 16, args.metric], [1500, 0.4, 25, args.metric], [1500, 0.4, 32, args.metric], [1500, 0.4, 50, args.metric], [1500, 0.4, 64, args.metric], \
							[1500, 0.55, 1, args.metric], [1500, 0.55, 2, args.metric], [1500, 0.55, 3, args.metric], [1500, 0.55, 4, args.metric], [1500, 0.55, 5, args.metric], [1500, 0.55, 8, args.metric], [1500, 0.55, 10, args.metric], [1500, 0.55, 16, args.metric], [1500, 0.55, 25, args.metric], [1500, 0.55, 32, args.metric], [1500, 0.55, 50, args.metric], [1500, 0.55, 64, args.metric], \
							[2000, 0.2, 1, args.metric], [2000, 0.2, 2, args.metric], [2000, 0.2, 3, args.metric], [2000, 0.2, 4, args.metric], [2000, 0.2, 5, args.metric], [2000, 0.2, 8, args.metric], [2000, 0.2, 10, args.metric], [2000, 0.2, 16, args.metric], [2000, 0.2, 25, args.metric], [2000, 0.2, 32, args.metric], [2000, 0.2, 50, args.metric], [2000, 0.2, 64, args.metric], \
							[2000, 0.4, 1, args.metric], [2000, 0.4, 2, args.metric], [2000, 0.4, 3, args.metric], [2000, 0.4, 4, args.metric], [2000, 0.4, 5, args.metric], [2000, 0.4, 8, args.metric], [2000, 0.4, 10, args.metric], [2000, 0.4, 16, args.metric], [2000, 0.4, 25, args.metric], [2000, 0.4, 32, args.metric], [2000, 0.4, 50, args.metric], [2000, 0.4, 64, args.metric], \
							[2000, 0.55, 1, args.metric], [2000, 0.55, 2, args.metric], [2000, 0.55, 3, args.metric], [2000, 0.55, 4, args.metric], [2000, 0.55, 5, args.metric], [2000, 0.55, 8, args.metric], [2000, 0.55, 10, args.metric], [2000, 0.55, 16, args.metric], [2000, 0.55, 25, args.metric], [2000, 0.55, 32, args.metric], [2000, 0.55, 50, args.metric], [2000, 0.55, 64, args.metric], \
							[4000, 0.2, 1, args.metric],[4000, 0.2, 2, args.metric], [4000, 0.2, 3, args.metric], [4000, 0.2, 4, args.metric], [4000, 0.2, 5, args.metric], [4000, 0.2, 8, args.metric], [4000, 0.2, 10, args.metric], [4000, 0.2, 16, args.metric], [4000, 0.2, 25, args.metric], [4000, 0.2, 32, args.metric], [4000, 0.2, 50, args.metric], [4000, 0.2, 64, args.metric], \
							[4000, 0.4, 1, args.metric], [4000, 0.4, 2, args.metric], [4000, 0.4, 3, args.metric], [4000, 0.4, 4, args.metric], [4000, 0.4, 5, args.metric], [4000, 0.4, 8, args.metric], [4000, 0.4, 10, args.metric], [4000, 0.4, 16, args.metric], [4000, 0.4, 25, args.metric], [4000, 0.4, 32, args.metric], [4000, 0.4, 50, args.metric], [4000, 0.4, 64, args.metric], \
							[4000, 0.55, 1, args.metric], [4000, 0.55, 2, args.metric], [4000, 0.55, 3, args.metric], [4000, 0.55, 4, args.metric], [4000, 0.55, 5, args.metric], [4000, 0.55, 8, args.metric], [4000, 0.55, 10, args.metric], [4000, 0.55, 16, args.metric], [4000, 0.55, 25, args.metric], [4000, 0.55, 32, args.metric], [4000, 0.55, 50, args.metric], [4000, 0.55, 64, args.metric]]


			# build_config = [[2000, 0.2, 2, args.metric], [2000, 0.2, 1, args.metric], [1500, 0.55, 2, args.metric], [1500, 0.55, 1, args.metric], [1000, 0.55, 2, args.metric], [1000, 0.55, 1, args.metric], \
			#  				  [1000, 0.2, 2, args.metric], [1000, 0.2, 1, args.metric], [1400, 0.15, 1, args.metric], [1400, 0.15, 2, args.metric], [1400, 0.15, 3, args.metric], \
			#  				  [800, 0.15, 2, args.metric], [800, 0.15, 1, args.metric]]
			# build_config = [[1400, 0.15, 3, args.metric], \
			#  				  [800, 0.15, 2, args.metric], [800, 0.15, 1, args.metric]]
			search_config = [[1, args.reorder], [2, args.reorder], [4, args.reorder], [8, args.reorder], [16, args.reorder], [25, args.reorder], [30, args.reorder], [35, args.reorder], [40, args.reorder], \
							 [45, args.reorder], [50, args.reorder], [55, args.reorder], [60, args.reorder], [65, args.reorder], [75, args.reorder], [90, args.reorder], [110, args.reorder], [130, args.reorder], [150, args.reorder], \
							 [170, args.reorder], [200, args.reorder], [220, args.reorder], [250, args.reorder], [310, args.reorder], [400, args.reorder], [500, args.reorder], [800, args.reorder], [1000, args.reorder], \
							 [1250, args.reorder], [1500, args.reorder], [1750, args.reorder], [1900, args.reorder], [2000, args.reorder], [2048, args.reorder]]

		f = open(sweep_result_path, "w")
		f.write("Program: " + args.program + " Topk: " + str(args.topk) + " Num_split: " + str(args.num_split)+ " Batch: "+str(args.batch)+"\n")
		f.write("L\tThreashold\tm\t|\tw\tr\tMetric\n")
	else:
		# assert D%args.m == 0
		build_config = [(args.L, args.threshold, int(D/args.m), args.metric)]
		search_config = [[args.w, args.reorder]]

	for bc in build_config:
		num_leaves, threshold, dims, metric = bc
		sc_list = check_available_search_config(args.program, bc, search_config)
		neighbors=np.empty((len(sc_list), queries.shape[0],0), dtype=np.int32)
		distances=np.empty((len(sc_list), queries.shape[0],0), dtype=np.float32)
		total_latency = np.zeros(len(sc_list))
		base_idx = 0
		if len(sc_list) > 0:
			for split in range(args.num_split):

				num_per_split = int(N/args.num_split) if split < args.num_split-1 else N-base_idx
				searcher_dir, searcher_path = get_searcher_path(split)
				print("Split ", split)
				# Load splitted dataset
				batch_size = min(args.batch, queries.shape[0])
				searcher = None
				searcher_path = searcher_path + '_' + str(num_leaves) + '_' + str(threshold) + '_' + str(dims) + '_' + metric + ("_reorder" if args.reorder!=-1 else '')

				if os.path.isdir(searcher_path):
					print("Loading searcher from ", searcher_path)
					searcher = scann.scann_ops_pybind.load_searcher(searcher_path, num_per_split, D)
				else:
					# Create ScaNN searcher
					print("Entering ScaNN builder, will be created to ", searcher_path)
					dataset = read_data(split_dataset_path + str(args.num_split) + "_" + str(split) if args.num_split>1 else dataset_basedir, base=False if args.num_split>1 else True, offset_=None if args.num_split>1 else 0, shape_=None)
					if args.reorder!=-1:
						searcher = scann.scann_ops_pybind.builder(dataset, 10, metric).tree(
							num_leaves=num_leaves, num_leaves_to_search=num_leaves, training_sample_size=args.coarse_training_size).score_ah(
							dims, anisotropic_quantization_threshold=threshold, training_sample_size=args.fine_training_size).reorder(args.reorder).build()
					else:
						searcher = scann.scann_ops_pybind.builder(dataset, 10, metric).tree(
								num_leaves=num_leaves, num_leaves_to_search=num_leaves, training_sample_size=args.coarse_training_size).score_ah(
								dims, anisotropic_quantization_threshold=threshold, training_sample_size=args.fine_training_size).build()
					print("Saving searcher to ", searcher_path)
					os.makedirs(searcher_path, exist_ok=True)
					searcher.serialize(searcher_path)
				print("sc_list: ", sc_list)
				n = list()
				d = list()
				for idx in range(len(sc_list)):
					leaves_to_search, reorder = search_config[sc_list[idx]]
					assert D%dims == 0

					if args.reorder!=-1:
						assert args.topk <= reorder
					else:
						if args.sweep:
							assert False, "Do you want reordering or not?"

					print(str(num_leaves)+"\t"+str(threshold)+"\t"+str(int(D/dims))+"\t|\t"+str(leaves_to_search)+"\t"+str(reorder)+"\t"+str(metric)+"\n")
					if args.batch > 1:
						start = time.time()
						local_neighbors, local_distances = searcher.search_batched_parallel(queries, leaves_to_search=leaves_to_search, pre_reorder_num_neighbors=reorder, final_num_neighbors=args.topk, batch_size=batch_size)
						# local_neighbors, local_distances = searcher.search_batched(queries, leaves_to_search=leaves_to_search, pre_reorder_num_neighbors=reorder, final_num_neighbors=args.topk)
						end = time.time()
						local_distances[local_neighbors==2147483647] =  math.inf if metric=="squared_l2" else -math.inf 		# 2147483647: maximum integer value
						total_latency[idx] = total_latency[idx] + 1000*(end - start)
						n.append(local_neighbors+base_idx)
						d.append(local_distances)
					else:
						# ScaNN search
						def single_query(query, base_idx):
							start = time.time()
							local_neighbors, local_distances = searcher.search(query, leaves_to_search=leaves_to_search, pre_reorder_num_neighbors=reorder, final_num_neighbors=args.topk)
							local_distances[local_neighbors==2147483647] =  math.inf if metric=="squared_l2" else -math.inf 		# 2147483647: maximum integer value
							return (time.time() - start, (local_neighbors, local_distances))
						# ScaNN search
						print("Entering ScaNN searcher")
						local_results = [single_query(q, base_idx) for q in queries]
						total_latency[idx] += (np.sum(np.array([time for time, _ in local_results]).reshape(queries.shape[0], 1)))*1000
						nd = [nd for _, nd in local_results]
						n.append(np.vstack([n for n,d in nd])+base_idx)
						d.append(np.vstack([d for n,d in nd]))
				base_idx = base_idx + num_per_split
				neighbors = np.append(neighbors, np.array(n, dtype=np.int32), axis=-1)
				distances = np.append(distances, np.array(d, dtype=np.float32), axis=-1)
				# print("type(neighbors): ", type(neighbors))
				# print("type(distances): ", type(distances))
				neighbors, distances = sort_neighbors(distances, neighbors)
				print("neighbors: ", neighbors.shape)
				print("distances: ", distances.shape)

			final_neighbors, _ = sort_neighbors(distances, neighbors)
			for idx in range(len(sc_list)):
				if args.sweep:
					leaves_to_search, reorder = search_config[sc_list[idx]]
					f.write(str(num_leaves)+"\t"+str(threshold)+"\t"+str(int(D/dims))+"\t|\t"+str(leaves_to_search)+"\t"+str(reorder)+"\t"+str(metric)+"\n")
				print(str(num_leaves)+"\t"+str(threshold)+"\t"+str(int(D/dims))+"\t|\t"+str(leaves_to_search)+"\t"+str(reorder)+"\t"+str(metric)+"\n")
				top1, top10, top100, top1000 = print_recall(final_neighbors[idx], gt)
				print("Top ", args.topk, " Total latency (ms): ", total_latency[idx])
				if args.sweep:
					f.write(str(top1)+" %\t"+str(top10)+" %\t"+str(top100)+" %\t"+str(top1000)+" %\t"+str(total_latency[idx])+"\n")
	if args.sweep:
		f.close()


def faiss_pad_dataset(dataset, train_dataset, queries, m):
	D = dataset.shape[1]
	if m==1 or m==2 or m==3 or m==4 or m==8 or m==12 or m==16 or m==20 or m==24 or m==28 or m==32 or m==40 or m==48 or m==56 or m==64 or m==96:
		return D, m, dataset, train_dataset, queries
	else:
		dim_per_block = int(D/m)
		if m<8:		# 4<m<8
			faiss_m = 8
		elif m<12:
			faiss_m = 12
		elif m<16:
			faiss_m = 16
		elif m<20:
			faiss_m = 20
		elif m<24:
			faiss_m = 24
		elif m<28:
			faiss_m = 28
		elif m<32:
			faiss_m = 32
		elif m<40:
			faiss_m = 40
		elif m<48:
			faiss_m = 48
		elif m<56:
			faiss_m = 56
		elif m<64:
			faiss_m = 64
		elif m<96:
			faiss_m = 96
		else:
			assert False, "somethings wrong.."
		padded_D = dim_per_block * faiss_m
		plus_dim = padded_D-D
		dataset=np.concatenate((dataset, np.full((dataset.shape[0], plus_dim), 0, dtype='float32')), axis=-1)
		train_dataset=np.concatenate((train_dataset, np.full((train_dataset.shape[0], plus_dim), 0)), axis=-1)
		queries=np.concatenate((queries, np.full((queries.shape[0], plus_dim), 0)), axis=-1)
		print("Dataset dimension is padded from ", D, " to ", dataset.shape[1])

		return padded_D, faiss_m, dataset, train_dataset, queries

def run_faiss(D):
	gt, queries = prepare_eval()
	if args.sweep:
		if args.is_gpu:
			log2kstar_ = 8
			build_config = [[800, int(D/64), log2kstar_, args.metric], [800, int(D/50), log2kstar_, args.metric], [800, int(D/32), log2kstar_, args.metric], [800, int(D/25), log2kstar_, args.metric], [800, int(D/16), log2kstar_, args.metric], [800, int(D/10), log2kstar_, args.metric], [800, int(D/8), log2kstar_, args.metric], [800, int(D/5), log2kstar_, args.metric], [800, int(D/4), log2kstar_, args.metric], [800, int(D/3), log2kstar_, args.metric], [800, int(D/2), log2kstar_, args.metric], [800, D, log2kstar_, args.metric], \
							[1000, int(D/64), log2kstar_, args.metric], [1000, int(D/50), log2kstar_, args.metric], [1000, int(D/32), log2kstar_, args.metric], [1000, int(D/25), log2kstar_, args.metric], [1000, int(D/16), log2kstar_, args.metric], [1000, int(D/10), log2kstar_, args.metric], [1000, int(D/8), log2kstar_, args.metric], [1000, int(D/5), log2kstar_, args.metric], [1000, int(D/4), log2kstar_, args.metric], [1000, int(D/3), log2kstar_, args.metric], [1000, int(D/2), log2kstar_, args.metric], [1000, D, log2kstar_, args.metric], \
							[1500, int(D/64), log2kstar_, args.metric], [1500, int(D/50), log2kstar_, args.metric], [1500, int(D/32), log2kstar_, args.metric], [1500, int(D/25), log2kstar_, args.metric], [1500, int(D/16), log2kstar_, args.metric], [1500, int(D/10), log2kstar_, args.metric], [1500, int(D/8), log2kstar_, args.metric], [1500, int(D/5), log2kstar_, args.metric], [1500, int(D/4), log2kstar_, args.metric], [1500, int(D/3), log2kstar_, args.metric], [1500, int(D/2), log2kstar_, args.metric], [1500, D, log2kstar_, args.metric], \
							[2000, int(D/64), log2kstar_, args.metric], [2000, int(D/50), log2kstar_, args.metric], [2000, int(D/32), log2kstar_, args.metric], [2000, int(D/25), log2kstar_, args.metric], [2000, int(D/16), log2kstar_, args.metric], [2000, int(D/10), log2kstar_, args.metric], [2000, int(D/8), log2kstar_, args.metric], [2000, int(D/5), log2kstar_, args.metric], [2000, int(D/4), log2kstar_, args.metric], [2000, int(D/3), log2kstar_, args.metric], [2000, int(D/2), log2kstar_, args.metric], [2000, D, log2kstar_, args.metric], \
							[4000, int(D/64), log2kstar_, args.metric], [4000, int(D/50), log2kstar_, args.metric], [4000, int(D/32), log2kstar_, args.metric], [4000, int(D/25), log2kstar_, args.metric], [4000, int(D/16), log2kstar_, args.metric], [4000, int(D/10), log2kstar_, args.metric], [4000, int(D/8), log2kstar_, args.metric], [4000, int(D/5), log2kstar_, args.metric], [4000, int(D/4), log2kstar_, args.metric], [4000, int(D/3), log2kstar_, args.metric], [4000, int(D/2), log2kstar_, args.metric], [4000, D, log2kstar_, args.metric]]
					
		else:
			build_config = [[1000, int(D/32), 4, args.metric], [1000, int(D/16), 4, args.metric], [1000, int(D/8), 4, args.metric], [1000, int(D/4), 4, args.metric], [1000, int(D/3), 4, args.metric], [1000, int(D/2), 4, args.metric], [1000, D, 4, args.metric], \
							[1000, int(D/32), 6, args.metric], [1000, int(D/16), 6, args.metric], [1000, int(D/8), 6, args.metric], [1000, int(D/4), 6, args.metric], [1000, int(D/3), 6, args.metric], [1000, int(D/2), 6, args.metric], [1000, D, 6, args.metric], \
							[1000, int(D/32), 8, args.metric], [1000, int(D/16), 8, args.metric], [1000, int(D/8), 8, args.metric], [1000, int(D/4), 8, args.metric], [1000, int(D/3), 8, args.metric], [1000, int(D/2), 8, args.metric], [1000, D, 8, args.metric], \
							[2000, int(D/32), 4, args.metric], [2000, int(D/16), 4, args.metric], [2000, int(D/8), 4, args.metric], [2000, int(D/4), 4, args.metric], [2000, int(D/3), 4, args.metric], [2000, int(D/2), 4, args.metric], [2000, D, 4, args.metric], \
							[2000, int(D/32), 6, args.metric], [2000, int(D/16), 6, args.metric], [2000, int(D/8), 6, args.metric], [2000, int(D/4), 6, args.metric], [2000, int(D/3), 6, args.metric], [2000, int(D/2), 6, args.metric], [2000, D, 6, args.metric], \
							[2000, int(D/32), 8, args.metric], [2000, int(D/16), 8, args.metric], [2000, int(D/8), 8, args.metric], [2000, int(D/4), 8, args.metric], [2000, int(D/3), 8, args.metric], [2000, int(D/2), 8, args.metric], [2000, D, 8, args.metric], \
							[800, int(D/32), 4, args.metric], [800, int(D/16), 4, args.metric], [800, int(D/8), 4, args.metric], [800, int(D/4), 4, args.metric], [800, int(D/3), 4, args.metric], [800, int(D/2), 4, args.metric], [800, D, 4, args.metric], \
							[800, int(D/32), 6, args.metric], [800, int(D/16), 6, args.metric], [800, int(D/8), 6, args.metric], [800, int(D/4), 6, args.metric], [800, int(D/3), 6, args.metric], [800, int(D/2), 6, args.metric], [800, D, 6, args.metric], \
							[800, int(D/32), 8, args.metric], [800, int(D/16), 8, args.metric], [800, int(D/8), 8, args.metric], [800, int(D/4), 8, args.metric], [800, int(D/3), 8, args.metric], [800, int(D/2), 8, args.metric], [800, D, 8, args.metric], \
							]	# L, m, log2(k*), metric

		search_config = [[1, args.reorder], [2, args.reorder], [4, args.reorder], [8, args.reorder], [16, args.reorder], [25, args.reorder], [130, args.reorder], [35, args.reorder], [40, args.reorder], \
						 [45, args.reorder], [50, args.reorder], [55, args.reorder], [60, args.reorder], [65, args.reorder], [75, args.reorder], [90, args.reorder], [110, args.reorder], [130, args.reorder], [150, args.reorder], \
						 [170, args.reorder], [200, args.reorder], [220, args.reorder], [250, args.reorder], [310, args.reorder], [400, args.reorder], [500, args.reorder], [800, args.reorder], [1000, args.reorder], \
						 [1250, args.reorder], [1500, args.reorder], [1750, args.reorder], [1900, args.reorder], [2000, args.reorder]]

		f = open(sweep_result_path, "w")
		f.write("Program: " + args.program + ("GPU" if args.is_gpu else "") + " Topk: " + str(args.topk) + " Num_split: " + str(args.num_split)+ " Batch: "+str(args.batch)+"\n")
		f.write("L\tm\tk_star\t|\tw\tMetric\n")
	else:
		build_config = [[args.L, args.m, math.log(args.k_star,2), args.metric]]
		search_config = [[args.w, args.reorder]]
	for bc in build_config:
		L, m, log2kstar, metric = bc
		# assert (not args.is_gpu and log2kstar<=8) or (log2kstar == 8)
		sc_list = check_available_search_config(args.program, bc, search_config)
		neighbors=np.empty((len(sc_list), queries.shape[0],0), dtype=np.int32)
		distances=np.empty((len(sc_list), queries.shape[0],0), dtype=np.float32)
		base_idx = 0
		total_latency = np.zeros(len(sc_list))
		if len(sc_list) > 0:
			for split in range(args.num_split):
				print("Split ", split)
				num_per_split = int(N/args.num_split) if split < args.num_split-1 else N-base_idx
				searcher_dir, searcher_path = get_searcher_path(split)
				args.batch = min(args.batch, queries.shape[0])
				# Load splitted dataset
				train_dataset = get_train(split, args.num_split)
				dataset = read_data(split_dataset_path + str(args.num_split) + "_" + str(split) if args.num_split>1 else dataset_basedir, base=False if args.num_split>1 else True, offset_=None if args.num_split>1 else 0, shape_=None)
				padded_D, faiss_m, padded_dataset, padded_train_dataset, padded_queries = faiss_pad_dataset(dataset, train_dataset, queries, m)
				# local_neighbors, local_distances, total_latency = run_local_faiss(args, searcher_dir, split, padded_D, "IVF"+str(L)+",PQ"+str(faiss_m)+"x"+str(log2kstar), padded_train_dataset, padded_dataset, padded_queries)
				index, preproc = build_faiss(args, searcher_dir, split, padded_D, "IVF"+str(L)+",PQ"+str(faiss_m)+"x"+str(log2kstar), padded_train_dataset, padded_dataset, padded_queries)

				n = list()
				d = list()
				for idx in range(len(sc_list)):
					w, reorder = search_config[sc_list[idx]]
					assert reorder == args.reorder
					# Build Faiss index
					# args.w = nprobe


					# if args.sweep:
					# 	f.write(str(L)+"\t"+str(m)+"\t"+str(2**log2kstar)+"\t|\t"+str(nprobe)+"\t"+str(args.reorder)+"\t"+str(metric)+"\n")		# faiss-gpu has no reorder
					print(str(L)+"\t"+str(m)+"\t"+str(2**log2kstar)+"\t|\t"+str(w)+"\t"+str(reorder)+"\t"+str(metric)+"\n")		# faiss-gpu has no reorder
					# Faiss search
					local_neighbors, local_distances, total_latency[idx] = faiss_search(index, preproc, args, reorder, w)
					n.append((local_neighbors+base_idx).astype(np.int32))
					d.append(local_distances.astype(np.float32))	

				base_idx = base_idx + num_per_split
				neighbors = np.append(neighbors, np.array(n, dtype=np.int32), axis=-1)
				distances = np.append(distances, np.array(d, dtype=np.float32), axis=-1)
				neighbors, distances = sort_neighbors(distances, neighbors)
				print("neighbors: ", neighbors.shape)
				print("distances: ", distances.shape)

			final_neighbors, _ = sort_neighbors(distances, neighbors)
			for idx in range(len(sc_list)):
				if args.sweep:
					w, reorder = search_config[sc_list[idx]]
					f.write(str(L)+"\t"+str(m)+"\t"+str(2**log2kstar)+"\t|\t"+str(w)+"\t"+str(reorder)+"\t"+str(metric)+"\n")		# faiss-gpu has no reorder

				top1, top10, top100, top1000 = print_recall(final_neighbors[idx], gt)
				print("Top ", args.topk, " Total latency (ms): ", total_latency[idx])
				if args.sweep:
					f.write(str(top1)+" %\t"+str(top10)+" %\t"+str(top100)+" %\t"+str(top1000)+" %\t"+str(total_latency)+"\n")

	if args.sweep:
		f.close()

def run_annoy(D):
	gt, queries = prepare_eval()
	assert args.metric!='angular', "[TODO] don't understand how angular works yet..."
	if args.sweep:
		build_config = [(args.metric, 50), (args.metric, 100), (args.metric, 150), (args.metric, 200), (args.metric, 250), (args.metric, 300), (args.metric, 400)]
		search_config = [100, 200, 400, 1000, 2000, 4000, 10000, 20000, 40000, 100000, 200000, 400000]
		f = open(sweep_result_path, "w")
		f.write("Program: " + args.program + " Topk: " + str(args.topk) + " Num_split: " + str(args.num_split)+ " Batch: "+str(args.batch)+"\n")
		f.write("Num trees\t|\tNum search\tReorder\tMetric\n")
	else:
		build_config = [(args.metric, args.n_trees)]
		search_config = [args.num_search]
	for bc in build_config:
		metric, n_trees = bc
		if metric == "dot_product":
			annoy_metric = "dot"
		elif metric == "squared_l2":
			annoy_metric = "euclidean"
		elif metric == "angular":
			annoy_metric = "angular"

		neighbors = np.empty((len(search_config), queries.shape[0],0), dtype=np.int32)
		distances = np.empty((len(search_config), queries.shape[0],0), dtype=np.float32)
		total_latency = np.zeros(len(search_config))
		base_idx = 0
		for split in range(args.num_split):
			num_per_split = int(N/args.num_split) if split < args.num_split-1 else N-base_idx
			searcher_dir, searcher_path = get_searcher_path(split)
			searcher_path = searcher_path + '_' + str(n_trees) + '_' + metric
			print("Split ", split)

			# Create Annoy index
			searcher = annoy.AnnoyIndex(D, metric=annoy_metric)
			if os.path.isfile(searcher_path):
				print("Loading searcher from ", searcher_path)
				searcher.load(searcher_path)
			else:
				# Load splitted dataset
				dataset = read_data(split_dataset_path + str(args.num_split) + "_" + str(split) if args.num_split>1 else dataset_basedir, base=False if args.num_split>1 else True, offset_=None if args.num_split>1 else 0, shape_=None)
				print("Annoy, adding items")
				for i, x in enumerate(dataset):
				    searcher.add_item(i, x.tolist())
				print("Annoy, building trees")
				searcher.build(n_trees)
				print("Saving searcher to ", searcher_path)
				os.makedirs(searcher_dir, exist_ok=True)
				searcher.save(searcher_path)
			n = list()
			d = list()			
			for idx, sc in enumerate(search_config):
				num_search = sc
				# if args.sweep:
				# 	f.write(str(n_trees)+"\t"+str(num_search)+"\t"+str(annoy_metric)+"\n")
				print(str(n_trees)+"\t"+str(num_search)+"\t"+str(annoy_metric))
				print("Entering Annoy searcher")
				# Annoy batch version
				if args.batch > 1:
					pool = ThreadPool(args.batch)
					start = time.time()
					result = pool.map(lambda q: searcher.get_nns_by_vector(q.tolist(), args.topk, num_search, include_distances=True), queries)
					end = time.time()
					ne = np.empty((0, args.topk))
					di = np.empty((0, args.topk))
					for nn, dd in result:
						if len(nn) < args.topk:
							plus_dim = args.topk-len(nn)
							ne = np.append(ne, np.array(nn+[N]*plus_dim).reshape(1, args.topk), axis=0)
							di = np.append(di, np.array(dd+[math.inf if metric=="squared_l2" else -math.inf]*plus_dim).reshape(1, args.topk), axis=0)
						else:
							ne = np.append(ne, np.array(nn).reshape(1, args.topk), axis=0)
							di = np.append(di, np.array(dd).reshape(1, args.topk), axis=0)
					total_latency[idx] = total_latency[idx] + 1000*(end - start)
					n.append(ne+base_idx)
					d.append(di)
				else:
					def single_query(query, base_idx):
						start = time.time()
						local_neighbors, local_distances = searcher.get_nns_by_vector(query.tolist(), args.topk, num_search, include_distances=True)
						if len(local_neighbors) < args.topk:
							plus_dim = args.topk-len(local_neighbors)
							local_neighbors=np.concatenate((local_neighbors, np.full((plus_dim), N)), axis=-1)
							local_distances=np.concatenate((local_distances, np.full((plus_dim), math.inf if metric=="squared_l2" else -math.inf)), axis=-1)
						return (time.time() - start, (local_neighbors, local_distances))

					local_results = [single_query(q, base_idx) for q in queries]
					total_latency[idx]  += (np.sum(np.array([time for time, _ in local_results]).reshape(queries.shape[0], 1)))*1000
					nd = [nd for _, nd in local_results]
					n.append(np.vstack([n for n,d in nd])+base_idx)
					d.append(np.vstack([d for n,d in nd]))
			base_idx = base_idx + num_per_split
			neighbors = np.append(neighbors, np.array(n), axis=-1)
			distances = np.append(distances, np.array(d), axis=-1)
		final_neighbors = sort_neighbors(distances, neighbors)
		for idx in range(len(search_config)):
			top1, top10, top100, top1000 = print_recall(final_neighbors[idx], gt)
			print("Top ", args.topk, " Total latency (ms): ", total_latency[idx])
			if args.sweep:
				f.write(str(n_trees)+"\t|\t"+str(search_config[idx])+"\t-1\t"+str(annoy_metric)+"\n")
				f.write(str(top1)+" %\t"+str(top10)+" %\t"+str(top100)+" %\t"+str(top1000)+" %\t"+str(total_latency[idx])+"\n")
	if args.sweep:
		f.close()

# only for faiss
def get_train(split=-1, total=-1):
	if "sift1m" in args.dataset:
		filename = dataset_basedir + 'sift_learn.fvecs' if split<0 else dataset_basedir + 'split_data/sift1m_learn%d_%d' % (total, split)
		return mmap_fvecs(filename)
	elif "sift1b" in args.dataset:
		filename = dataset_basedir + 'bigann_learn.bvecs' if split<0 else dataset_basedir + 'split_data/sift1b_learn%d_%d' % (total, split)
		return bvecs_read(filename)
	elif "glove" in args.dataset:
		return np.array(h5py.File(dataset_basedir+"glove-100-angular.hdf5", "r")['test'], dtype='float32')
	else:
		assert False

def get_groundtruth():
	print("Reading grountruth from ", groundtruth_path)
	if os.path.isfile(groundtruth_path)==False:
		run_groundtruth()
	if "glove" in args.dataset:
		return read_data(groundtruth_path, base=False)
	else:
		return ivecs_read(groundtruth_path)

	# elif "sift1m" in args.dataset:
	# 	return ivecs_read(groundtruth_path)
	# 	# filename = dataset_basedir + 'sift_groundtruth.ivecs' if args.metric=="squared_l2" else groundtruth_path
	# 	# print("Reading from ", filename)
	# 	# return ivecs_read(filename)
	# elif "sift1b" in args.dataset:
	#  	filename = dataset_basedir +  'gnd/idx_1000M.ivecs' if args.metric=="squared_l2" else groundtruth_path
	#  	print("Reading from ", filename)
	#  	return ivecs_read(filename)
	# elif "glove" in args.dataset:
	# 	if args.metric == "dot_product":
	# 		print("Reading from ", dataset_basedir+"glove-100-angular.hdf5")
	# 		return h5py.File(dataset_basedir+"glove-100-angular.hdf5", "r")['neighbors']
	# 	else:
	# 		print("Reading from ", groundtruth_path)
	# 		return read_data(groundtruth_path, base=False)
	# else:
	#  	assert False

def get_queries():
	if "sift1m" in args.dataset:
		return mmap_fvecs(dataset_basedir + 'sift_query.fvecs')
	elif "sift1b" in args.dataset:
		return bvecs_read(dataset_basedir+'bigann_query.bvecs')
	elif "glove" in args.dataset:
		return np.array(h5py.File(dataset_basedir+"glove-100-angular.hdf5", "r")['test'], dtype='float32')
	else:
		assert False


if os.path.isdir("/arc-share"):
	basedir = "/arc-share/MICRO21_ANNA/"
else:
	basedir = "./"

os.makedirs("./result", exist_ok=True)
split_dataset_path = None
if args.sweep:
	sweep_result_path = "./result/"+args.program+("GPU_" if args.is_gpu else "_")+args.dataset+"_topk_"+str(args.topk)+"_num_split_"+str(args.num_split)+"_batch_"+str(args.batch)+"_"+args.metric+"_sweep_result.txt"
index_key = None
N = -1
D = -1
num_iter = -1
qN = -1

if "sift1m" in args.dataset:
	dataset_basedir = basedir + "SIFT1M/"
	split_dataset_path =dataset_basedir+"split_data/sift1m_"
	groundtruth_path = dataset_basedir + "sift1m_"+args.metric+"_gt"
	N=1000000
	D=128
	num_iter = 1
	qN = 10000
	index_key = "IVF4096,PQ64"
elif "sift1b" in args.dataset:
	dataset_basedir = basedir + "SIFT1B/"
	split_dataset_path = dataset_basedir+"split_data/sift1b_"
	groundtruth_path = dataset_basedir +  'gnd/idx_1000M.ivecs' if args.metric=="squared_l2" else dataset_basedir + "sift1b_"+args.metric+"_gt"
	N=1000000000
	D=128
	num_iter = 4
	qN = 10000
	index_key = "OPQ8_32,IVF262144,PQ8"
elif "glove" in args.dataset:
	dataset_basedir = basedir + "GLOVE/"
	split_dataset_path = dataset_basedir+"split_data/glove_"
	groundtruth_path = dataset_basedir + "glove_"+args.metric+"_gt"
	N=1183514
	D=100
	num_iter = 10
	qN = 10000


# main
if args.split:
	split(args.dataset, num_iter, N, D)
if args.eval_split or args.sweep:
	if args.program == "scann":
		run_scann()
	elif args.program == "faiss":
		run_faiss(D)
	elif args.program == "annoy":
		run_annoy(D)
	else:
		assert False

if args.groundtruth:
	run_groundtruth()
