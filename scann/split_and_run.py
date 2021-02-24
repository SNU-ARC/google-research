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
parser.add_argument('--dataset', type=str, help='sift1b, glove ...')
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
		assert args.L!=-1 and args.w!=-1 and args.topk!=-1 and args.k_star == -1 and args.m!=-1 and (args.topk <= args.reorder if args.reorder!=-1 else True)
	assert args.topk!=-1
elif args.program == "faiss":
	from runfaiss import train_faiss, build_faiss, search_faiss
	import math
	if args.sweep == False:
		assert args.L!=-1 and args.k_star!=-1 and args.w!=-1 and args.m!=-1
elif args.program == "annoy":
	import annoy
	if args.batch==True:
		from multiprocessing.pool import ThreadPool
	assert args.topk!=-1

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
	return x.reshape(-1, d + 1)[:, 1:].view('float32')

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
			dataset = dataset['train']			
			normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
			if offset_!=None and shape_!=None:
				return normalized_dataset[offset_:offset_+shape_]
			else:
				return normalized_dataset
		else:
			dataset = h5py.File(dataset_path, "r")
			return dataset['dataset']
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
	dataset = read_data(dataset_basedir, base=True, offset_=0, shape_=None).astype('float32')
	queries = np.array(get_queries(), dtype='float32')
	groundtruth = np.empty([qN, 1000], dtype=np.int32)
	xpp_handles = [np.ctypeslib.as_ctypes(row) for row in dataset]
	ypp_handles = [np.ctypeslib.as_ctypes(row) for row in queries]
	gpp_handles = [np.ctypeslib.as_ctypes(row) for row in groundtruth]
	xpp = (ctypes.POINTER(ctypes.c_float) * N)(*xpp_handles)
	ypp = (ctypes.POINTER(ctypes.c_float) * qN)(*ypp_handles)
	gpp = (ctypes.POINTER(ctypes.c_int) * qN)(*gpp_handles)

	libc = ctypes.CDLL('./groundtruth.so')
	libc.compute_groundtruth.restype=None
	libc.compute_groundtruth(N, D, qN, xpp, ypp, gpp, True if args.metric=="dot_product" else False)
	write_gt_data(groundtruth)

def sort_neighbors(distances, neighbors):
	if "dot_product" == args.metric or "angular" == args.metric:
		return np.take_along_axis(neighbors, np.argsort(-distances, axis=-1), -1)
	elif "squared_l2" == args.metric:
		return np.take_along_axis(neighbors, np.argsort(distances, axis=-1), -1)
	else:
		assert False

def prepare_eval():
	gt = get_groundtruth()
	queries = get_queries()
	return gt, queries

def print_recall(final_neighbors, gt):
	top1 = compute_recall(final_neighbors[:,:1], gt[:, :1])
	top10 = compute_recall(final_neighbors[:,:10], gt[:, :10])
	top100 = compute_recall(final_neighbors[:,:100], gt[:, :100])
	top1000 = compute_recall(final_neighbors[:,:1000], gt[:, :1000])
	top1000_10000 = compute_recall(final_neighbors[:,:10000], gt[:, :1000])
	print("Recall 1@1:", top1)
	print("Recall 10@10:", top10)
	print("Recall 100@100:", top100)
	print("Recall 1000@1000:", top100)
	print("Recall 1000@10000:", top1000)
	return top1, top10, top100, top1000

def get_searcher_path(split):
	searcher_dir = basedir + args.program + '_searcher_' + args.metric + '/' + args.dataset + '/Split_' + str(args.num_split) + '/'
	searcher_path = searcher_dir + args.dataset + '_searcher_' + str(args.num_split)+'_'+str(split)
	return searcher_dir, searcher_path

def run_scann():
	gt, queries = prepare_eval()
	if args.sweep:
		build_config = [(2000, 0.2, 2, args.metric), (2000, 0.2, 1, args.metric), (1500, 0.55, 2, args.metric), (1500, 0.55, 1, args.metric), (1000, 0.55, 2, args.metric), (1000, 0.55, 1, args.metric), (1000, 0.2, 2, args.metric), (1000, 0.2, 1, args.metric), (1400, 0.15, 1, args.metric), (1400, 0.15, 2, args.metric), (1400, 0.15, 3, args.metric), (800, 0.15, 2, args.metric), (800, 0.15, 1, args.metric)]
		# search_config = [[[1, 30], [2, 30], [4, 30], [8, 30], [30, 120], [35, 100], [40, 80], [45, 80], [50, 80], [55, 95], [60, 110], [65, 110], [75, 110], [90, 110], [110, 120], [130, 150], [150, 200], [170, 200], [200, 300], [220, 500], [250, 500], [310, 300], [400, 300], [500, 500], [800, 1000]],\
		#           		[[1, 30], [2, 30], [4, 30], [8, 30], [8, 25], [10, 25], [12, 25], [13, 25], [14, 27], [15, 30], [17, 30], [18, 40], [20, 40], [22, 40], [25, 50], [30, 50], [35, 55], [50, 60], [60, 60], [80, 80], [100, 100]], \
		#           		[[1, 30], [2, 30], [4, 30], [8, 30], [9, 25], [11, 35], [12, 35], [13, 35], [14, 40], [15, 40], [16, 40], [17, 45], [20, 45], [20, 55], [25, 55], [25, 70], [30, 70], [40, 90], [50, 100], [60, 120], [70, 140]], \
		#           		[[1, 30], [4, 30], [9, 30], [16, 32], [25, 50], [36, 72], [49, 98], [70, 150], [90, 200], [120, 210], [180, 270], [210, 330], [260, 400], [320, 500], [400, 600], [500, 700], [800, 900]]]
		search_config = [1, 2, 4, 8, 16, 25, 30, 35, 40, 45, 50, 55, 60, 65, 75, 90, 110, 130, 150, 170, 200, 220, 250, 310, 400, 500, 800, 1000, 1250, 1500, 1750, 1900, 2000] 
		          		
		f = open(sweep_result_path, "w")
		f.write("Program: " + args.program + " Topk: " + str(args.topk) + " Num_split: " + str(args.num_split)+ " Batch: "+str(args.batch)+"\n")
		f.write("L\tThreashold\tm\t|\tw\tr\tMetric\n")
	else:
		build_config = [(args.L, args.threshold, int(D/args.m), args.metric)]
		search_config = [[[args.w, args.reorder]]]
	# for bc, sc in zip(build_config, search_config):
	for bc in build_config:
		num_leaves, threshold, dims, metric = bc
		for arg in search_config:
			# leaves_to_search, reorder = arg[0], arg[1]
			leaves_to_search, reorder = arg, args.reorder
			if args.reorder!=-1:
				assert args.topk <= reorder
			else:
				if args.sweep:
					assert False, "Do you want reordering or not?"
			if args.sweep:
				f.write(str(num_leaves)+"\t"+str(threshold)+"\t"+str(int(D/dims))+"\t|\t"+str(leaves_to_search)+"\t"+str(reorder)+"\t"+str(metric)+"\n")
			print(str(num_leaves)+"\t"+str(threshold)+"\t"+str(dims)+"\t"+str(metric)+"\t"+str(leaves_to_search)+"\t"+str(reorder))
			neighbors=np.empty((queries.shape[0],0))
			distances=np.empty((queries.shape[0],0))
			total_latency = 0
			base_idx = 0
			for split in range(args.num_split):
				searcher_dir, searcher_path = get_searcher_path(split)  	
				print("Split ", split)
				# Load splitted dataset
				dataset = read_data(split_dataset_path + str(args.num_split) + "_" + str(split) if args.num_split>1 else dataset_basedir, base=False if args.num_split>1 else True, offset_=None if args.num_split>1 else 0, shape_=None)
				batch_size = min(args.batch, queries.shape[0])
				# Create ScaNN searcher
				print("Entering ScaNN builder")
				searcher = None
				searcher_path = searcher_path + '_' + str(num_leaves) + '_' + str(threshold) + '_' + str(dims) + '_' + metric + ("_reorder" if args.reorder!=-1 else '')
				if os.path.isdir(searcher_path):
					print("Loading searcher from ", searcher_path)
					searcher = scann.scann_ops_pybind.load_searcher(searcher_path)
				else:
					if reorder!=-1:
						searcher = scann.scann_ops_pybind.builder(dataset, 10, metric).tree(
							num_leaves=num_leaves, num_leaves_to_search=leaves_to_search, training_sample_size=args.coarse_training_size).score_ah(
							dims, anisotropic_quantization_threshold=threshold, training_sample_size=args.fine_training_size).reorder(reorder).build()			
					else:
						searcher = scann.scann_ops_pybind.builder(dataset, 10, metric).tree(
								num_leaves=num_leaves, num_leaves_to_search=leaves_to_search, training_sample_size=args.coarse_training_size).score_ah(
								dims, anisotropic_quantization_threshold=threshold, training_sample_size=args.fine_training_size).build()			
					print("Saving searcher to ", searcher_path)
					os.makedirs(searcher_path, exist_ok=True)
					searcher.serialize(searcher_path)

				if args.batch > 1:
					start = time.time()
					local_neighbors, local_distances = searcher.search_batched_parallel(queries, leaves_to_search=leaves_to_search, pre_reorder_num_neighbors=reorder, final_num_neighbors=args.topk, batch_size=batch_size)
					# local_neighbors, local_distances = searcher.search_batched(queries, leaves_to_search=leaves_to_search, pre_reorder_num_neighbors=reorder, final_num_neighbors=args.topk)
					end = time.time()
					total_latency = total_latency + 1000*(end - start)
					neighbors = np.append(neighbors, local_neighbors+base_idx, axis=1)
					distances = np.append(distances, local_distances, axis=1)
				else:
					# ScaNN search
					def single_query(query, base_idx):
						start = time.time()
						local_neighbors, local_distances = searcher.search(query, leaves_to_search=leaves_to_search, pre_reorder_num_neighbors=reorder, final_num_neighbors=args.topk)
						if local_neighbors.shape[0] < args.topk:
							plus_dim = args.topk-local_neighbors.shape[0]
							local_neighbors=np.concatenate((local_neighbors, np.full((plus_dim), N)), axis=-1)
							local_distances=np.concatenate((local_distances, np.full((plus_dim), math.inf if metric=="squared_l2" else -math.inf)), axis=-1)

						return (time.time() - start, (local_neighbors, local_distances))
					# ScaNN search
					print("Entering ScaNN searcher")
					local_results = [single_query(q, base_idx) for q in queries]
					total_latency += (np.sum(np.array([time for time, _ in local_results]).reshape(queries.shape[0], 1)))*1000
					nd = [nd for _, nd in local_results]
					neighbors = np.append(neighbors, np.vstack([n for n,d in nd])+base_idx, axis=1)
					distances = np.append(distances, np.vstack([d for n,d in nd]), axis=1)
				base_idx = base_idx + dataset.shape[0]

			final_neighbors = sort_neighbors(distances, neighbors)
			top1, top10, top100, top1000 = print_recall(final_neighbors, gt)
			print("Top ", args.topk, " Total latency (ms): ", total_latency)
			if args.sweep:
				f.write(str(top1)+" %\t"+str(top10)+" %\t"+str(top100)+" %\t"+str(top1000)+" %\t"+str(total_latency)+"\n")
	if args.sweep:
		f.close()
def run_faiss(D, index_key):
	gt, queries = prepare_eval()
	if args.sweep:
		M = 20 if "glove" in args.dataset else 64
		build_config = [(4096, M, 4, args.metric), (4096, M, 8, args.metric), (4096, M, 16, args.metric), (8192, M, 4, args.metric), (8192, M, 8, args.metric), (8192, M, 16, args.metric)]	# L, m, log2(k*), metric
		search_config = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]	# nprobe
		f = open(sweep_result_path, "w")
		f.write("Program: " + args.program + " Topk: " + str(args.topk) + " Num_split: " + str(args.num_split)+"\n")
		f.write("L\tm\tk_star\t|\tw\tMetric\n")
	else:
		build_config = [(args.L, args.m, math.log(args.k_star,2), args.metric)]
		search_config = [args.w]
	for bc in build_config:
		L, m, log2kstar, metric = bc
		assert D%m == 0 and (m==1 or m==2 or m==3 or m==4 or m==8 or m==12 or m==16 or m==20 or m==24 or m==28 or m==32 or m==40 or m==48 or m==56 or m==64 or m==96)	# Faiss only suports these
		index_key = "IVF"+str(L)+",PQ"+str(m)
		for sc in search_config:
			nprobe = sc
			if args.sweep:
				f.write(str(L)+"\t"+str(m)+"\t"+str(2**log2kstar)+"\t|\t"+str(nprobe)+"\t"+str(metric)+"\n")
			print(str(L)+"\t"+str(m)+"\t"+str(log2kstar)+"\t"+str(metric)+"\t"+str(nprobe)+"\n")
			neighbors=np.empty((queries.shape[0],0))
			distances=np.empty((queries.shape[0],0))
			base_idx = 0
			total_latency = 0
			for split in range(args.num_split):
				print("Split ", split)
				# Load splitted dataset
				xt = get_train(split, args.num_split)
				# Build Faiss index
				searcher_dir, searcher_path = get_searcher_path(split)
				preproc = train_faiss(args.dataset, split_dataset_path, D, xt, split, args.num_split, args.metric, index_key, log2kstar, searcher_dir)
				dataset = read_data(split_dataset_path + str(args.num_split) + "_" + str(split) if args.num_split>1 else dataset_basedir, base=False if args.num_split>1 else True, offset_=None if args.num_split>1 else 0, shape_=None)
				batch_size = min(args.batch, queries.shape[0])
				# Create Faiss index
				index = build_faiss(dataset, split, preproc)
				# Faiss search
				local_neighbors, local_distances, total_latency = search_faiss(queries, index, preproc, nprobe, args.topk, batch_size)
				neighbors = np.append(neighbors, local_neighbors+base_idx, axis=1)
				distances = np.append(distances, local_distances, axis=1)
				base_idx = base_idx + dataset.shape[0]
			final_neighbors = sort_neighbors(distances, neighbors)
			top1, top10, top100, top1000 = print_recall(final_neighbors, gt)
			print("Top ", args.topk, " Total latency (ms): ", total_latency)
			if args.sweep:
				f.write(str(top1)+" %\t"+str(top10)+" %\t"+str(top100)+" %\t"+str(top1000)+" %\t"+str(total_latency)+"\n")
	if args.sweep:
		f.close()
	# Below is for faiss's recall
	# gtc = gt[:, :1]
	# nq = queries.shape[0]
	# for rank in 1, 10, 100:
	# 	if rank > 100: continue
	# 	nok = (final_neighbors[:, :rank] == gtc).sum()
	# 	print("1-R@%d: %.4f" % (rank, nok / float(nq)), end=' ')
	# print()
	# print("Total latency (ms): ", total_latency)

def run_annoy(D):
	gt, queries = prepare_eval()
	assert args.metric!='angular', "[TODO] don't understand how angular works yet..."
	if args.sweep:
		build_config = [(args.metric, 100), (args.metric, 200), (args.metric, 400)]
		search_config = [100, 200, 400, 1000, 2000, 4000, 10000, 20000, 40000, 100000, 200000, 400000]
		f = open(sweep_result_path, "w")
		f.write("Program: " + args.program + " Topk: " + str(args.topk) + " Num_split: " + str(args.num_split)+"\n")
		f.write("Num trees\tNum search\tMetric\n")
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
		for sc in search_config:
			num_search = sc
			if args.sweep:
				f.write(str(n_trees)+"\t"+str(num_search)+"\t"+str(annoy_metric)+"\n")
			print(str(n_trees)+"\t"+str(num_search)+"\t"+str(annoy_metric))
			neighbors=np.empty((queries.shape[0],0))
			distances=np.empty((queries.shape[0],0))
			base_idx = 0
			total_latency = 0
			for split in range(args.num_split):
				searcher_dir, searcher_path = get_searcher_path(split)
				searcher_path = searcher_path + '_' + str(num_trees) + '_' + metric
				print("Split ", split)

				# Load splitted dataset
				dataset = read_data(split_dataset_path + str(args.num_split) + "_" + str(split) if args.num_split>1 else dataset_basedir, base=False if args.num_split>1 else True, offset_=None if args.num_split>1 else 0, shape_=None)
				# Create Annoy index
				searcher = annoy.AnnoyIndex(D, metric=annoy_metric)

				if os.path.isfile(searcher_path):
					print("Loading searcher from ", searcher_path)
					searcher.load(searcher_path)
				else:
					print("Annoy, adding items")
					for i, x in enumerate(dataset):
					    searcher.add_item(i, x.tolist())
					print("Annoy, building trees")
					searcher.build(n_trees)
					print("Saving searcher to ", searcher_path)
					os.makedirs(searcher_dir, exist_ok=True)
					searcher.save(searcher_path)

				print("Entering Annoy searcher")
				# Annoy batch version
				if args.batch > 1:
					pool = ThreadPool()
					start = time.time()
					result = pool.map(lambda q: searcher.get_nns_by_vector(q.tolist(), args.topk, num_search, include_distances=True), queries)
					end = time.time()
					result = np.array(result)
					local_neighbors = result[:,0,:]
					local_distances = result[:,1,:]
					total_latency = total_latency + (end - start)*1000
					neighbors = np.append(neighbors, local_neighbors+base_idx, axis=1)
					distances = np.append(distances, local_distances, axis=1)
				else:
					def single_query(query, base_idx):
						start = time.time()
						result = searcher.get_nns_by_vector(query.tolist(), args.topk, num_search, include_distances=True)
						return (time.time() - start, result)
					local_results = [single_query(q, base_idx) for q in queries]
					total_latency += (np.sum(np.array([time for time, _ in local_results]).reshape(queries.shape[0], 1)))*1000
					nd = [nd for _, nd in local_results]
					neighbors = np.append(neighbors, np.array([n for n,d in nd])+base_idx, axis=1)
					distances = np.append(distances, np.array([d for n,d in nd]), axis=1)
				base_idx = base_idx + dataset.shape[0]

			final_neighbors = sort_neighbors(distances, neighbors)
			top1, top10, top100, top1000 = print_recall(final_neighbors, gt)
			print("Top ", args.topk, " Total latency (ms): ", total_latency)
			if args.sweep:
				f.write(str(top1)+" %\t"+str(top10)+" %\t"+str(top100)+" %\t"+str(top1000)+" %\t"+str(total_latency)+"\n")
	if args.sweep:
		f.close()

# only for faiss
def get_train(split=-1, total=-1):
	if "sift1m" in args.dataset:
		filename = dataset_basedir + 'sift_learn.fvecs' if split<0 else dataset_basedir + 'split_data/sift1m_learn%d_%d' % (total, split)
		return mmap_fvecs(filename)
	elif "sift1b" in args.dataset:
		return bvecs_read(dataset_basedir+'bigann_learn.bvecs')
	elif "glove" in args.dataset:
		return h5py.File(dataset_basedir+"glove-100-angular.hdf5", "r")['test']
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

	# if "sift1m" in args.dataset:
	# 	filename = dataset_basedir + 'sift_groundtruth.ivecs' if args.metric=="squared_l2" else groundtruth_path
	# 	print("Reading from ", filename)
	# 	return ivecs_read(filename)
	# elif "sift1b" in args.dataset:
	# 	filename = dataset_basedir +  'gnd/idx_1000M.ivecs' if args.metric=="squared_l2" else groundtruth_path
	# 	print("Reading from ", filename)
	# 	return ivecs_read(filename)
	# elif "glove" in args.dataset:
	# 	if args.metric == "dot_product":
	# 		print("Reading from ", dataset_basedir+"glove-100-angular.hdf5")
	# 		return h5py.File(dataset_basedir+"glove-100-angular.hdf5", "r")['neighbors']
	# 	else:
	# 		print("Reading from ", groundtruth_path)
	# 		return read_data(groundtruth_path, base=False)
	# else:
	# 	assert False

def get_queries():
	if "sift1m" in args.dataset:
		return mmap_fvecs(dataset_basedir + 'sift_query.fvecs')
	elif "sift1b" in args.dataset:
		return bvecs_read(dataset_basedir+'bigann_query.bvecs')
	elif "glove" in args.dataset:
		return h5py.File(dataset_basedir+"glove-100-angular.hdf5", "r")['test']
	else:
		assert False


if os.path.isdir("/arc-share"):
	basedir = "/arc-share/MICRO21_ANNA/"
else:
	basedir = "./"

os.makedirs("./result", exist_ok=True)
split_dataset_path = None
sweep_result_path = "./result/"+args.program+"_"+args.dataset+"_topk_"+str(args.topk)+"_num_split_"+str(args.num_split)+"_batch_"+str(args.batch)+"_sweep_result.txt"
index_key = None
N = -1
D = -1
num_iter = -1
qN = -1

if "sift1m" in args.dataset:
	dataset_basedir = basedir + "SIFT1M/"
	if args.split != True:
		split_dataset_path =dataset_basedir+"split_data/sift1m_"
		groundtruth_path = dataset_basedir + "sift1m_"+args.metric+"_gt"
	N=1000000
	D=128
	num_iter = 1
	qN = 10000
	index_key = "IVF4096,PQ64"
elif "sift1b" in args.dataset:
	dataset_basedir = basedir + "SIFT1B/"
	if args.split != True:
		split_dataset_path = dataset_basedir+"split_data/sift1b_"
		groundtruth_path = dataset_basedir + "sift1b_"+args.metric+"_gt"
	N=1000000000
	D=128
	num_iter = 4
	qN = 10000
	index_key = "OPQ8_32,IVF262144,PQ8"
elif "glove" in args.dataset:
	dataset_basedir = basedir + "GLOVE/"
	if args.split != True:
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
		run_faiss( D, index_key)
	elif args.program == "annoy":
		run_annoy(D)
	else:
		assert False

if args.groundtruth:
	run_groundtruth()
