'''
	Usage: python3 split_and_run.py --dataset [dataset name] --num_split [# of split] --metric [distance measure] --num_leaves [num_leaves] --num_search [num_leaves_to_search] --training_size [traing sample size] --threshold [threshold] --reorder [reorder size] [--split] [--eval_split]
'''
import sys
import numpy as np
import h5py
import scann
import time
import argparse
import os

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--dataset', type=str, help='sift1b, glove ...')
parser.add_argument('--num_split', type=int, default=-1, required=True, help='# of splits')
parser.add_argument('--split', action='store_true')
parser.add_argument('--eval_split', action='store_true')
parser.add_argument('--metric', type=str, help='dot_product, squared_l2, angular')
parser.add_argument('--num_leaves', type=int, default=-1, required=True, help='# of leaves')
parser.add_argument('--num_search', type=int, default=-1, required=True, help='# of searching leaves')
parser.add_argument('--training_size', type=int, default=-1, required=True, help='training sample size')
parser.add_argument('--threshold', type=float, default=0.2, required=True, help='anisotropic_quantization_threshold')
parser.add_argument('--reorder', type=int, default=-1, required=True, help='reorder size')
args = parser.parse_args()


def compute_recall(neighbors, true_neighbors):
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total / true_neighbors.size	

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

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

def read_data(dataset_path, offset_=None, shape_=None, base=True):
	if "sift1b" in args.dataset:
		file = dataset_path+"bigann_base.bvecs" if base else dataset_path
		print("Reading from ", file)
		return bvecs_mmap(file, offset_=offset_, shape_=shape_)
	elif "glove" in args.dataset:
		file = dataset_path+"glove-100-angular.hdf5" if base else dataset_path
		print("Reading from ", file)
		dataset = h5py.File(file, "r")
		if base:
			dataset = dataset['train']
			normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
			return normalized_dataset[offset_:offset_+shape_]
		else:
			dataset = h5py.File(dataset_path, "r")
			return dataset['dataset']
	else:
		assert(false)

def write_data(split_data_path, split_data):
	if "sift1b" in args.dataset:
		bvecs_write(split_data_path, split_data)
	elif "glove" in args.dataset:
		hf = h5py.File(split_data_path, 'w')
		hf.create_dataset('dataset', data=split_data)
	print("Wrote to ", split_data_path, ", shape ", split_data.shape)

def split(filename, data_path, split_dataset_path, num_iter, N, D):
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
		while True:
			if (split+1)*num_per_split > dataset_per_iter*(it+1):
				if it!=num_iter-1:
					print("Entering next iter..")
					dataset = dataset[count*num_per_split:]
				else:
					write_data(split_dataset_path + str(args.num_split) + "_" + str(split), dataset[count*num_per_split:])
					num_split_list.append(dataset[count*num_per_split:].shape[0])
					split = split+1
				break
			elif split < args.num_split:
				write_data(split_dataset_path + str(args.num_split) + "_" + str(split), dataset[count*num_per_split:(count+1)*num_per_split])
				num_split_list.append(dataset[count*num_per_split:(count+1)*num_per_split].shape[0])
				split = split+1
			count = count+1
	print("num_split_lists: ", num_split_list)

def run(dataset_basedir, split_dataset_path):
	gt = get_groundtruth(dataset_basedir)
	queries = get_queries(dataset_basedir)
	neighbors=np.empty((queries.shape[0],0))
	distances=np.empty((queries.shape[0],0))
	base_idx = 0
	total_latency = 0
	for split in range(args.num_split):
		searcher_path = './scann_searcher/'+args.dataset+'/Split_'+str(args.num_split)+'/'+args.dataset+'_searcher_'+str(args.num_split)+'_'+str(split)
		print("Split ", split)
		# Load splitted dataset
		dataset = read_data(split_dataset_path + str(args.num_split) + "_" + str(split), base=False)
		# Create ScaNN searcher
		print("Entering ScaNN builder")
		searcher = None
		if os.path.isdir(searcher_path):
			print("Loading searcher from ", searcher_path)
			searcher = scann.scann_ops_pybind.load_searcher(searcher_path)
		else:
			searcher = scann.scann_ops_pybind.builder(dataset, 10, args.metric).tree(
			    num_leaves=args.num_leaves, num_leaves_to_search=args.num_search, training_sample_size=args.training_size).score_ah(
			    2, anisotropic_quantization_threshold=args.threshold).reorder(args.reorder).build()
			print("Saving searcher to ", searcher_path)
			os.makedirs(searcher_path, exist_ok=True)
			searcher.serialize(searcher_path)

		# ScaNN search
		print("Entering ScaNN searcher")
		start = time.time()
		local_neighbors, local_distances = searcher.search_batched(queries, final_num_neighbors=100)
		end = time.time()
		total_latency = total_latency + 1000*(end - start)
		neighbors = np.append(neighbors, local_neighbors+base_idx, axis=1)
		distances = np.append(distances, local_distances, axis=1)
		base_idx = base_idx + dataset.shape[0]
	final_neighbors = np.take_along_axis(neighbors, np.argsort(-distances, axis=-1), -1)
	print("Recall@1:", compute_recall(final_neighbors[:,:1], gt[:, :1]))
	print("Recall@10:", compute_recall(final_neighbors[:,:10], gt[:, :10]))
	print("Recall@100:", compute_recall(final_neighbors[:,:100], gt[:, :100]))
	print("Total latency (ms): ", total_latency)

def get_groundtruth(dataset_basedir):
	if "sift1b" in args.dataset:
		return ivecs_read(dataset_basedir + 'gnd/idx_1000M.ivecs')
	elif "glove" in args.dataset:
		return h5py.File(dataset_basedir+"glove-100-angular.hdf5", "r")['neighbors']

def get_queries(dataset_basedir):
	if "sift1b" in args.dataset:
		return bvecs_read(dataset_basedir+'bigann_query.bvecs')
	elif "glove" in args.dataset:
		return h5py.File(dataset_basedir+"glove-100-angular.hdf5", "r")['test']

dataset_basedir = None
split_dataset_path = None
N = -1
D = -1
num_iter = -1
if "sift1b" in args.dataset:
	dataset_basedir="/arc-share/MICRO21_ANNA/SIFT1B/"
	split_dataset_path=dataset_basedir+"split_data/sift1b_"
	N=1000000000
	D=128
	num_iter = 4
elif "glove" in args.dataset:
	dataset_basedir="/arc-share/MICRO21_ANNA/GLOVE/"
	split_dataset_path=dataset_basedir+"split_data/glove_"
	N=1183514
	D=100
	num_iter = 10

if args.split:
	split(args.dataset, dataset_basedir, split_dataset_path, num_iter, N, D)
if args.eval_split:
	run(dataset_basedir, split_dataset_path)
