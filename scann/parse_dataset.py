'''
	Usage: python3 parse_dataset.py [dataset_path] [# of split]
'''
import sys
import numpy as np
import h5py
filename = sys.argv[1]
num_split = int(sys.argv[2])

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def ivecs_mmap(fname):
    a = np.memmap(fname, dtype='int32', mode='r')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]


def fvecs_mmap(fname):
    return ivecs_mmap(fname).view('float32')


def bvecs_mmap(fname, offset_, shape_):
    x = np.memmap(fname, dtype='uint8', mode='r', offset=offset_*132, shape=(shape_*132))
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def ivecs_write(fname, m, type='float'):
	n, d = m.shape
	# dimension_arr = np.empty((n, 1), dtype=type)
	# dimension_arr[:] = d
	# m = np.append(dimension_arr, m, axis=1)
	# m.tofile(fname)
	m1 = np.empty((n, d + 1), dtype=type)
	m1[:, 0] = d
	m1[:, 1:] = m
	m1.tofile(fname)
	print(m.shape, " ", fname)


def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))

def sanitize(x):
    return np.ascontiguousarray(x, dtype='float32')

class DatasetBigANN():
    """
    The original dataset is available at: http://corpus-texmex.irisa.fr/
    (ANN_SIFT1B)
    """

    def __init__(self, split=False, num_split=-1, nb_M=1000):
        # Dataset.__init__(self)
        # assert nb_M in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
        self.nb_M = nb_M
        if split:
        	nb = 10**9/num_split
        else:
        	nb = 10**9
        self.d, self.nt, self.nb, self.nq = 128, 10**8, nb, 10000
        self.basedir = dataset_basedir

    def get_queries(self):
        return sanitize(bvecs_mmap(self.basedir + 'bigann_query.bvecs')[:])

    def get_train(self, maxtrain=None):
        maxtrain = maxtrain if maxtrain is not None else self.nt
        return sanitize(bvecs_mmap(self.basedir + 'bigann_learn.bvecs')[:maxtrain])

    def get_groundtruth(self, k=None):
        gt = ivecs_read(self.basedir + 'gnd/idx_%dM.ivecs' % self.nb_M)
        if k is not None:
            assert k <= 100
            gt = gt[:, :k]
        return gt

    def get_database(self, index):		# [YJ] used for splitted dataset
        # assert self.nb_M < 100, "dataset too large, use iterator"
        return sanitize(bvecs_mmap(self.basedir + './split_data/sift1b_'+str(index)))

    def database_iterator(self, bs=128, split=(1, 0)):
        xb = bvecs_mmap(self.basedir + 'bigann_base.bvecs')
        nsplit, rank = split
        i0, i1 = self.nb * rank // nsplit, self.nb * (rank + 1) // nsplit
        for j0 in range(i0, i1, bs):
            yield sanitize(xb[j0: min(j0 + bs, i1)])

def bvecs_read(fname):
	b = np.fromfile(fname, dtype=np.uint8)
	d = b[:4].view('int32')[0]
	return b.reshape(-1, d+4)[:, 4:].copy()

# def read_data()

def split(data_path, split_dataset_path, num_iter, N, D, read_file)
	
	split = 0
	for it in range(num_iter):
		print("Iter: ", it)
		# if it==num_iter-1:
		# 	dataset = np.append(dataset, bvecs_read(dataset_basedir+"bigann_base.bvecs")[it*dataset_per_iter:], axis=0)
		# else:
		# 	dataset = np.append(dataset, bvecs_read(dataset_basedir+"bigann_base.bvecs")[it*dataset_per_iter:(it+1)*dataset_per_iter], axis=0)
		if it==num_iter-1:
			dataset = np.append(dataset, bvecs_mmap(dataset_basedir+"bigann_base.bvecs", offset_=it*dataset_per_iter, shape_=(N-it*dataset_per_iter)), axis=0)
		else:
			dataset = np.append(dataset, bvecs_mmap(dataset_basedir+"bigann_base.bvecs", offset_=it*dataset_per_iter, shape_=dataset_per_iter), axis=0)
		print("Reading dataset done")
		count=0
		while True:
			if (split+1)*num_per_split > dataset_per_iter*(it+1):
				if it!=num_iter-1:
					print("Entering next iter..")
					dataset = dataset[count*num_per_split:]
				else:
					ivecs_write(split_dataset_path + str(num_split) + "_" + str(split), dataset[count*num_per_split:], 'uint8')
					split = split+1
				break
			else:
				ivecs_write(split_dataset_path + str(num_split) + "_" + str(split), dataset[count*num_per_split:(count+1)*num_per_split], 'uint8')
				split = split+1
			count = count+1

if "sift" in filename:
	dataset_basedir="/arc-share/MICRO21_ANNA/SIFT1B/"
	split_dataset_path=dataset_basedir+"split_data/sift1b_"

	N=1000000000
	assert(N%num_split == 0)
	num_per_split = int(N/num_split)
	ds = DatasetBigANN()
	# q = ds.get_queries()
	# print("query: ", q)
	# split = 0
	# counter = 0
	# read_batch = 1048576
	# print("Dataset size: ", N, " / num_split: ", num_split, " / num_per_split: ", num_per_split)
	# split_dataset = np.empty((0,ds.d))
	# written_data = 0
	# for test in ds.database_iterator(bs=read_batch):
	# 	split_dataset = np.append(split_dataset, test, axis=0)
	# 	counter = counter+read_batch
	# 	if counter % read_batch == 0:
	# 		print("[Split "+str(split)+"/"+str(num_split)+"] Counter: ", counter, " / num_per_split: ", num_per_split, "\r")
	# 	if counter >= num_per_split:
	# 		counter = 0
	# 		split = split+1
	# 		ivecs_write(dataset_basedir + "split_data/sift1b_" + str(num_split) + "_" + str(split), split_dataset[:num_per_split])
	# 		print(dataset_basedir + "split_data/sift1b_" + str(num_split) + "_" + str(split), " / total: ", num_split)
	# 		# print(split_dataset[0])
	# 		written_data += split_dataset[:num_per_split].shape[0]
	# 		split_dataset = split_dataset[num_per_split:]

	# a=bvecs_read(dataset_basedir+"bigann_base.bvecs")
	# print("a.shape:" , a.shape)
	# exit(1)

	dataset = np.empty((0,128), dtype=np.uint8)
	num_iter = 4
	dataset_per_iter = int(N/num_iter)
	num_per_split = int(N/num_split)
	print("dataset_per_iter: ", dataset_per_iter, " / num_per_split: ", num_per_split)
	split = 0
	for it in range(num_iter):
		print("Iter: ", it)
		# if it==num_iter-1:
		# 	dataset = np.append(dataset, bvecs_read(dataset_basedir+"bigann_base.bvecs")[it*dataset_per_iter:], axis=0)
		# else:
		# 	dataset = np.append(dataset, bvecs_read(dataset_basedir+"bigann_base.bvecs")[it*dataset_per_iter:(it+1)*dataset_per_iter], axis=0)
		if it==num_iter-1:
			dataset = np.append(dataset, bvecs_mmap(dataset_basedir+"bigann_base.bvecs", offset_=it*dataset_per_iter, shape_=(N-it*dataset_per_iter)), axis=0)
		else:
			dataset = np.append(dataset, bvecs_mmap(dataset_basedir+"bigann_base.bvecs", offset_=it*dataset_per_iter, shape_=dataset_per_iter), axis=0)
		print("Reading dataset done")
		count=0
		while True:
			if (split+1)*num_per_split > dataset_per_iter*(it+1):
				if it!=num_iter-1:
					print("Entering next iter..")
					dataset = dataset[count*num_per_split:]
				else:
					ivecs_write(split_dataset_path + str(num_split) + "_" + str(split), dataset[count*num_per_split:], 'uint8')
					split = split+1
				break
			else:
				ivecs_write(split_dataset_path + str(num_split) + "_" + str(split), dataset[count*num_per_split:(count+1)*num_per_split], 'uint8')
				split = split+1
			count = count+1


if "glove" in filename:
	dataset_basedir="/arc-share/MICRO21_ANNA/GLOVE/"
	split_dataset_path=dataset_basedir+"split_data/glove_"
	N=1183514
	dataset = np.empty((0,100))
	num_iter = 1000
	dataset_per_iter = int(N/num_iter)
	num_per_split = int(N/num_split)
	print("dataset_per_iter: ", dataset_per_iter, " / num_per_split: ", num_per_split)
	split = 0
	for it in range(num_iter):
		print("Iter: ", it)
		if it==num_iter-1:
			dataset = np.append(dataset, h5py.File(dataset_basedir+"glove-100-angular.hdf5", "r")['train'][it*dataset_per_iter:], axis=0)
		else:
			dataset = np.append(dataset, h5py.File(dataset_basedir+"glove-100-angular.hdf5", "r")['train'][it*dataset_per_iter:(it+1)*dataset_per_iter], axis=0)
		print("Reading dataset done")
		count=0
		while True:
			if (split+1)*num_per_split > dataset_per_iter*(it+1):
				if it!=num_iter-1:
					# print("Entering next iter..")
					dataset = dataset[count*num_per_split:]
				else:
					ivecs_write(split_dataset_path + str(num_split) + "_" + str(split), dataset[count*num_per_split:])
					split = split+1
				break
			else:
				ivecs_write(split_dataset_path + str(num_split) + "_" + str(split), dataset[count*num_per_split:(count+1)*num_per_split])
				split = split+1
			count = count+1


# for i in range(num_split):
# 	dataset = ds.get_database(i)
# 	print(dataset.shape)
# 	searcher = scann.scann_ops_pybind.builder(dataset, 10, "dot_product").tree(
# 	    num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
# 	    2, anisotropic_quantization_threshold=0.2).reorder(100).build()

