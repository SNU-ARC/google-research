from __future__ import print_function
import numpy as np
import time
import os
import sys
import faiss
import re

from multiprocessing.dummy import Pool as ThreadPool

########### Set Faiss arguments here #############
ngpu = faiss.get_num_gpus()

replicas = 1  # nb of replicas of sharded dataset
add_batch_size = 32768
query_batch_size = 16384
nprobe = 32
use_precomputed_tables = False
tempmem = 1536*1024*1024
max_add = -1
use_float16 = True
use_cache = True
nnn = 100
altadd = False
I_fname = None
D_fname = None
dim = -1
##################################################

class IdentPreproc:
	"""a pre-processor is either a faiss.VectorTransform or an IndentPreproc"""

	def __init__(self, d):
		self.d_in = self.d_out = d

	def apply_py(self, x):
		return x

def make_vres_vdev(i0=0, i1=-1):
	" return vectors of device ids and resources useful for gpu_multiple"
	vres = faiss.GpuResourcesVector()
	vdev = faiss.IntVector()
	if i1 == -1:
		i1 = ngpu
	for i in range(i0, i1):
		vdev.push_back(i)
		vres.push_back(gpu_resources[i])
	return vres, vdev

def sanitize(x):
	""" convert array to a c-contiguous float array """
	return np.ascontiguousarray(x.astype('float32'))

def rate_limited_imap(f, l):
	"""A threaded imap that does not produce elements faster than they
	are consumed"""
	pool = ThreadPool(1)
	res = None
	for i in l:
		res_next = pool.apply_async(f, (i, ))
		if res:
			yield res.get()
		res = res_next
	yield res.get()


def prepare_trained_index(preproc, xt):
	fmetric = None
	if "dot_product" == metric or "angular" == metric:
		fmetric = faiss.METRIC_INNER_PRODUCT
	elif "squared_l2" == metric:
		fmetric = faiss.METRIC_L2
	else:
		assert False

	coarse_quantizer = prepare_coarse_quantizer(preproc, xt)
	d = preproc.d_out
	if pqflat_str == 'Flat':
		print("making an IVFFlat index")
		idx_model = faiss.IndexIVFFlat(coarse_quantizer, d, ncent,
									   fmetric)
	else:
		m = int(pqflat_str[2:])
		assert m < 56 or use_float16, "PQ%d will work only with -float16" % m
		print("making an IVFPQ index, m = ", m)
		idx_model = faiss.IndexIVFPQ(coarse_quantizer, d, ncent, m, 8, fmetric)

	coarse_quantizer.this.disown()
	idx_model.own_fields = True

	# finish training on CPU
	t0 = time.time()
	print("Training vector codes")
	x = preproc.apply_py(sanitize(xt[:1000000]))
	idx_model.train(x)
	print("  done %.3f s" % (time.time() - t0))

	return idx_model
	
def compute_populated_index(preproc, xb):
	"""Add elements to a sharded index. Return the index and if available
	a sharded gpu_index that contains the same data. """
	indexall = prepare_trained_index(preproc, xb)

	co = faiss.GpuMultipleClonerOptions()
	co.useFloat16 = use_float16
	co.useFloat16CoarseQuantizer = False
	co.usePrecomputed = use_precomputed_tables
	co.indicesOptions = faiss.INDICES_CPU
	co.verbose = True
	co.reserveVecs = max_add if max_add > 0 else xb.shape[0]
	co.shard = True
	assert co.shard_type in (0, 1, 2)
	vres, vdev = make_vres_vdev()
	gpu_index = faiss.index_cpu_to_gpu_multiple(
		vres, vdev, indexall, co)

	print("add...")
	t0 = time.time()
	nb = xb.shape[0]
	for i0, xs in dataset_iterator(xb, preproc, add_batch_size):
		i1 = i0 + xs.shape[0]
		gpu_index.add_with_ids(xs, np.arange(i0, i1))
		if max_add > 0 and gpu_index.ntotal > max_add:
			print("Flush indexes to CPU")
			for i in range(ngpu):
				index_src_gpu = faiss.downcast_index(gpu_index.at(i))
				index_src = faiss.index_gpu_to_cpu(index_src_gpu)
				print("  index %d size %d" % (i, index_src.ntotal))
				index_src.copy_subset_to(indexall, 0, 0, nb)
				index_src_gpu.reset()
				index_src_gpu.reserveMemory(max_add)
			gpu_index.sync_with_shard_indexes()

		print('\r%d/%d (%.3f s)  ' % (
			i0, nb, time.time() - t0), end=' ')
		sys.stdout.flush()
	print("Add time: %.3f s" % (time.time() - t0))

	print("Aggregate indexes to CPU")
	t0 = time.time()

	if hasattr(gpu_index, 'at'):
		# it is a sharded index
		for i in range(ngpu):
			index_src = faiss.index_gpu_to_cpu(gpu_index.at(i))
			print("  index %d size %d" % (i, index_src.ntotal))
			index_src.copy_subset_to(indexall, 0, 0, nb)
	else:
		# simple index
		index_src = faiss.index_gpu_to_cpu(gpu_index)
		index_src.copy_subset_to(indexall, 0, 0, nb)

	print("  done in %.3f s" % (time.time() - t0))

	if max_add > 0:
		# it does not contain all the vectors
		gpu_index = None

	return gpu_index, indexall

def compute_populated_index_2(preproc):

    indexall = prepare_trained_index(preproc)

    # set up a 3-stage pipeline that does:
    # - stage 1: load + preproc
    # - stage 2: assign on GPU
    # - stage 3: add to index

    stage1 = dataset_iterator(xb, preproc, add_batch_size)

    vres, vdev = make_vres_vdev()
    coarse_quantizer_gpu = faiss.index_cpu_to_gpu_multiple(
        vres, vdev, indexall.quantizer)

    def quantize(args):
        (i0, xs) = args
        _, assign = coarse_quantizer_gpu.search(xs, 1)
        return i0, xs, assign.ravel()

    stage2 = rate_limited_imap(quantize, stage1)

    print("add...")
    t0 = time.time()
    nb = xb.shape[0]

    for i0, xs, assign in stage2:
        i1 = i0 + xs.shape[0]
        if indexall.__class__ == faiss.IndexIVFPQ:
            indexall.add_core_o(i1 - i0, faiss.swig_ptr(xs),
                                None, None, faiss.swig_ptr(assign))
        elif indexall.__class__ == faiss.IndexIVFFlat:
            indexall.add_core(i1 - i0, faiss.swig_ptr(xs), None,
                              faiss.swig_ptr(assign))
        else:
            assert False

        print('\r%d/%d (%.3f s)  ' % (
            i0, nb, time.time() - t0), end=' ')
        sys.stdout.flush()
    print("Add time: %.3f s" % (time.time() - t0))

    return None, indexall

def get_populated_index(preproc, xb, split):

	if not index_cachefile or not os.path.exists(index_cachefile):
		if not altadd:
			gpu_index, indexall = compute_populated_index(preproc, xb)
		else:
			gpu_index, indexall = compute_populated_index_2(preproc)
		if index_cachefile:
			print("store", index_cachefile)
			faiss.write_index(indexall, index_cachefile)
	else:
		print("load", index_cachefile)
		indexall = faiss.read_index(index_cachefile)
		gpu_index = None

	co = faiss.GpuMultipleClonerOptions()
	co.useFloat16 = use_float16
	co.useFloat16CoarseQuantizer = False
	co.usePrecomputed = use_precomputed_tables
	co.indicesOptions = 0
	co.verbose = True
	co.shard = True    # the replicas will be made "manually"
	t0 = time.time()
	print("CPU index contains %d vectors, move to GPU" % indexall.ntotal)
	if replicas == 1:

		if not gpu_index:
			print("copying loaded index to GPUs")
			vres, vdev = make_vres_vdev()
			index = faiss.index_cpu_to_gpu_multiple(
				vres, vdev, indexall, co)
		else:
			index = gpu_index

	else:
		del gpu_index # We override the GPU index

		print("Copy CPU index to %d sharded GPU indexes" % replicas)

		index = faiss.IndexReplicas()

		for i in range(replicas):
			gpu0 = ngpu * i / replicas
			gpu1 = ngpu * (i + 1) / replicas
			vres, vdev = make_vres_vdev(gpu0, gpu1)

			print("   dispatch to GPUs %d:%d" % (gpu0, gpu1))

			index1 = faiss.index_cpu_to_gpu_multiple(
				vres, vdev, indexall, co)
			index1.this.disown()
			index.addIndex(index1)
		index.own_fields = True
	del indexall
	print("move to GPU done in %.3f s" % (time.time() - t0))
	return index

def train_preprocessor(xt):
	print("train preproc", preproc_str)
	d = xt.shape[1]
	t0 = time.time()
	if preproc_str.startswith('OPQ'):
		fi = preproc_str[3:-1].split('_')
		m = int(fi[0])
		dout = int(fi[1]) if len(fi) == 2 else d
		preproc = faiss.OPQMatrix(d, m, dout)
	elif preproc_str.startswith('PCAR'):
		dout = int(preproc_str[4:-1])
		preproc = faiss.PCAMatrix(d, dout, 0, True)
	else:
		assert False
	preproc.train(sanitize(xt[:1000000]))
	print("preproc train done in %.3f s" % (time.time() - t0))
	return preproc

def get_preprocessor(xt):
	if preproc_str:
		if not preproc_cachefile or not os.path.exists(preproc_cachefile):
			preproc = train_preprocessor(xt)
			if preproc_cachefile:
				print("store", preproc_cachefile)
				faiss.write_VectorTransform(preproc, preproc_cachefile)
		else:
			print("load", preproc_cachefile)
			preproc = faiss.read_VectorTransform(preproc_cachefile)
	else:
		preproc = IdentPreproc(dim)
	return preproc

def get_centroids(index):
	pq = index.pq
	# read the PQ centroids
	cen = faiss.vector_to_array(pq.centroids)
	cen = cen.reshape(pq.M, pq.ksub, pq.dsub)
	
	return cen

def train_coarse_quantizer(x, k, preproc):
	d = preproc.d_out
	clus = faiss.Clustering(d, k)
	clus.verbose = True
	# clus.niter = 2
	clus.max_points_per_centroid = 10000000

	print("apply preproc on shape", x.shape, 'k=', k)
	t0 = time.time()
	x = preproc.apply_py(sanitize(x))
	print("   preproc %.3f s output shape %s" % (
		time.time() - t0, x.shape))

	vres, vdev = make_vres_vdev()
	index = faiss.index_cpu_to_gpu_multiple(
		vres, vdev, faiss.IndexFlatL2(d))
	clus.train(x, index)
	centroids = faiss.vector_float_to_array(clus.centroids)

	return centroids.reshape(k, d)

def prepare_coarse_quantizer(preproc, xt):

	if cent_cachefile and os.path.exists(cent_cachefile):
		print("load centroids", cent_cachefile)
		centroids = np.load(cent_cachefile)
	else:
		nt = max(1000000, 256 * ncent)
		print("train coarse quantizer...")
		t0 = time.time()
		centroids = train_coarse_quantizer(xt[:nt], ncent, preproc)
		print("Coarse train time: %.3f s" % (time.time() - t0))
		if cent_cachefile:
			print("store centroids", cent_cachefile)
			np.save(cent_cachefile, centroids)

	coarse_quantizer = faiss.IndexFlatL2(preproc.d_out)
	coarse_quantizer.add(centroids)

	return coarse_quantizer

def prepare_pq_quantizer(preproc, xt, coarse_quantizer):
	idx_model = None
	centroids = None
	d = preproc.d_out
	if pqflat_str == 'Flat':
		print("making an IVFFlat index")
		idx_model = faiss.IndexIVFFlat(coarse_quantizer, d, ncent,
									   faiss.METRIC_L2)
	else:
		m = int(pqflat_str[2:])
		assert m < 56 or use_float16, "PQ%d will work only with -float16" % m
		print("making an IVFPQ index, m = ", m)
		idx_model = faiss.IndexIVFPQ(coarse_quantizer, d, ncent, m, 8)

	if pq_cachefile and os.path.exists(pq_cachefile):
		print("load sub-quantizers", pq_cachefile)
		centroids = np.load(pq_cachefile)
	else:

		# finish training on CPU
		t0 = time.time()
		print("Training vector codes")
		x = preproc.apply_py(sanitize(xt[:1000000]))
		idx_model.train(x)
		print("  done %.3f s" % (time.time() - t0))

		centroids = get_centroids(idx_model)
		if pq_cachefile:
			print("store sub-quantizers", pq_cachefile)
			np.save(pq_cachefile, centroids)
	
	faiss.copy_array_to_vector(centroids.ravel(), idx_model.pq.centroids)

	return idx_model

def dataset_iterator(x, preproc, bs):
	""" iterate over the lines of x in blocks of size bs"""

	nb = x.shape[0]
	block_ranges = [(i0, min(nb, i0 + bs))
					for i0 in range(0, nb, bs)]

	def prepare_block(i01):
		i0, i1 = i01
		xb = sanitize(x[i0:i1])
		return i0, preproc.apply_py(xb)

	return rate_limited_imap(prepare_block, block_ranges)

def search_faiss(xq, index, preproc):
	print("--------------- search_faiss ----------------")
	ps = faiss.GpuParameterSpace()
	ps.initialize(index)

	print("search...")
	sl = query_batch_size
	nq = xq.shape[0]
	ps.set_index_parameter(index, 'nprobe', nprobe)
	print(index.metric_type)

	if sl == 0:
		D, I = index.search(preproc.apply_py(sanitize(xq)), nnn)
	else:
		I = np.empty((nq, nnn), dtype='int32')
		D = np.empty((nq, nnn), dtype='float32')

		inter_res = ''

		for i0, xs in dataset_iterator(xq, preproc, sl):
			i1 = i0 + xs.shape[0]
			Di, Ii = index.search(xs, nnn)

			I[i0:i1] = Ii
			D[i0:i1] = Di

	print("--------------------------------------------\n")
	return I, D

def train_faiss(db, split_dataset_path, D, xt, split, num_split, met, index_key):
	print("--------------- train_faiss ----------------")
	global cacheroot
	global preproc_cachefile
	global cent_cachefile
	global pq_cachefile
	global index_cachefile
	global preproc_str
	global ivf_str
	global pqflat_str
	global ncent
	global dim
	global dbname
	global metric
	
	dim = D
	dbname = db
	metric = met
	cacheroot = split_dataset_path + 'tmp'
	# cacheroot
	if not os.path.isdir(cacheroot):
		print("%s does not exist, creating it" % cacheroot)
		os.mkdir(cacheroot)

	# index pattern
	pat = re.compile('(OPQ[0-9]+(_[0-9]+)?,|PCAR[0-9]+,)?' +
				 '(IVF[0-9]+),' +
				 '(PQ[0-9]+|Flat)')
	matchobject = pat.match(index_key)
	assert matchobject, 'could not parse ' + index_key
	mog = matchobject.groups()
	preproc_str = mog[0]
	ivf_str = mog[2]
	pqflat_str = mog[3]
	ncent = int(ivf_str[3:])
	prefix = ''

	# check cache files
	if preproc_str:
		preproc_cachefile = '%s/%spreproc_%s_%s.vectrans' % (
			cacheroot, prefix, dbname, preproc_str[:-1])
	else:
		preproc_cachefile = None
		preproc_str = ''

	cent_cachefile = '%s/%s_%scent_%s_%s_%s_%s%s.npy' % (
		cacheroot, metric, prefix, dbname, split, num_split, preproc_str, ivf_str)

	pq_cachefile = '%s/%s_%spq_%s_%s_%s_%s%s,%s.npy' % (
		cacheroot, metric, prefix, dbname, split, num_split, preproc_str, ivf_str, pqflat_str)

	index_cachefile = '%s/%s_%s%s_%s_%s_%s%s,%s.index' % (
		cacheroot, metric, prefix, dbname, split, num_split, preproc_str, ivf_str, pqflat_str)


	print("cachefiles:")
	print(preproc_cachefile)
	print(cent_cachefile)
	print(pq_cachefile)
	print(index_cachefile)

	#################################################################
	# Wake up GPUs
	#################################################################

	print("preparing resources for %d GPUs" % ngpu)

	global gpu_resources 
	gpu_resources = []
	for i in range(ngpu):
		res = faiss.StandardGpuResources()
		if tempmem >= 0:
			res.setTempMemory(tempmem)
		gpu_resources.append(res)

	co = faiss.GpuMultipleClonerOptions()
	co.useFloat16 = use_float16
	co.useFloat16CoarseQuantizer = False
	co.usePrecomputed = use_precomputed_tables
	co.indicesOptions = 0
	co.verbose = True
	co.shard = True    # the replicas will be made "manually"

	preproc = get_preprocessor(xt)
	# indexall = prepare_trained_index(preproc, xt)
	print("--------------------------------------------\n")
	# return preproc, indexall
	return preproc

def build_faiss(dataset, split, preproc):
	print("--------------- build_faiss ----------------")
	index = get_populated_index(preproc, dataset, split)
	print("--------------------------------------------\n")
	return index




