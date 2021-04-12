import re
import faiss
import time
import os, sys
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool


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


class IdentPreproc:
    """a pre-processor is either a faiss.VectorTransform or an IndentPreproc"""

    def __init__(self, d):
        self.d_in = self.d_out = d

    def apply_py(self, x):
        return x


def sanitize(x):
    """ convert array to a c-contiguous float array """
    return np.ascontiguousarray(x.astype('float32'))


def train_preprocessor(preproc_str):
    print("train preproc", preproc_str)
    t0 = time.time()
    if preproc_str.startswith('OPQ'):
        fi = preproc_str[3:-1].split('_')
        m = int(fi[0])
        dout = int(fi[1]) if len(fi) == 2 else dim
        preproc = faiss.OPQMatrix(dim, m, dout)
    elif preproc_str.startswith('PCAR'):
        dout = int(preproc_str[4:-1])
        preproc = faiss.PCAMatrix(dim, dout, 0, True)
    else:
        assert False
    preproc.train(sanitize(xt))
    print("preproc train done in %.3f s" % (time.time() - t0))
    return preproc


def get_preprocessor(preproc_str, preproc_cachefile):
    if preproc_str:
        if not preproc_cachefile or not os.path.exists(preproc_cachefile):
            preproc = train_preprocessor(preproc_str)
            if preproc_cachefile:
                print("store", preproc_cachefile)
                faiss.write_VectorTransform(preproc, preproc_cachefile)
        else:
            print("load", preproc_cachefile)
            preproc = faiss.read_VectorTransform(preproc_cachefile)
    else:
        preproc = IdentPreproc(dim)
    return preproc


def train_coarse_quantizer(x, k, preproc, is_gpu):
    d = preproc.d_out
    clus = faiss.Clustering(d, k)
    clus.verbose = True
    clus.max_points_per_centroid = 10000000

    print("apply preproc on shape", x.shape, 'k=', k)
    t0 = time.time()
    x = preproc.apply_py(sanitize(x))
    print("   preproc %.3f s output shape %s" % (
        time.time() - t0, x.shape))

    if is_gpu:
        vres, vdev = make_vres_vdev()
        index = faiss.index_cpu_to_gpu_multiple(
            vres, vdev, faiss.IndexFlat(d, fmetric))
    else:
        index = faiss.IndexFlat(d, fmetric)
    clus.train(x, index)
    centroids = faiss.vector_float_to_array(clus.centroids)

    return centroids.reshape(k, d)


def prepare_coarse_quantizer(preproc, cent_cachefile, ncent, is_gpu):

    if cent_cachefile and os.path.exists(cent_cachefile):
        print("load centroids", cent_cachefile)
        centroids = np.load(cent_cachefile)
    else:
        nt = max(1000000, 256 * ncent)
        print("train coarse quantizer...")
        t0 = time.time()
        centroids = train_coarse_quantizer(xt, ncent, preproc, is_gpu)
        print("centroids:", centroids[128])
        print("Coarse train time: %.3f s" % (time.time() - t0))
        if cent_cachefile:
            print("store centroids", cent_cachefile)
            np.save(cent_cachefile, centroids)

    coarse_quantizer = faiss.IndexFlat(preproc.d_out, fmetric)
    coarse_quantizer.add(centroids)

    return coarse_quantizer


def prepare_trained_index(preproc, coarse_quantizer, ncent, pqflat_str):

    d = preproc.d_out
    if pqflat_str == 'Flat':
        print("making an IVFFlat index")
        idx_model = faiss.IndexIVFFlat(coarse_quantizer, d, ncent, fmetric)
    elif 'SQ' in pqflat_str:
        print("making a SQ index")
        if fmetric == faiss.METRIC_L2:
            quantizer = faiss.IndexFlatL2(d)
        elif fmetric == faiss.METRIC_INNER_PRODUCT:
            quantizer = faiss.IndexFlatIP(d)
        if pqflat_str.split("SQ")[1] == "16":
            name = "QT_fp16"
        else:
            name = "QT_" + str(pqflat_str.split("SQ")[1]) + "bit"
        qtype = getattr(faiss.ScalarQuantizer, name)
        idx_model = faiss.IndexIVFScalarQuantizer(quantizer, d, ncent, qtype, fmetric)
    else:
        key = pqflat_str[2:].split("x")
        assert len(key) == 2, "use format PQ(m)x(log2kstar)"
        m, log2kstar = map(int, pqflat_str[2:].split("x"))
        assert m < 56 or useFloat16, "PQ%d will work only with -float16" % m

        print("making an IVFPQ index, m = %d, log2kstar = %d" % (m, log2kstar))
        idx_model = faiss.IndexIVFPQ(coarse_quantizer, d, ncent, m, log2kstar, fmetric)

    coarse_quantizer.this.disown()
    idx_model.own_fields = True

    # finish training on CPU
    t0 = time.time()
    print("Training vector codes")
    x = preproc.apply_py(sanitize(xt))
    idx_model.train(x)
    print("  done %.3f s" % (time.time() - t0))

    return idx_model


def add_vectors(index_cpu, preproc, is_gpu, addBatchSize):

    # copy to GPU
    if is_gpu:
        index = copyToGpu(index_cpu)
    else:
        index = index_cpu

    # add
    nb = xb.shape[0]
    t0 = time.time()
    for i0, xs in dataset_iterator(xb, preproc, addBatchSize):
        i1 = i0 + xs.shape[0]
        index.add_with_ids(xs, np.arange(i0, i1))
        print('\r%d/%d (%.3f s)  ' % (
            i0, nb, time.time() - t0), end=' ')
        sys.stdout.flush()
    print("Add time: %.3f s" % (time.time() - t0))

    # copy to CPU
    if is_gpu:
        index_all = index_cpu
        print("Aggregate indexes to CPU")
        t0 = time.time()
        if hasattr(index, 'at'):

            # it is a sharded index
            for i in range(ngpu):
                index_src = faiss.index_gpu_to_cpu(index.at(i))
                print("  index %d size %d" % (i, index_src.ntotal))
                index_src.copy_subset_to(index_all, 0, 0, nb)
        else:
            # simple index
            index_src = faiss.index_gpu_to_cpu(index)
            index_src.copy_subset_to(index_all, 0, 0, nb)
        print("  done in %.3f s" % (time.time() - t0))
        return index_all, index
    else:
        return index, None


def copyToGpu(index_cpu):

    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = useFloat16
    co.useFloat16CoarseQuantizer = False
    co.usePrecomputed = usePrecomputed
    co.indicesOptions = faiss.INDICES_CPU
    co.verbose = True
    co.reserveVecs = N
    co.shard = True
    assert co.shard_type in (0, 1, 2)
    vres, vdev = make_vres_vdev()
    index_gpu = faiss.index_cpu_to_gpu_multiple(
        vres, vdev, index_cpu, co)

    return index_gpu


def process_index_key(index_key):
    pattern = None
    if "SQ" in index_key:
        pattern = re.compile('(OPQ[0-9]+(_[0-9]+)?,|PCAR[0-9]+,)?' +
                         '(IVF[0-9]+),' +
                         '(SQ[0-9]+)')
    else:
        pattern = re.compile('(OPQ[0-9]+(_[0-9]+)?,|PCAR[0-9]+,)?' +
                         '(IVF[0-9]+),' +
                         '(PQ[0-9]+(x[0-9]+)?|Flat)')

    matchobject = pattern.match(index_key)
    assert matchobject, 'could not parse ' + index_key

    mog = matchobject.groups()

    return mog[0], mog[2], mog[3]   # preproc, ivf, pqflat


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
    pool.terminate()

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


def build_faiss(args, cacheroot, coarse_dir, split, N_, D, index_key, is_cached, query_, train=None, base=None):

    # set global variables
    name1_to_metric = {
        "dot_product": faiss.METRIC_INNER_PRODUCT,
        "squared_l2": faiss.METRIC_L2
    }
    global fmetric
    fmetric = name1_to_metric[args.metric]
    global xt
    if is_cached == False:
        xt = sanitize(train)
    global xb
    if is_cached == False:
        xb = base
    global dbname
    dbname = args.dataset
    global dim
    dim = D
    global gpu_resources
    global ngpu
    global usePrecomputed
    global useFloat16
    global query
    query = sanitize(query_)
    global N
    N = N_

    usePrecomputed = False
    useFloat16 = True
    print("usefloat16? ", useFloat16)
    replicas = 1
    addBatchSize = 32768
    ngpu = faiss.get_num_gpus()
    tempmem = -1

    if ngpu == 0 and args.is_gpu==True:
        assert False, "Cannot detect gpu in this machine"

    # process index_key
    preproc_str, ivf_str, pqflat_str = process_index_key(index_key)
    ncentroid = int(ivf_str[3:])

    # check cache files
    if not os.path.isdir(cacheroot):
        print("%s does not exist, creating it" % cacheroot)
        os.makedirs(cacheroot, exist_ok=True)

    print("cachefiles:")
    if preproc_str:
        preproc_cachefile = '%s%s_preproc_%s_%s.vectrans' % (
            cacheroot, args.metric, dbname, preproc_str[:-1])
        print(preproc_cachefile)
    else:
        preproc_str = ''
        preproc_cachefile = None

    cent_cachefile = '%s%s_cent_%s_%s%s_%s.npy' % (
        coarse_dir, args.metric, dbname, preproc_str, ivf_str, D)
    print(cent_cachefile)

    index_cachefile = '%s%s_%s_%s_%s_%s%s,%s.index' % (
        cacheroot, args.metric, dbname, split, args.num_split, preproc_str, ivf_str, pqflat_str)
    print(index_cachefile)

    first_index_cachefile = '%s%s_%s_0_%s_%s%s,%s.index' % (
        cacheroot, args.metric, dbname, args.num_split, preproc_str, ivf_str, pqflat_str)
    print(index_cachefile)

    # GPU resources
    if args.is_gpu:
        gpu_resources = []
        for i in range(ngpu):
            res = faiss.StandardGpuResources()
            if tempmem >= 0:
                res.setTempMemory(tempmem)
            gpu_resources.append(res)

    # pre-processing
    preproc = get_preprocessor(preproc_str, preproc_cachefile)

    # build index
    if not index_cachefile or not os.path.exists(index_cachefile):
        # train index
        coarse_quantizer = prepare_coarse_quantizer(preproc, cent_cachefile, ncentroid, args.is_gpu)
        if split == 0:
            index_trained = prepare_trained_index(preproc, coarse_quantizer, ncentroid, pqflat_str)
        else:
            index_trained = faiss.read_index(first_index_cachefile)
            index_trained.ntotal = 0
            index_trained.invlists.reset()

        # centroids = faiss.vector_to_array(index_trained.pq.centroids).reshape(index_trained.pq.M, index_trained.pq.ksub, index_trained.pq.dsub)
        # print("index_load: ", centroids.shape)
        # print("index_load: ", centroids)

        index_all, index_gpu = add_vectors(index_trained, preproc, args.is_gpu, addBatchSize)

        if index_cachefile:
            print("store", index_cachefile)
            faiss.write_index(index_all, index_cachefile)

        if args.is_gpu:
            index = index_gpu
        else:
            index = index_all
    else:
        print("load", index_cachefile)
        index_load = faiss.read_index(index_cachefile)

        # move to GPU
        if args.is_gpu:
            index = copyToGpu(index_load)
            del index_load
        else:
            index = index_load

    global ps
    index.use_precomputed_table = usePrecomputed
    if args.is_gpu:
        ps = faiss.GpuParameterSpace()
        ps.initialize(index)
        # ps.set_index_parameter(index, 'nprobe', w)
    else:
        # faiss.omp_set_num_threads(faiss.omp_get_max_threads())
        faiss.omp_set_num_threads(args.batch)
        # index.nprobe = w

    return index, preproc

def check_cached(cacheroot, args, dbname, split, num_split, index_key):
    preproc_str, ivf_str, pqflat_str = process_index_key(index_key)
    if preproc_str == None:
        preproc_str = ''
    index_cachefile = '%s%s_%s_%s_%s_%s%s,%s.index' % (
        cacheroot, args.metric, dbname, split, num_split, preproc_str, ivf_str, pqflat_str)
    print("Checking ", index_cachefile)
    if not index_cachefile or not os.path.exists(index_cachefile):
        print("Cache file does not exist..")
        return False
    else:
        print("Cache file exists!")
        return True

def faiss_search(index, preproc, args, reorder, w):
    # search environment
    # index.use_precomputed_table = usePrecomputed
    # if args.is_gpu:
    #     ps = faiss.GpuParameterSpace()
    #     ps.initialize(index)
    #     ps.set_index_parameter(index, 'nprobe', w)
    # else:
    #     faiss.omp_set_num_threads(faiss.omp_get_max_threads())
    #     index.nprobe = w
    if args.is_gpu:
        # ps = faiss.GpuParameterSpace()
        # ps.initialize(index)
        ps.set_index_parameter(index, 'nprobe', w)
    else:
        # faiss.omp_set_num_threads(faiss.omp_get_max_threads())
        index.nprobe = w

    # reorder
    if reorder != -1 and not args.is_gpu:
        index_refine = faiss.IndexRefineFlat(index, faiss.swig_ptr(xb))
        index_refine.k_factor = reorder / args.topk
        index_ready = index_refine
    else:
        index_ready = index

    # search
    print("Batch size: ", args.batch)
    nq = query.shape[0]
    I = np.empty((nq, args.topk), dtype='int32')
    D = np.empty((nq, args.topk), dtype='float32')
    SOW = np.empty((nq*(w+1), 1), dtype=np.uint)

    total_latency = 0.0
    iter = 0
    previous_end = 0
    for i0, xs in dataset_iterator(query, preproc, args.batch):
        i1 = i0 + xs.shape[0]
        start = time.time()
        Di, Ii, SOWi = index_ready.search(xs, args.topk, None, None, None, w)
        total_latency += 1000*(time.time()-start)
        I[i0:i1] = Ii
        D[i0:i1] = Di
        curr_batch = i1 - i0
        chunk_size = curr_batch * (w+1)
        # print("start =", previous_end, ", chunk_size =", chunk_size)
        # SOW[ chunk_size*iter : chunk_size*iter+chunk_size ] = SOWi
        SOW[ previous_end : previous_end+chunk_size ] = SOWi
        previous_end = chunk_size*iter+chunk_size
        iter += 1
    # for i in range(nq*(w+1)):
    #     print("SOW[", i, "] =", SOW[i])
    return I, D, SOW, total_latency
