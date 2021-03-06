'''Usage for sensitivity:
1. lsens
(1) python3 anna_plot_result.py --program scann --dataset sift1m --metric squared_l2 --reorder -1 --topk 1000 --build_config --lsens 1000 2000 800 --m 32
(2) python3 anna_plot_result.py --program faissGPU --dataset music1m --metric dot_product --reorder -1 --topk 1000 --build_config --lsens 1000 2000 3000 4000 --kstar 256 --m 50 --target 10
(3 for all possible Ls) python3 anna_plot_result.py --program scann --dataset sift1m --metric squared_l2 --reorder -1 --topk 1000 --build_config --lsens 9999 --m 32 --target 1000
2. msens
(1) python3 anna_plot_result.py --program scann --dataset sift1m --metric squared_l2 --reorder -1 --topk 1000 --build_config --msens 64 32 --L 2000
(2) python3 anna_plot_result.py --program faissGPU --dataset music1m --metric dot_product --reorder -1 --topk 1000 --build_config --msens 20 25 --kstar 256 --L 2000
3. kstarsens
python3 anna_plot_result.py --program faiss --dataset deep1m --metric squared_l2 --reorder -1 --topk 1000 --build_config --kstarsens 256 128 64 32 --L 600 --m 128
'''
import os
import matplotlib as mpl
mpl.use('Agg')  # noqa
import matplotlib.pyplot as plt
import numpy as np
import argparse

from anna_plot_util_result import *

def create_plot(dataset, results, linestyles, build_config, args):
    # Now generate each plot
    handles = []
    labels = []
    plt.figure(figsize=(12, 9))


    # Sorting by mean y-value helps aligning plots with labels
    # def mean_y(algo):
    #     xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        # return -np.log(np.array(ys)).mean()
    # Find range for logit x-scale
    min_x, max_x = 1, 0
    for i, algo in enumerate(results):
        xs = algo['acc']
        ys = algo['time']
        min_x = min([min_x]+[x for x in xs if x > 0])
        max_x = max([max_x]+[x for x in xs if x < 1])
        if build_config:
            print(algo['build_key'])
            color, faded, linestyle, marker = linestyles[algo['build_key']]
            handle, = plt.plot(xs, ys, '-', label=algo['build_key'], color=color,
                               ms=0.5, mew=3, lw=1.5, linestyle=linestyle,
                               marker=marker)
            for i, sc in enumerate(algo['search_key']):
                #test = [plt.annotate('${}$'.format(sc), xy=(xs[i], ys[i]), xytext=(xs[i]-0.7*(len(results)-i), ys[i]+max_time/100*i*10), color=color, arrowprops=dict(facecolor=color, width=0.1, headwidth=5, headlength=5, lw=0))]
                plt.annotate('${}$'.format(sc), xy=(xs[i], ys[i]), xytext=(xs[i]-0.3, ys[i]+max_time/1000*i), color=color, arrowprops=dict(facecolor=color, width=0.1, headwidth=0.5, headlength=0.5, lw=0))
                # from adjustText import adjust_text
                # adjust_text(test)


            handles.append(handle)
            labels.append(algo['algorithm']+"_"+algo['build_key'])
        else:
            color, faded, linestyle, marker = linestyles[algo['algorithm']]
            handle, = plt.plot(xs, ys, '-', label=algo['algorithm'], color=color,
                               ms=4, mew=3, lw=3, linestyle=linestyle,
                               marker=marker)
            handles.append(handle)
            labels.append(algo['algorithm'])

    ax = plt.gca()
    # ax.set_ylabel(ym['description'])
    # ax.set_xlabel(xm['description'])
    ax.set_ylabel('Query Total Latency (ms)')
    ax.set_xlabel('Recall (%)')

    # # Custom scales of the type --x-scale a3
    # if x_scale[0] == 'a':
    #     alpha = int(x_scale[1:])
    #     fun = lambda x: 1-(1-x)**(1/alpha)
    #     inv_fun = lambda x: 1-(1-x)**alpha
    #     ax.set_xscale('function', functions=(fun, inv_fun))
    #     if alpha <= 3:
    #         ticks = [inv_fun(x) for x in np.arange(0,1.2,.2)]
    #         plt.xticks(ticks)
    #     if alpha > 3:
    #         from matplotlib import ticker
    #         ax.xaxis.set_major_formatter(ticker.LogitFormatter())
    #         #plt.xticks(ticker.LogitLocator().tick_values(min_x, max_x))
    #         plt.xticks([0, 1/2, 1-1e-1, 1-1e-2, 1-1e-3, 1-1e-4, 1])
    # # Other x-scales
    # else:
    #     ax.set_xscale(x_scale)
    title = get_plot_label(dataset, program, batch_size, metric, args.target, reorder)
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_title(title)
    box = plt.gca().get_position()
    # plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles, labels, loc='center left',
              bbox_to_anchor=(1, 0.5), prop={'size': 9})
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.setp(ax.get_xminorticklabels(), visible=True)

    # # Logit scale has to be a subset of (0,1)
    # if 'lim' in xm and x_scale != 'logit':
    #     x0, x1 = xm['lim']
    #     plt.xlim(max(x0,0), min(x1,1))
    # elif x_scale == 'logit':
    #     plt.xlim(min_x, max_x)
    # if 'lim' in ym:
    #     plt.ylim(ym['lim'])

    x0, x1 = [0.0, 1.03]
    plt.xlim(max(x0,0), min(x1,1))


    # Workaround for bug https://github.com/matplotlib/matplotlib/issues/6789
    ax.spines['bottom']._adjust_location()
    os.makedirs("./plots/", exist_ok=True)

    if args.lsens != None:
        title += "_lsens_threshold_"+str(args.threshold)+"_m_"+str(args.m)+"_k_star_"+str(k_star)
    elif args.kstarsens != None:
        title += "_kstarsens_L_"+str(args.L)+"_threshold_"+str(args.threshold)+"_m_"+str(args.m)
    elif args.msens != None:
        title += "_msens_L_"+str(args.L)+"_threshold_"+str(args.threshold)+"_k_star_"+str(k_star)

    plt.savefig("./plots/"+title+".pdf", bbox_inches='tight')
    plt.close()

def collect_result(path, args):
    global batch_size
    global program
    global metric
    global topk
    global reorder
    global max_time
    print("Reading result from ", path)
    max_time = -1
    acc = []
    time = []
    sc = []
    build_keys = []
    collected_result = []
    build_key = None
    reorder = -2
    with open(path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            # print(line)
            if i==0:
                program, topk, num_split, batch_size = line.split()[1], int(line.split()[3]), int(line.split()[5]), int(line.split()[7])
                print("Program: ", program, " / Topk: ", topk, " / Num split: ", num_split, " / Batch size: ", batch_size)
                if args.build_config:
                    assert program == args.program and (batch_size==128 if args.program != "annoy" else True)
            elif i==1:
                continue
            elif i%2==0:
                result = line.split()
                print(line)
                print(result)
                temp_build_key = '/'.join(result[:result.index("|")])
                search_key = "/".join(result[result.index("|")+1:-2])  # make sure the last two is reorder and metric
                if reorder == -2:
                    reorder = result[-2]
                elif reorder!=result[-2]:
                    reorder="various"
                metric = result[-1]
                if args.build_config:
                    if temp_build_key not in build_keys:
                        build_keys.append(temp_build_key)
                        if i==2:
                            build_key = temp_build_key
                            # search_key = temp_search_key
                            continue
                        if len(acc)>0 and len(time)>0:
                            res = sorted(zip(acc, time, sc), key = lambda x: x[0])
                            acc, time, sc = zip(*res)
                            collected_result.append({'acc': acc, 'time': time, 'algorithm': program, 'build_key': build_key, 'search_key': sc})
                            acc = []
                            time = []
                            sc = []
                            build_key = temp_build_key
                            print(build_key)
                            # search_key = temp_search_key
            else:
                result = line.split()
                print(result)
                if args.target == "1@1":
                    acc.append(float(result[0]))
                elif args.target == "10@10":
                    acc.append(float(result[2]))
                elif args.target == "100@100":       # topk == 100
                    acc.append(float(result[4]))
                elif args.target == "1000@1000":
                    acc.append(float(result[6]))
                # time.append(float(result[8]))
                time.append(float(result[-1]))
                sc.append(search_key)
                # if max_time < float(result[8]):
                    # max_time = float(result[8])
                if max_time < float(result[-1]):
                    max_time = float(result[-1])

    res = sorted(zip(acc, time, sc), key = lambda x: x[0])
    acc, time, sc = zip(*res)
    # print("acc: ", acc)
    # print("time: ", time)
    collected_result.append({'acc': acc, 'time': time, 'algorithm': program, 'build_key': build_key, 'search_key': sc})
    assert len(collected_result) == len(build_keys)
    print("build_keys:", build_keys)

    for res in collected_result:
        print("--")
        print(res)
    return collected_result


def sensitivity_filter(collected_result, args):
    acc = []
    time = []
    sc = []
    build_keys = []
    filtered_collected_result = []
    idx_to_save = []
    if args.lsens != None:
        L_list = list(args.lsens)
        print(L_list)
    elif args.kstarsens != None:
        kstar_list = list(args.kstarsens)
        print(kstar_list)
    elif args.msens != None:
        m_list = list(args.msens)
        print(m_list)

    for i, res in enumerate(collected_result):
        build_key = res['build_key']
        curr_L_in_build_key = int(build_key.split("/")[0])
        curr_thresh_in_build_key = float(build_key.split("/")[1]) if args.program == "scann" else float(args.threshold)
        curr_m_in_build_key = int(build_key.split("/")[2]) if args.program == "scann" else int(build_key.split("/")[1])
        curr_kstar_in_build_key = int(k_star) if args.program == "scann" else int(build_key.split("/")[2])
        if args.lsens != None:
            if 9999 in L_list:
                if curr_thresh_in_build_key == float(args.threshold) and curr_m_in_build_key == int(args.m) and curr_kstar_in_build_key == k_star:
                    idx_to_save.append(i)
            else:
                if curr_L_in_build_key in L_list and curr_thresh_in_build_key == float(args.threshold) and curr_m_in_build_key == int(args.m) and curr_kstar_in_build_key == k_star:
                    idx_to_save.append(i)
        elif args.kstarsens != None:
            assert args.program != "scann"
            if 9999 in kstar_list:
                if curr_L_in_build_key == int(args.L) and curr_thresh_in_build_key == float(args.threshold) and curr_m_in_build_key == int(args.m):
                    idx_to_save.append(i)
            else:
                if curr_kstar_in_build_key in kstar_list and curr_L_in_build_key == int(args.L) and curr_thresh_in_build_key == float(args.threshold) and curr_m_in_build_key == int(args.m):
                    idx_to_save.append(i)
        elif args.msens != None:
            if 9999 in m_list:
                if curr_L_in_build_key == int(args.L) and curr_thresh_in_build_key == float(args.threshold) and curr_kstar_in_build_key == k_star:
                    idx_to_save.append(i)
            else:
                if curr_m_in_build_key in m_list and curr_L_in_build_key == int(args.L) and curr_thresh_in_build_key == float(args.threshold) and curr_kstar_in_build_key == k_star:
                    idx_to_save.append(i)

    for i, res in enumerate(collected_result):
        if i in idx_to_save:
            filtered_collected_result.append(res)

    print("filtered_collected_result =", filtered_collected_result)
    return filtered_collected_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', metavar="DATASET", default=None)
    parser.add_argument('--program', metavar="ALGO", default=None)
    parser.add_argument('--metric', metavar="METRIC", default=None)
    parser.add_argument('--topk', metavar="TOPK", default=None)
    parser.add_argument('--reorder', metavar="REORDER", default=None)
    parser.add_argument('-o', '--output')
    parser.add_argument('--build_config', help='Whether to plot according to the build_config', action='store_true', default=False)
    parser.add_argument('--lsens', metavar="LSENS", nargs='+', type=int, default=None)
    parser.add_argument('--kstarsens', metavar="KSTARSENS", nargs='+', type=int, default=None)
    parser.add_argument('--msens', metavar="MSENS", nargs='+', type=int, default=None)
    parser.add_argument('--L', metavar="L", default=None)
    parser.add_argument('--m', metavar="M", default=None)
    parser.add_argument('--kstar', metavar="KSTAR", default=None)
    parser.add_argument('--threshold', metavar="THRESH", default=0.2)
    parser.add_argument('--target', metavar="TARGET", default="1000@1000")


    args = parser.parse_args()
    global k_star
    if args.kstarsens == None and (args.lsens != None or args.msens != None):
        k_star = 16 if args.program == "scann" else int(args.kstar)

    if args.build_config:
        assert args.program!=None and args.dataset!=None and args.metric!=None and args.reorder!=None

    if args.lsens != None:
        assert args.L == None and args.kstarsens == None and args.msens == None and args.m != None and args.threshold != None and k_star != None
    elif args.kstarsens != None:
        k_star = None
        assert k_star == None and args.lsens == None and args.msens == None and args.L != None and args.m != None
    elif args.msens != None:
        assert args.m == None and args.lsens == None and args.kstarsens == None and args.L != None and k_star != None and args.threshold != None

    def check_build_config(fn):
        return (args.metric in fn) and ("GPU" in fn if ("GPU" in args.program) else "GPU" not in fn)

    results = list()
    for root, _, files in os.walk('./'):
        if "plot" in root:
            continue
        for fn in files:
            if args.dataset in fn and (args.topk in fn) and (args.program in fn if args.program!=None else True) and (args.metric in fn) and (args.reorder in fn) and (check_build_config(fn) if args.build_config==True else True):
                res = collect_result(os.path.join(root, fn), args)
                if args.lsens != None or args.kstarsens != None or args.msens != None:
                    res = sensitivity_filter(res, args)
                results+=res
    assert len(results) > 0
    linestyles = create_linestyles([key['build_key'] for key in results])

    ## create_plot(runs, args.raw, args.x_scale,
    ##             args.y_scale, args.x_axis, args.y_axis, args.output,
    ##             linestyles, args.batch)
    create_plot(args.dataset, results, linestyles, args.build_config, args)
