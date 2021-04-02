import os
import matplotlib as mpl
mpl.use('Agg')  # noqa
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

from anna_plot_util import *

def calculate_traffic_theory(args, result, w):
    m = -1
    log2kstar = -1
    log2n = -1
    SOW = -1
    n = -1
    qn = -1
    L = int(result['build_key'].split('/')[0])
    ###################################
    # FIXME:: Check n, qn of datasets #
    ###################################
    if args.dataset == "sift1m":
        n = 1000000
        qn = 10000
    elif args.dataset == "glove":
        n = 1183514
        qn = 10000
    elif args.dataset == "sift1b":
        n = 1000000000
        qn = 10000
    elif args.dataset == "deep1b":
        n = 1000000000
        qn = 10000
    elif args.dataset == "gist":
        n = 1000000
        qn = 1000
    else:
        assert False

    if args.program == "scann":
        m = result['build_key'].split('/')[2]
        log2kstar = 4
    elif "faiss" in args.program:
        m = result['build_key'].split('/')[1]
        log2kstar = math.log(int(result['build_key'].split('/')[2]), 2)
    else:
        assert False

    log2n = math.log(n, 2)
    SOW = n / L * w

    assert m != -1 and log2kstar != -1 and log2n != -1 and SOW != -1 and n != -1 and qn != -1

    mem_traffic = int(qn) * ( int(m) * int(log2kstar) + float(log2n) ) * float(SOW) / (1024 * 1024 * 1024 * 8)
    return mem_traffic


def calculate_traffic(args, result, SOW):
    m = -1
    log2kstar = -1
    log2n = -1
    n = -1
    qn = -1
    ##################################
    # FIXME: Check n, qn of datasets #
    ##################################
    if args.dataset == "sift1m":
        n = 1000000
        qn = 10000
    elif args.dataset == "glove":
        n = 1183514
        qn = 10000
    elif args.dataset == "sif1b":
        n = 1000000000
        qn = 10000
    elif args.dataset == "deep1b":
        n = 1000000000
        qn = 10000
    elif args.dataset == "gist":
        n = 1000000
        qn = 1000
    else:
        assert False

    if args.program == "scann":
        m = result['build_key'].split('/')[2]
        log2kstar = 4
    elif "faiss" in args.program:
        m = result['build_key'].split('/')[1]
        log2kstar = math.log(int(result['build_key'].split('/')[2]), 2)
    else:
        assert False

    log2n = math.log(n, 2)

    assert m != -1 and log2kstar != -1 and log2n != -1 and n != -1 and qn != -1

    mem_traffic = int(qn) * ( int(m) * int(log2kstar) + float(log2n) ) * float(SOW) / (1024 * 1024 * 1024 * 8)
    return mem_traffic


def create_traffic_theory_plot(args, results, linestyles):
    print("arcm::create_traffic_theory_plot\n");
    dataset = args.dataset
    build_config = args.build_config
    # Now generate each plot
    handles = []
    labels = []
    plt.figure(figsize=(12, 9))

    # Sorting by mean y-value helps aligning plots with labels
    # def mean_y(algo):
    #     xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        # return -np.log(np.array(ys)).mean()
    # Find range for logit x-scale

    for idx, algo in enumerate(results):
        xs = list()
        for i, sc in enumerate(algo['search_key']):
            x_val = calculate_traffic_theory(args, results[idx], int(sc))
            xs.append(x_val)
        ys = algo['time']
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
    ax.set_xlabel('Memory Traffic (GiB)')

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
    title = get_plot_label(dataset, program, batch_size, metric, topk, reorder)
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

    # x0, x1 = [0.0, 1.03]
    # plt.xlim(max(x0,0), min(x1,1))

    # Workaround for bug https://github.com/matplotlib/matplotlib/issues/6789
    ax.spines['bottom']._adjust_location()
    os.makedirs("./result/plots_traffic/", exist_ok=True)

    plt.savefig("./result/plots_traffic/"+title+"-traffic_theory"+".pdf", bbox_inches='tight')
    plt.close()


def create_traffic_plot(dataset, results, linestyles, build_config):
    print("arcm::create_traffic__plot\n");
    # Now generate each plot
    handles = []
    labels = []
    plt.figure(figsize=(12, 9))


    # Sorting by mean y-value helps aligning plots with labels
    # def mean_y(algo):
    #     xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        # return -np.log(np.array(ys)).mean()
    # Find range for logit x-scale
    for idx, algo in enumerate(results):
        xs = list()
        for i, sc in enumerate(algo['search_key']):
            x_val = calculate_traffic(args, results[idx], float(results[idx]['SOW_avg'][i]))
            xs.append(x_val)
        ys = algo['time']
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
    ax.set_xlabel('Memory Traffic (GiB)')

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
    title = get_plot_label(dataset, program, batch_size, metric, topk, reorder)
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

    # x0, x1 = [0.0, 1.03]
    # plt.xlim(max(x0,0), min(x1,1))


    # Workaround for bug https://github.com/matplotlib/matplotlib/issues/6789
    ax.spines['bottom']._adjust_location()
    os.makedirs("./result/plots_traffic/", exist_ok=True)

    plt.savefig("./result/plots_traffic/"+title+"-traffic"+".pdf", bbox_inches='tight')
    plt.close()

def create_plot(dataset, results, linestyles, build_config):
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
        # print("arcm::xs =", xs)
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
    title = get_plot_label(dataset, program, batch_size, metric, topk, reorder)
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
    os.makedirs("./result/plots/", exist_ok=True)

    plt.savefig("./result/plots/"+title+".pdf", bbox_inches='tight')
    plt.close()

def collect_traffic_result(path, args):
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
    SOW_avg = []
    SOW_per_time = []
    sc = []
    build_keys = []
    collected_result = []
    build_key = None
    reorder = -2
    with open(path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i==0:
                program, topk, num_split, batch_size = line.split()[1], int(line.split()[3]), int(line.split()[5]), int(line.split()[7])
                print("Program: ", program, " / Topk: ", topk, " / Num split: ", num_split, " / Batch size: ", batch_size)
                if args.build_config:
                    assert program == args.program and (batch_size==128 if args.program != "annoy" else True)
            elif i==1:
                continue
            elif i%2==0:
                result = line.split()
                # print(line)
                # print(result)
                assert len(result) == 7
                temp_build_key = '/'.join(result[:result.index("|")])
                search_key = "/".join(result[result.index("|")+1:-2])  # make sure the last two are reorder and metric
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
                            res = sorted(zip(acc, time, sc, SOW_avg, SOW_per_time), key = lambda x: x[0])
                            acc, time, sc, SOW_avg, SOW_per_time = zip(*res)
                            collected_result.append({'acc': acc, 'time': time, 'algorithm': program, 'build_key': build_key, 'search_key': sc, 'SOW_avg': SOW_avg, 'SOW_per_time': SOW_per_time})
                            acc = []
                            time = []
                            sc = []
                            SOW_avg = []
                            SOW_per_time = []
                            build_key = temp_build_key
                            print(build_key)
                            # search_key = temp_search_key
            else:
                result = line.split()
                assert len(result) == 11
                # print(result)
                if topk == 1:
                    acc.append(float(result[0]))
                elif topk == 10:
                    acc.append(float(result[2]))
                elif topk == 100:       # topk == 100
                    acc.append(float(result[4]))
                else:
                    acc.append(float(result[6]))
                time.append(float(result[8]))
                SOW_avg.append(float(result[9]))
                SOW_per_time.append(float(result[10]))
                sc.append(search_key)
                if max_time < float(result[8]):
                    max_time = float(result[8])

    res = sorted(zip(acc, time, sc, SOW_avg, SOW_per_time), key = lambda x: x[0])
    acc, time, sc, SOW_avg, SOW_per_time = zip(*res)
    # print("acc: ", acc)
    # print("time: ", time)
    collected_result.append({'acc': acc, 'time': time, 'algorithm': program, 'build_key': build_key, 'search_key': sc, 'SOW_avg': SOW_avg, 'SOW_per_time': SOW_per_time})
    assert len(collected_result) == len(build_keys)
    print("build_keys:", build_keys)

    for res in collected_result:
        print("--")
        print(res)
    return collected_result

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
                # print(line)
                # print(result)
                assert len(result) == 7
                temp_build_key = '/'.join(result[:result.index("|")])
                search_key = "/".join(result[result.index("|")+1:-2])  # make sure the last two are reorder and metric
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
                if topk == 1:
                    acc.append(float(result[0]))
                elif topk == 10:
                    acc.append(float(result[2]))
                elif topk == 100:       # topk == 100
                    acc.append(float(result[4]))
                else:
                    acc.append(float(result[6]))
                time.append(float(result[8]))
                sc.append(search_key)
                if max_time < float(result[8]):
                    max_time = float(result[8])

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', metavar="DATASET", default=None)
    parser.add_argument('--program', metavar="ALGO", default=None)
    parser.add_argument('--metric', metavar="METRIC", default=None)
    parser.add_argument('--topk', metavar="TOPK", default=None)
    parser.add_argument('--reorder', metavar="REORDER", default=None)
    parser.add_argument('--traffic', help='Whether to plot actual memory traffic', action='store_true', default=False)
    parser.add_argument('--traffic_theory', help='Whether to plot memory traffic in theory', action='store_true', default=False)
    parser.add_argument('-o', '--output')
    parser.add_argument('--build_config', help='Whether to plot according to the build_config', action='store_true', default=False)
    args = parser.parse_args()

    if args.traffic:
        assert args.traffic_theory == False
    elif args.traffic_theory:
        assert args.traffic == False

    if args.build_config:
        assert args.program!=None and args.dataset!=None and args.metric!=None and args.reorder!=None

    def check_build_config(fn):
        return (args.metric in fn) and ("GPU" in fn if ("GPU" in args.program) else "GPU" not in fn)

    results = list()
    for root, _, files in os.walk('./result'):
        if "plot" in root:
            continue
        for fn in files:
            topk_part = "topk_" + args.topk
            reorder_part = "reorder_" + args.reorder
            if args.dataset in fn and (topk_part in fn) and (args.program in fn if args.program!=None else True) and (args.metric in fn) and (reorder_part in fn) and (check_build_config(fn) if args.build_config==True else True):
                if args.traffic:
                    if "traffic" not in fn:
                        continue
                    res = collect_traffic_result(os.path.join(root, fn), args)
                    results+=res
                else:
                    res = collect_result(os.path.join(root, fn), args)
                    results+=res
    assert len(results) > 0
    linestyles = create_linestyles([key['build_key'] for key in results])

    # create_plot(runs, args.raw, args.x_scale,
    #             args.y_scale, args.x_axis, args.y_axis, args.output,
    #             linestyles, args.batch)
    if args.traffic:
        create_traffic_plot(args.dataset, results, linestyles, args.build_config)
    elif args.traffic_theory:
        create_traffic_theory_plot(args, results, linestyles)
    else:
        create_plot(args.dataset, results, linestyles, args.build_config)
