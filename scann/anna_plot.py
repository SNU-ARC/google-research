import os
import matplotlib as mpl
mpl.use('Agg')  # noqa
import matplotlib.pyplot as plt
import numpy as np
import argparse

from anna_plot_util import *

# from ann_benchmarks.datasets import get_dataset
# from ann_benchmarks.algorithms.definitions import get_definitions
# from ann_benchmarks.plotting.metrics import all_metrics as metrics
# from ann_benchmarks.plotting.utils import (get_plot_label, compute_metrics,
#                                            create_linestyles, create_pointset)
# from ann_benchmarks.results import (store_results, load_all_results,
#                                     get_unique_algorithms)


# def create_plot(all_data, raw, x_scale, y_scale, xn, yn, fn_out, linestyles,
#                 batch):
def create_plot(dataset, results, linestyles):
    # xm, ym = (metrics[xn], metrics[yn])
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
    for algo in results:
        # xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        xs = algo['acc']
        ys = algo['time']
        min_x = min([min_x]+[x for x in xs if x > 0])
        max_x = max([max_x]+[x for x in xs if x < 1])
        color, faded, linestyle, marker = linestyles[algo['algorithm']]
        color, faded, linestyle, marker = linestyles[algo['algorithm']]
        handle, = plt.plot(xs, ys, '-', label='scann', color=color,
                           ms=4, mew=3, lw=3, linestyle=linestyle,
                           marker=marker)
        handles.append(handle)
    # if raw:
    #     handle2, = plt.plot(axs, ays, '-', label=algo, color=faded,
    #                         ms=5, mew=2, lw=2, linestyle=linestyle,
    #                         marker=marker)
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
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_title(get_plot_label(dataset))
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


    # # Workaround for bug https://github.com/matplotlib/matplotlib/issues/6789
    ax.spines['bottom']._adjust_location()

    plt.savefig("./result.png", bbox_inches='tight')
    plt.close()

def collect_result(path):
    acc = []
    time = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            print(line)
            if i==0:
                program, topk, num_split = line.split()[1], int(line.split()[3]), int(line.split()[5])
                print("Program: ", program, " / Topk: ", topk, " / Num split: ", num_split)
            elif i==1:
                continue
            elif i%2==0:
                line.split()
            else:
                result = line.split()
                if topk == 1:
                    acc.append(float(result[0]))
                elif topk == 10:
                    acc.append(float(result[2]))
                else:       # topk == 100
                    acc.append(float(result[4]))
                time.append(float(result[6]))

    # import operator 
    res = sorted(zip(acc, time), key = lambda x: x[0]) 
    # res = sorted(zip(acc, time), key = operator.itemgetter(0)) 
    acc, time = zip(*res)
    print("acc: ", acc)
    print("time: ", time)
    return {'acc': acc, 'time': time, 'algorithm': program}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        metavar="DATASET",
        default='glove')
    parser.add_argument(
        '--count',
        default=10)
    parser.add_argument(
        '--definitions',
        metavar='FILE',
        help='load algorithm definitions from FILE',
        default='algos.yaml')
    parser.add_argument(
        '--limit',
        default=-1)
    parser.add_argument(
        '-o', '--output')
    parser.add_argument(
        '-x', '--x-axis',
        help='Which metric to use on the X-axis',
        # choices=metrics.keys(),
        default="k-nn")
    parser.add_argument(
        '-y', '--y-axis',
        help='Which metric to use on the Y-axis',
        # choices=metrics.keys(),
        default="qps")
    parser.add_argument(
        '-X', '--x-scale',
        help='Scale to use when drawing the X-axis. Typically linear, logit or a2',
        default='linear')
    parser.add_argument(
        '-Y', '--y-scale',
        help='Scale to use when drawing the Y-axis',
        choices=["linear", "log", "symlog", "logit"],
        default='linear')
    parser.add_argument(
        '--raw',
        help='Show raw results (not just Pareto frontier) in faded colours',
        action='store_true')
    parser.add_argument(
        '--batch',
        help='Plot runs in batch mode',
        action='store_true')
    parser.add_argument(
        '--recompute',
        help='Clears the cache and recomputes the metrics',
        action='store_true')
    args = parser.parse_args()

    if not args.output:
        args.output = 'results/%s.png' % (args.dataset + ('-batch' if args.batch else ''))
        print('writing output to %s' % args.output)

    # dataset = get_dataset(args.dataset)
    # count = int(args.count)
    # unique_algorithms = get_unique_algorithms()
    # results = load_all_results(args.dataset, count, args.batch)
    # linestyles = create_linestyles(sorted(unique_algorithms))
    # runs = compute_metrics(np.array(dataset["distances"]),
    #                        results, args.x_axis, args.y_axis, args.recompute)
    # if not runs:
    #     raise Exception('Nothing to plot')

    results = list()
    for root, _, files in os.walk('./result'):
        for fn in files:
            if args.dataset in fn: 
                res = collect_result(os.path.join(root, fn))
                results.append(res)
    linestyles = create_linestyles([key['algorithm'] for key in results])

    # create_plot(runs, args.raw, args.x_scale,
    #             args.y_scale, args.x_axis, args.y_axis, args.output,
    #             linestyles, args.batch)
    create_plot(args.dataset, results, linestyles)
