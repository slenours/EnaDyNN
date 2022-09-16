"""
visualize the results
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from branchynet import utils


def plot_layers(values, save_path=None, save_name=None, xlabel='', ylabel=''):
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 18}
    matplotlib.rc('font', **font)

    plt.figure(figsize=(12, 10))
    if type(values[0]) in [tuple, list]:
        for i, val in enumerate(values):
            plt.plot(val, label='Layer ' + str(i), linewidth=4.0)
    else:
        plt.plot(values, label='Last Layer', linewidth=4.0)

    plt.legend(loc='best')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save_path != None and save_name != None:
        plt.title(save_name)
        plt.savefig(save_path + save_name + '.png')
    plt.show()


def plot_line_tradeoff(accs, diffs, ps, exits, baseacc, basediff, orig_label='Baseline', title=None,
                       our_label='Our Method',
                       xlabel='Runtime (s)', ylabel='Classification Accuracy', all_samples=False, knee_idx=None,
                       xlim=None, ylim=None, inc_amt=-0.0005, output_path=None, output_name=None):
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 18}
    matplotlib.rc('font', **font)
    plt.figure(figsize=(12, 10))

    inc_accs, inc_rts, _, _ = utils.get_inc_points(accs, diffs, ps, exits, inc_amt=inc_amt)
    if all_samples:
        plt.plot(diffs, accs, 'go', linewidth=4, label=our_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if knee_idx:
        plt.plot(inc_rts[0], np.array(inc_accs[0]) * 100.,
                 '-o', color='#5163d3', ms=6, linewidth=4, label=our_label)
        plt.plot(inc_rts, np.array(inc_accs) * 100.,
                 '-', color='#5163d3', ms=6, linewidth=4)
        plt.plot(np.delete(inc_rts, knee_idx), np.delete(np.array(inc_accs) * 100., knee_idx),
                 'o', color='#5163d3', ms=6)
        plt.plot(inc_rts[knee_idx], np.array(inc_accs)[knee_idx] * 100., '*', color='#50d350', ms=16,
                 label=our_label + ' knee')

    else:
        plt.plot(inc_rts, np.array(inc_accs) * 100.,
                 '-o', color='#5163d3', ms=6, linewidth=4, label=our_label)
    plt.plot(basediff, baseacc * 100., 'D', color='#d3515a', ms=8, label=orig_label)

    plt.legend(loc='best')

    if title:
        plt.title(title)

    if output_path:
        plt.savefig(output_path + output_name + '.png', bbox_inches='tight')
        plt.show()


def plot_acc_entropy_exit(accs, exits, entropies, xlabel='Samples Exited at 1st Branch (%)',
                          ylabel1='Classification Accuracy', ylabel2='Max Entropy',
                          save_path=None, save_name=None):
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 18}
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(111)
    first_exits = []
    for exit in exits:
        first_exits.append(exit[0] / sum(exit))

    ln1 = ax1.plot(np.asarray(first_exits) * 100, accs * 100., label='Accuracy', linewidth=4.0)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1)

    ax2 = ax1.twinx()
    ln2 = ax2.plot(np.asarray(first_exits) * 100, entropies, 'g', label='Entropy', linewidth=4.0)
    ax2.set_ylabel(ylabel2)

    ln = ln1 + ln2
    labs = [l.get_label() for l in ln]
    plt.legend(ln, labs, loc='best')
    if save_path != None and save_name != None:
        plt.title(save_name)
        plt.savefig(save_path + save_name + '.png')

    plt.show()


def plot_acc_layers(accs, xlabel='Conv Layers in 1st Branch',
                    ylabel='Classification Accuracy', save_path=None, save_name=None):
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 18}
    matplotlib.rc('font', **font)

    plt.plot(accs, '-o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save_path != None and save_name != None:
        plt.title(save_name)
        plt.savefig(save_path + save_name + '.png')

    plt.show()
