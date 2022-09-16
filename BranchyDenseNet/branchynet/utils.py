
from chainer import Variable
from itertools import product
import chainer
import numpy as np
import time, csv


def train(branchyNet, x_train, y_train, batchsize=10000, num_epoch=20, main=False):
    datasize = x_train.shape[0]

    losses_list = []
    accuracies_list = []
    totaltime = 0

    for epoch in range(100, 199):
        indexes = np.random.permutation(datasize)

        if branchyNet.verbose:
            print("Epoch{}:".format(epoch))

        losses = []
        accuracies = []

        for i in range(0, datasize, batchsize):
            input_data = x_train[indexes[i: i + batchsize]]
            label_data = y_train[indexes[i: i + batchsize]]

            input_data = branchyNet.xp.asarray(input_data, dtype=branchyNet.xp.float32)
            label_data = branchyNet.xp.asarray(label_data, dtype=branchyNet.xp.int32)

            x = Variable(input_data)
            t = Variable(label_data)

            start_time = time.time()
            if main:
                loss, accuracy = branchyNet.train_main(x, t)  # train main network
            else:
                loss, accuracy = branchyNet.train(x, t)  # train network with branches
            end_time = time.time()
            run_time = end_time - start_time  # test time of a batch
            totaltime += run_time  # the total test time

            losses.append(loss)
            accuracies.append(accuracy)

        avg_losses = branchyNet.xp.mean(branchyNet.xp.asarray(losses, dtype=branchyNet.xp.float32), 0)
        avg_accuracies = branchyNet.xp.mean(branchyNet.xp.asarray(accuracies, dtype=branchyNet.xp.float32), 0)

        losses_list.append(avg_losses)
        accuracies_list.append(avg_accuracies)
    return losses_list, accuracies_list, totaltime


def test(branchyNet, x_test, y_test=None, batchsize=10000, main=False):
    datasize = x_test.shape[0]

    overall = 0.
    totaltime = 0.
    nsamples = 0
    num_exits = np.zeros(branchyNet.numexits()).astype(int)  # list of exiting samples
    accbreakdowns = np.zeros(branchyNet.numexits())  # accuracy of exit point
    max_entropy = 0

    for i in range(0, datasize, batchsize):
        input_data = x_test[i: i + batchsize]
        label_data = y_test[i: i + batchsize]

        input_data = branchyNet.xp.asarray(input_data, dtype=branchyNet.xp.float32)
        label_data = branchyNet.xp.asarray(label_data, dtype=branchyNet.xp.int32)

        x = Variable(input_data)
        t = Variable(label_data)

        with chainer.no_backprop_mode():
            if main:
                # test in main network, return accuracy and time
                acc, diff = branchyNet.test_main(x, t)
            else:
                # test in network with branches
                acc, accuracies, test_exits, diff, entropy = branchyNet.test(x, t)
                # get number of exiting samplesat each exit point
                for i, exits in enumerate(test_exits):
                    num_exits[i] += exits
                # get accuracy at each exit point
                for i in range(branchyNet.numexits()):
                    accbreakdowns[i] += accuracies[i] * test_exits[i]
                # max entropy of the first exit point
                if entropy > max_entropy:
                    max_entropy = entropy

            totaltime += diff
            overall += input_data.shape[0] * acc
            nsamples += input_data.shape[0]

    overall /= nsamples

    for i in range(branchyNet.numexits()):
        if num_exits[i] > 0:
            accbreakdowns[i] /= num_exits[i]

    return overall, totaltime, num_exits, accbreakdowns, max_entropy


# test with each threshold in the list
def test_suite_B(branchyNet, x_test, y_test, batchsize=10000, ps=np.linspace(0.1, 2.0, 10)):
    accs = []
    diffs = []  # test time list
    num_exits = []
    max_entropies = []
    for p in ps:
        branchyNet.thresholdExits = p  # set threshold
        acc, diff, num_exit, _, max_entropy = test(branchyNet, x_test, y_test, batchsize=batchsize)
        accs.append(acc)
        diffs.append(diff)
        num_exits.append(num_exit)
        max_entropies.append(max_entropy)
    return ps, np.array(accs), np.array(diffs) / float(len(y_test)), num_exits, max_entropies


# get acc, time and number of exiting samples of each exit point with the thresholds
def screen_branchy(branchyNet, x_test, y_test, base_ts, batchsize=1, enumerate_ts=True, verbose=False):
    # generate threshold list for all exit points
    if enumerate_ts:
        ts = generate_thresholds(base_ts, branchyNet.numexits())
    # generate threshold list for the 1st exit point
    else:
        ts = generate_threshold1(base_ts, branchyNet.numexits())

    # test with each threshold in the list
    ts, accs, diffs, exits, max_entropies = test_suite_B(branchyNet, x_test, y_test, batchsize=batchsize, ps=ts)

    return ts, accs, diffs, exits, max_entropies


# generate the list
def generate_thresholds(base_ts, num_layers):
    ts = list(product(*([base_ts] * (num_layers - 1))))
    ts = [list(l) for l in ts]

    return ts


# only generate for the 1st exit point
def generate_threshold1(base_ts, num_layers):
    ts = list()
    for threshold in base_ts:
        ts_exit = list()
        ts_exit.append(threshold)
        for i in range(num_layers - 2):
            ts_exit.append(0)
        ts.append(ts_exit)
    return ts


def get_inc_points(accs, diffs, ts, exits, inc_amt=-0.0005):
    idxs = np.argsort(diffs)
    accs = np.array(accs)
    diffs = np.array(diffs)
    inc_accs = [accs[idxs[0]]]
    inc_rts = [diffs[idxs[0]]]
    inc_ts = [ts[idxs[0]]]
    inc_exits = [exits[idxs[0]]]
    for i, idx in enumerate(idxs[1:]):
        if accs[idx] > inc_accs[-1] + inc_amt:
            inc_accs.append(accs[idx])
            inc_rts.append(diffs[idx])
            inc_ts.append(ts[idx])
            inc_exits.append(exits[idx])

    return inc_accs, inc_rts, inc_ts, inc_exits


# save the test results as csv file
def branchy_save_csv(baseacc, basediff, accs, diffs, exits, ts, filepath='', filename=''):
    print_lst = lambda xs: '{' + ', '.join(map(str, xs)) + '}'
    data = list()
    result_name = ['Network', 'Acc.(%)', 'Time(ms)', 'Gain', 'Thrshld.T', 'Exit(%)']
    data.append(result_name)
    # baseline result
    base_result = [filename, baseacc * 100., basediff, 1.00, '-', '-']
    data.append(base_result)
    # branchynet result
    for i, (acc, diff, exit, t) in enumerate(zip(accs, diffs, exits, ts)):
        branch_result = ['B-' + filename, acc * 100., diff, basediff / diff, print_lst(t),
                         print_lst(round((100. * (e / float(sum(exit)))), 2) for e in exit)]
        data.append(branch_result)

    out_file = open(filepath + filename + '.csv', 'w', newline='')
    csv_write = csv.writer(out_file, dialect='excel')
    for d in data:
        csv_write.writerow(d)
    print("write csv over")
    out_file.close()


# print the results
def branchy_table_results(filepath='', filename=''):
    in_file = open(filepath + filename + '.csv', 'r')
    data = csv.reader(in_file)
    for i, d in enumerate(data):
        if i == 0:
            print("{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}".format(
                d[0], d[1], d[2], d[3], d[4], d[5]))
        elif i == 1:
            print("{:>15}{:>14.2f}{:>14.2f}{:>17.2f}{:>11}{:>15}".format(
                d[0], float(d[1]), float(d[2]), float(d[3]), d[4], d[5]))
        else:
            print("{:>15}{:>14.2f}{:>14.2f}{:>17.2f}{:>15}{:>20}".format(
                d[0], float(d[1]), float(d[2]), float(d[3]), d[4], d[5]))
    in_file.close()
