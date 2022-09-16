"""
build BranchyNet structure
"""

from branchynet.function import *
from branchynet.links import *
from chainer import Variable, optimizers
from scipy.stats import entropy
import cupy
import numpy as np
import time


class BranchyNet:
    def __init__(self, network, thresholdExits=None, percentTestExits=.9, percentTrianKeeps=1., learning_rate=0.1,
                 momentum=0.9,
                 weight_decay=0.0001, alpha=0.001, opt="Adam", joint=True, verbose=False):

        self.opt = opt  # optimizer
        self.alpha = alpha  # step size of optimizer Adam
        self.weight_decay = weight_decay  # weight decay
        self.momentum = momentum  # momentum
        self.learning_rate = learning_rate  # learning rate
        self.joint = joint
        self.forwardMain = None  # main network forward

        self.main = Net()  # main network/baseline
        self.models = []  # network with branches, a list
        starti = 0
        curri = 0  # pointer to the current network layer
        for link in network:
            # if this layer is not a Branch, than add it into the main network
            if not isinstance(link, Branch):
                curri += 1
                self.main.add_link(link)
            else:
                # if it is a Branch, than build a branching structure
                net = Net(link.weight)
                net.starti = starti
                starti = curri
                net.endi = curri
                # add main network before the branch to the whole network
                for prevlink in self.main:
                    newlink = copy.deepcopy(prevlink)
                    newlink.name = None
                    net.add_link(newlink)
                # add branch to the whole network
                for branchlink in link:
                    newlink = copy.deepcopy(branchlink)
                    newlink.name = None
                    net.add_link(newlink)
                self.models.append(net)
        # add the last branch to the main network
        for branchlink in link:
            newlink = copy.deepcopy(branchlink)
            newlink.name = None
            self.main.add_link(newlink)

        # set parameters of the optimizer
        if self.opt == 'MomentumSGD':
            self.optimizer = optimizers.MomentumSGD(learning_rate=self.learning_rate, momentum=self.momentum)
        else:
            self.optimizer = optimizers.Adam(alpha=self.alpha)

        # start the optimizer in main network
        self.optimizer.setup(self.main)

        if self.opt == 'MomentumSGD':
            self.optimizer.add_hook(chainer.optimizer.WeightDecay(self.weight_decay))

        self.optimizers = []

        # start the optimizer in branches
        for model in self.models:
            if self.opt == 'MomentumSGD':
                optimizer = optimizers.MomentumSGD(learning_rate=self.learning_rate, momentum=0.9)
            else:
                optimizer = optimizers.Adam()
            optimizer.setup(model)

            if self.opt == 'MomentumSGD':
                optimizer.add_hook(chainer.optimizer.WeightDecay(self.weight_decay))

            self.optimizers.append(optimizer)

        self.percentTrainKeeps = percentTrianKeeps
        self.percentTestExits = percentTestExits  # percentage of test samples at exit point
        self.thresholdExits = thresholdExits  # threshold of exit point
        self.clearLearnedExitsThresholds()

        self.verbose = verbose
        self.gpu = False
        self.xp = np

    # get threshold
    def getLearnedExitsThresholds(self):
        return self.learnedExitsThresholds / self.learnedExitsThresholdsCount

    # set threshold as 0
    def clearLearnedExitsThresholds(self):
        self.learnedExitsThresholds = np.zeros(len(self.models))
        self.learnedExitsThresholdsCount = np.zeros(len(self.models))

    # get the number of exits
    def numexits(self):
        return len(self.models)

    # train
    def training(self):
        # train main structure
        for link in self.main:
            link.train = True
        # train network with branches
        for model in self.models:
            for link in model:
                link.train = True

    # test
    def testing(self):
        # test main structure
        for link in self.main:
            link.train = False
        # test network with branches
        for model in self.models:
            for link in model:
                link.train = False


    # train on gpu
    def to_gpu(self):
        self.xp = cupy
        self.gpu = True
        self.main.to_gpu()
        for model in self.models:
            model.to_gpu()

    # train on cpu
    def to_cpu(self):
        self.xp = np
        self.gpu = False
        self.main.to_cpu()
        for model in self.models:
            model.to_cpu()

    # copy the layers of main network
    def copy_main(self):
        self.main_copy = copy.deepcopy(self.main)
        return

    # copy main network's train function
    def train_main_copy(self, x, t=None):
        return self.train_model(self.main_copy, x, t)

    # copy main network's test function
    def test_main_copy(self, x, t=None):
        return self.test_model(self.main_copy, x, t)

    # train function of main network
    def train_main(self, x, t=None):
        return self.train_model(self.main, x, t)

    # train function of each branch
    def train_branch(self, i, x, t=None):
        return self.train_model(self.models[i], x, t)

    # test function of main network
    def test_main(self, x, t=None):
        return self.test_model(self.main, x, t)

    # train the main network
    def train_model(self, model, x, t=None):
        self.main.cleargrads()  # reset the gradient
        loss = self.main.train(x, t)  # get loss function
        accuracy = self.main.accuracy  # get accuracy
        loss.backward()
        self.optimizer.update()  # update parameters using optimizer
        if self.gpu:
            lossesdata = loss.data.get()
            accuraciesdata = accuracy.data
        else:
            lossesdata = loss.data
            accuraciesdata = accuracy.data

        # print loss and acc
        if self.verbose:
            print("losses: {}, accuracies: {}".format(lossesdata, accuraciesdata))

        return lossesdata, accuraciesdata

    # test main network
    def test_model(self, model, x, t=None):
        totaltime = 0
        start_time = time.time()  # starting time of test
        h = self.main.test(x)  # start test in main
        end_time = time.time()  # ending time of test
        totaltime += end_time - start_time

        accuracy = F.accuracy(h, t)
        if self.gpu:
            accuracydata = accuracy.data.get()
        else:
            accuracydata = accuracy.data

        if self.verbose:
            print("accuracies", accuracydata)

        return accuracydata, totaltime

    # train network with branches
    def train(self, x, t=None):
        # Parameter sharing of the common layers between the model with the branches and the main network
        for i, link in enumerate(self.main):
            for model in self.models:
                for j, modellink in enumerate(model[:model.endi]):
                    if i == j:
                        modellink.copyparams(link)
                        break

        # reset parameters
        self.main.cleargrads()
        [model.cleargrads() for model in self.models]

        if self.forwardMain is not None:
            mainLoss = self.main.train(x, t)

        # remaining training samples
        remainingXVar = x
        remainingTVar = t

        numexits = []
        losses = []
        accuracies = []
        nummodels = len(self.models)
        numsamples = x.data.shape[0]

        for i, model in enumerate(self.models):

            if type(remainingXVar) == None or type(remainingTVar) == None:
                break

            loss = model.train(remainingXVar, remainingTVar)
            losses.append(loss)
            accuracies.append(model.accuracy)

            if i == nummodels - 1:
                break

            # get entropy of samples at the exit points
            softmax = F.softmax(model.h)
            if self.gpu:
                entropy_value = entropy_gpu(softmax).get()
            else:
                entropy_value = np.array([entropy(s) for s in softmax.data])
            total = entropy_value.shape[0]
            idx = np.zeros(total, dtype=bool)  # to determine which samples are exited early, and True means exit

            if self.thresholdExits is not None:
                min_ent = 0
                # if entropy < threshold，than set idx as True，and the sample can be exited
                if isinstance(self.thresholdExits, list):
                    idx[entropy_value < min_ent + self.thresholdExits[i]] = True
                    numexit = sum(idx)
                else:
                    idx[entropy_value < min_ent + self.thresholdExits] = True
                    numexit = sum(idx)

            elif hasattr(self, 'percentTrainExits') and self.percentTrainExits is not None:
                if isinstance(self.percentTestExits, list):
                    numexit = int(self.percentTrainExits[i] * numsamples)
                else:
                    numexit = int(self.percentTrainExits * total)
                esorted = entropy_value.argsort()
                idx[esorted[:numexit]] = True

            else:
                if isinstance(self.percentTrainKeeps, list):
                    numkeep = (self.percentTrainKeeps[i] * numsamples)
                else:
                    numkeep = self.percentTrainKeeps * total
                numexit = int(total - numkeep)
                esorted = entropy_value.argsort()
                idx[esorted[:numexit]] = True

            numkeep = int(total - numexit)
            numexits.append(numexit)

            if self.gpu:
                xdata = remainingXVar.data.get()
                tdata = remainingTVar.data.get()
            else:
                xdata = remainingXVar.data
                tdata = remainingTVar.data

            if numkeep > 0:
                with chainer.no_backprop_mode():

                    remainingXVar = Variable(self.xp.array(xdata[~idx]))
                    remainingTVar = Variable(self.xp.array(tdata[~idx]))

            else:
                remainingXVar = None
                remainingTVar = None

        if self.forwardMain is not None:
            mainLoss.backward()

        for i, loss in enumerate(losses):
            net = self.models[i]
            loss = net.weight * loss
            loss.backward()

        # add the gradient of branch to main network
        if self.joint:
            if self.forwardMain is not None:
                models = self.models[:-1]
            else:
                models = self.models
            for i, link in enumerate(self.main):
                for model in models:
                    for j, modellink in enumerate(model[:model.endi]):
                        if i == j:
                            link.addgrads(modellink)
        else:
            for i, link in enumerate(self.main):
                for model in self.models[-1:]:
                    for j, modellink in enumerate(model[:model.endi]):
                        if i == j:
                            link.addgrads(modellink)
        self.optimizer.update()
        [optimizer.update() for optimizer in self.optimizers]

        # share parameters again
        for i, link in enumerate(self.main):
            for model in self.models:
                for j, modellink in enumerate(model[:model.endi]):
                    if i == j:
                        modellink.copyparams(link)

        if self.gpu:
            lossesdata = [loss.data.get() for loss in losses]
            accuraciesdata = [accuracy.data.get() for accuracy in accuracies]
        else:
            lossesdata = [loss.data for loss in losses]
            accuraciesdata = [accuracy.data for accuracy in accuracies]

        if self.verbose:
            print("numexits:{},losses:{},accuracies:{}".format(numexits, losses, accuracies))

        return lossesdata, accuraciesdata

    # test model with branches
    def test(self, x, t=None):
        numexits = []
        accuracies = []
        remainingXVar = x
        remainingTVar = t
        nummodels = len(self.models)
        numsamples = x.data.shape[0]
        totaltime = 0
        max_entropy = 0

        for i, model in enumerate(self.models):

            if remainingXVar is None or remainingTVar is None:
                numexits.append(0)
                accuracies.append(0)
                continue

            # test time record
            start_time = time.time()
            h = model.test(remainingXVar, model.starti, model.endi)
            end_time = time.time()
            totaltime += end_time - start_time

            # calculate the entropy
            smh = model.test(h, model.endi)
            softmax = F.softmax(smh)
            if self.gpu:
                entropy_value = entropy_gpu(softmax).get()
            else:
                entropy_value = np.array([entropy(s) for s in softmax.data])

            # compare entropy with threshold
            idx = np.zeros(entropy_value.shape[0], dtype=bool)
            if i == nummodels - 1:
                idx = np.ones(entropy_value.shape[0], dtype=bool)
                numexit = sum(idx)
            else:
                if self.thresholdExits is not None:
                    min_ent = 0
                    if isinstance(self.thresholdExits, list):
                        idx[entropy_value < min_ent + self.thresholdExits[i]] = True
                        numexit = sum(idx)
                    else:
                        idx[entropy_value < min_ent + self.thresholdExits] = True
                        numexit = sum(idx)
                else:
                    if isinstance(self.percentTestExits, list):
                        numexit = int((self.percentTestExits[i]) * numsamples)
                    else:
                        numexit = int(self.percentTestExits * entropy_value.shape[0])
                    esorted = entropy_value.argsort()
                    idx[esorted[:numexit]] = True

            total = entropy_value.shape[0]
            numkeep = total - numexit
            numexits.append(numexit)

            if i == 0:
                for j, value in enumerate(entropy_value):
                    if idx[j]:
                        if value > max_entropy:
                            max_entropy = value

            if self.gpu:
                xdata = h.data.get()
                tdata = remainingTVar.data.get()
            else:
                xdata = h.data
                tdata = remainingTVar.data

            if numkeep > 0:
                xdata_keep = xdata[~idx]
                tdata_keep = tdata[~idx]

                remainingXVar = Variable(self.xp.array(xdata_keep, dtype=x.data.dtype))
                remainingTVar = Variable(self.xp.array(tdata_keep, dtype=t.data.dtype))

            else:
                remainingXVar = None
                remainingTVar = None

            # get the samples that have been exited
            if numexit > 0:
                xdata_exit = xdata[idx]
                tdata_exit = tdata[idx]

                exitXVar = Variable(self.xp.array(xdata_exit, dtype=x.data.dtype))
                exitTVar = Variable(self.xp.array(tdata_exit, dtype=t.data.dtype))

                with chainer.no_backprop_mode():
                    exitH = model.test(exitXVar, model.endi)
                    accuracy = F.accuracy(exitH, exitTVar)

                    if self.gpu:
                        accuracies.append(accuracy.data.get())
                    else:
                        accuracies.append(accuracy.data)
            else:
                accuracies.append(0.)
        # get the overall accuracy
        overall = 0
        for i, accuracy in enumerate(accuracies):
            overall += accuracy * numexits[i]
        overall /= np.sum(numexits)

        if self.verbose:
            print("numexits", numexits)
            print("accuracies", accuracies)
            print("overall accuracy", overall)

        return overall, accuracies, numexits, totaltime, max_entropy

    # print the model
    def print_models(self):
        for model in self.models:
            print("----", model.starti, model.endi)
            for link in model:
                print(link)
        print("----", self.main.starti, model.endi)
        for link in self.main:
            print(link)
        print("----")
