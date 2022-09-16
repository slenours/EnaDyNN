
import chainer
from chainer import Link, ChainList
import chainer.functions as F
import inspect
import copy


# define the general structure of network for training or testing
class Net(ChainList):
    def __init__(self, weight=1.):
        super(Net, self).__init__()
        self.weight = weight
        self.starti = 0
        self.endi = 0

    def __call__(self, x, test=False, starti=0, endi=None):
        h = x
        for link in self[starti:endi]:
            h = link(h)
            # if len(inspect.getfullargspec(link.__call__)[0]) == 2:
            #     h = link(h)
            # else:
            #     h = link(h,test)
        self.h = h
        return h

    # train
    def train(self, x, t, starti=0, endi=None):
        h = self(x, False, starti, endi)

        self.accuracy = F.accuracy(h, t)  # 精确度
        self.loss = F.softmax_cross_entropy(h, t)  # 损失函数
        return self.loss

    # test
    def test(self, x, starti=0, endi=None):
        h = self(x, True, starti, endi)
        return h


# define branching structure
class Branch(ChainList):
    def __init__(self, branch, weight=1.):
        super(Branch, self).__init__()
        self.branch = branch
        self.weight = weight
        for link in branch:
            self.add_link(link)

    def cleargrads(self):
        super(SL, self).cleargrads()
        for link in self.branch:
            link.cleargrads()

    def __deepcopy__(self, memo):
        newbranches = []
        for link in self.branch:
            newbranches.append(copy.deepcopy(link, memo))
        new = type(self)(newbranches, self.weight)
        return new

    def __call__(self, x, test=False, starti=0, endi=None):
        h = x
        for link in self[starti:endi]:
            if len(inspect.getfullargspec(link.__call__)[0]) == 2:
                h = link(h)
            else:
                h = link(h, test)
        return h


# operation of the network layer command function
class FL(Link):
    def __init__(self, fn, *arguments, **keywords):
        super(FL, self).__init__()
        self.fn = fn
        self.arguments = arguments
        self.keywords = keywords

    def __call__(self, x, test=False):
        return self.fn(x, *self.arguments, **self.keywords)


# selection of training mode or testing mode
class SL(Link):
    def __init__(self, fnTrain, fnTest=None):
        super(SL, self).__init__()
        self.fnTrain = fnTrain
        self.fnTest = fnTest

    def cleargrads(self):
        super(SL, self).cleargrads()
        if self.fnTrain is not None:
            self.fnTrain.cleargrads()
        if self.fnTest is not None:
            self.fnTest.cleargrads()

    # deepcopy
    def __deepcopy__(self, memo):
        fnTrain = copy.deepcopy(self.fnTrain, memo)
        fnTest = copy.deepcopy(self.fnTest, memo)
        new = type(self)(fnTrain, fnTest)
        return new

    def to_gpu(self):
        if self.fnTrain is not None:
            self.fnTrain.to_gpu()
        if self.fnTest is not None:
            self.fnTest.to_gpu()

    def to_cpu(self):
        if self.fnTrain is not None:
            self.fnTrain.to_cpu()
        if self.fnTest is not None:
            self.fnTest.to_cpu()

    def __call__(self, x, test=False):
        if not test:
            return self.fnTrain(x, test)
        else:
            if self.fnTest is None:
                return x
            return self.fnTest(x, test)
        return x
