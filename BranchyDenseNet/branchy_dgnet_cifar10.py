"""
Using cifar-10 on BranchyDenseNet
"""

from branchynet import utils, visualize
from networks.branchy_dgnet import DGNet
from datasets import cifar10
import dill

# Define Network
branchyNet = DGNet().build()
branchyNet.to_gpu()
branchyNet.training()
branchyNet.verbose = True

# Set parameters
TRAIN_BATCHSIZE = 64
TEST_BATCHSIZE = 64
TRAIN_NUM_EPOCHES = 100
#SAVE_PATH = 'branchy_dgnet_cifar10/'  # save pics
MODEL_NAME = 'branchy_dgnet_cifar10(' + str(100) + ').bn'  # model name
#MODEL_NAME = 'branchy_dgnet_cifar10(' + str(TRAIN_NUM_EPOCHES) + ').bn'  
CSV_NAME = 'branchy_dgnet(' + str(TRAIN_NUM_EPOCHES) + ')'  # csv name

# import cifar10
X_train, Y_train, X_test, Y_test = cifar10.get_data()

print("X_train:{} Y_train:{}".format(X_train.shape, Y_train.shape))
print("X_test: {} Y_test: {}".format(X_test.shape, Y_test.shape))

""""
# load model has been saved
MODEL_NAME = 'branchy_dgnet_main(100).bn'
with open(MODEL_NAME, "rb") as f:
    branchyNet = dill.load(f)
"""

# train on main network
main_loss, main_acc, main_time = utils.train(branchyNet, X_train, Y_train, main=True, batchsize=TRAIN_BATCHSIZE,
                                             num_epoch=TRAIN_NUM_EPOCHES)
print("main_time:", main_time)
# save model
MODEL_NAME_100 = 'branchy_dgnet_main(' + str(100) + ').bn'
with open(MODEL_NAME_100, "wb") as f:
    dill.dump(branchyNet, f)
'''
# visualize the results
visualize.plot_layers(main_loss, save_path=SAVE_PATH,
                      save_name='main_loss(' + str(TRAIN_NUM_EPOCHES) + ')',
                      xlabel='Epoches', ylabel='Training Loss')
visualize.plot_layers(main_acc, save_path=SAVE_PATH,
                      save_name='main_acc(' + str(TRAIN_NUM_EPOCHES) + ')',
                      xlabel='Epoches', ylabel='Training Accuracy')
'''

# train on network with branches
branch_loss, branch_acc, branch_time = utils.train(branchyNet, X_train, Y_train, batchsize=TRAIN_BATCHSIZE,
                                                   num_epoch=TRAIN_NUM_EPOCHES)
print("branch_time:", branch_time)

'''
# visualize the results
visualize.plot_layers(list(zip(*branch_loss)), save_path=SAVE_PATH,
                      save_name='branch_loss(' + str(TRAIN_NUM_EPOCHES) + ')',
                      xlabel='Epoches', ylabel='Training Loss')
visualize.plot_layers(list(zip(*branch_acc)), save_path=SAVE_PATH,
                      save_name='branch_acc(' + str(TRAIN_NUM_EPOCHES) + ')',
                      xlabel='Epoches', ylabel='Training Accuracy')
'''

# save model
with open(MODEL_NAME, "wb") as f:
    dill.dump(branchyNet, f)

# load model
with open(MODEL_NAME, "rb") as f:
    branchyNet = dill.load(f)

# test model，to get the test time and test accuracy of baseline model
branchyNet.testing()
branchyNet.verbose = False
branchyNet.to_gpu()
g_baseacc, g_basediff, _, _, _ = utils.test(branchyNet, X_test, Y_test, main=True, batchsize=TEST_BATCHSIZE)
g_basediff = (g_basediff / float(len(Y_test))) * 1000.

print("g_baseacc:", g_baseacc)
print("g_basediff:", g_basediff)
# set thresholds
thresholds = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.75, 1., 5., 10.]
# get the test results
g_ts, g_accs, g_diffs, g_exits, g_entropies = utils.screen_branchy(branchyNet, X_test, Y_test, thresholds,
                                                                   batchsize=TEST_BATCHSIZE, verbose=True)
g_diffs *= 1000.

'''
# visualize the results
visualize.plot_line_tradeoff(g_accs, g_diffs, g_ts, g_exits, g_baseacc, g_basediff,
                             all_samples=False, inc_amt=0.0001, our_label='Branchy_DGNet',
                             orig_label='branchy_DGNet', xlabel='Runtime(ms)', title='branchy_DGNet Gpu',
                             output_path=SAVE_PATH, output_name='branchy_dgnet_gpu(' + str(TRAIN_NUM_EPOCHES) + ')')
'''
# save the results as csv document
utils.branchy_save_csv(g_baseacc, g_basediff, g_accs, g_diffs, g_exits, g_ts, filename=CSV_NAME)

# print results
print("GPU Results:")
utils.branchy_table_results(filename=CSV_NAME)

