# BranchyDenseNet
To explore the effect of dynamic mechanism on the baseline model, I propose Branchy-DenseNet, an architecture based on DenseNet in which additional side-branch classifiers are added between each DenseBlock. The architecture allows the predictions of some of the test samples to exit the network early through these branches when the samples can already be inferred with high confidence. The idea is ispired by "BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks"https://gitlab.com/kunglab/branchynet
## Structure of BranchyDenseNet
<img width="316" alt="image" src="https://github.com/slenours/EnaDyNN/blob/main/BranchyDenseNet/img/BranchyDenseNet.png">
## Environment
* Google colab GPU
* Python3.7
## Python Dependencies
* chainer-7.8.1
* dill
* 
