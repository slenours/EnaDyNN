# Skip_DGNet

The baseline.py is different from baseline_model.py. It contains the baseline model suitable for image classification tasks.

The baseline_dynamic.py includes our dynamic model. We adjust the baseline model with a gating function to implement layer-skipping. The gating function is a feed-forward gate from the paper “SkipNet: Learning Dynamic Routing in Convolutional Networks” .

The train_base.py is a training process of the baseline model for image classification tasks with cifar-10 dataset. If you want to train the baseline model, you should download train_base.py, baseline.py and imports.py to your computer and run train_base.py directly. Before running the training process, ensure that all packages have been installed to your computer.

The train_Skip_DGNet.py is a training process of our dynamic model Skip_DGNet for image classification tasks with cifar-10 dataset. If you want to train this dynamic model, you should download train_Skip_DGNet.py, baseline_dynamic.py and imports.py to your computer and run train_Skip_DGNet.py directly. Before running the training process, ensure that all packages have been installed to your computer.

These two models can be trained with both  GPU and CPU. And training with GPU is greatly faster than CPU.

To ensure all programs can be run normally, we need a python with a version 3.7.
