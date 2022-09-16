# Skip_DGNet

The DGNet.py contains the baseline model for image classification task.

The Skip_DGNet.py includes the dynamic model Skip_DGNet. We adjust the baseline model with a gating function to implement layer-skipping. The gating function is a feed-forward gate from the paper “SkipNet: Learning Dynamic Routing in Convolutional Networks” .

The train_DGNet.py is a training process of the baseline model for image classification tasks with cifar-10 dataset. If you want to train the baseline model, you should download train_DGNet.py, DGNet.py and imports.py to your computer and run train_base.py directly. Before running the training process, ensure that all packages have been installed to your computer. This model can be trained with both GPU and CPU. The train process file can be run in PyCharm or Google colab.

The train_Skip_DGNet.py is a training process of our dynamic model Skip_DGNet for image classification tasks with cifar-10 dataset. If you want to train this dynamic model, you should download train_Skip_DGNet.py, Skip_DGNet.py and imports.py to your computer and run train_Skip_DGNet.py directly. Before running the training process, ensure that all packages have been installed to your computer.This model can be trained with both GPU and CPU. The train process file can be run in PyCharm or Google colab.

The prediction_base.py is the prediction file for the baseline model DGNet. You can predict the results with trained model through running this file directly. Like the training file, you should download prediction_base.py, DGNet.py and imports.py to your computer and run prediction_base.py directly. Before running the prediction process, prepare your trained model and ensure that all packages have been installed to your computer.

The prediction.py is the prediction file for the dynamic model Skip_DGNet. You can predict the results with trained model through running this file directly. Like the training file, you should download prediction_base.py, DGNet.py and imports.py to your computer and run prediction_base.py directly. Before running the prediction process, prepare your trained model and ensure that all packages have been installed to your computer.

The image0 to image9 the images for the prediction process. You can download them for the purpose of perdiction or you can pick your own images if you want.

The Skip_DGNet_Noactivation_coomment_code.py includes the dynamic model Skip_DGNet having no activations of gating functions through commenting out certain part of code. The Skip_DGNet_Noactivation_execute_all_layers.py includes the dynamic model Skip_DGNet having no activations of gating functions through executing all layers. We use these two special models, DGNet model and Skip_DGNet to make a small experiment to support the results shown in the report DyNN.

The experiment_DGNet.py is the experiment training file for training DGNet for 20 epochs.

The experiment_Skip_DGNet.py is the experiment training file for training Skip_DGNet for 20 epochs.

The Skip_DGNet_Noactivation_coomment_code.py s the experiment training file for training Skip_DGNet having no activations of gating functions through commenting out certain part of code for 20 epochs.

The Skip_DGNet_Noactivation_coomment_code.py s the experiment training file for training Skip_DGNet having no activations of gating functions through executing all layers for 20 epochs.

To ensure all programs can be run normally, we need a python with a version 3.7.
