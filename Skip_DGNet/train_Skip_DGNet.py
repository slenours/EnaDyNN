# File : train.py
"""
This is a training process for dynamic model Skip_DGNet.

"""

# Import packages
import torchvision
from torch.utils.tensorboard import SummaryWriter
from Skip_DGNet import *
from torch.utils.data import DataLoader

# Create train record file to record training results
file = open('.\_train_record1_1.txt', 'w', encoding='utf-8')
file.write('Train record 1 : \n')
file.write('\n')
file.close()

# prepare dataset
train_data = torchvision.datasets.CIFAR10(root='.\data', train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root='.\data', train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)
# length of train_data and test_data
train_data_size = len(train_data)
test_data_size = len(test_data)

print("length of train_data：{}".format(train_data_size))
print("length of test_data：{}".format(test_data_size))

# Load dataset
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64)

# call model
model = DGModel()

# Check the availability of GPU. If available, use GPU otherwise use CPU
if torch.cuda.is_available():
    print('GPU is available')
    model = model.cuda()

# loss_function
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# optimizer
# learning_rate = 0.1  # 0.1
# optimizer = torch.optim.SGD(dgmodel_dynamic.parameters(), lr=learning_rate)

# set some parameters. train the DGNet for 300 epochs
total_train_step = 0  # train step
total_test_step = 0  # test step
epoch = 300  # train epoch

# add tensorboard
writer = SummaryWriter('./logs_train')

for i in range(1, epoch + 1):

    # adjust learning rate at 50% and 75% of train epochs
    if i < 150:
        learning_rate = 0.1  # 0.1
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif i < 225:
        learning_rate = 0.01  # 0.01
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        learning_rate = 0.001  # 0.01
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print('--------Train epoch {}--------'.format(i))

    # train process
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()

        outputs = model(imgs)
        outputs = outputs.view(outputs.size(0),
                               outputs.size(1))  # reshape the output size to fit loss_function calculation
        # print('outputs size is: {}'.format(outputs.shape))
        # print('targets size is: {}'.format(outputs.shape))
        loss = loss_fn(outputs, targets)

        # optimize
        optimizer.zero_grad()  # clear previous gardients
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        print('train step：{}，Loss：{}'.format(total_train_step, loss.item()))
        writer.add_scalar('train_loss', loss.item(), total_train_step)

    # test process
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            # print('Size of test imgs : {}'.format(imgs.shape))
            outputs = model(imgs)
            # print('Size of the outputs : {}'.format(outputs.shape))
            # outputs = outputs.view(1,10)
            loss = loss_fn(outputs, targets)  # test loss
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()  # calculate accuracy
            print('The accuracy is : {}'.format(accuracy))
            total_accuracy = total_accuracy + accuracy  # Total_accuracy of test data

    print('total_test_loss：{}'.format(total_test_loss))
    print("test_accuracy: {}".format(total_accuracy / test_data_size))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1
    torch.save(model, 'Skip_DGNet_{}'.format(i))  # Save the model in order to reuse it
    print('model is saved')

    # Set checkpoint
    checkpoint = {
        'epoch': i + 1,  # save train epoch
        'model_state_dict': model.state_dict(),  # save  model parameters
        'optimizer_state_dict': optimizer.state_dict(),  # save optimizer parameters
        'total_train_step': total_train_step,
        'total_test_step': total_test_step,
    }

    # Save the checkpoint.
    # If the training process is interrupted, the checkpoint can continue training process quickly.
    torch.save(checkpoint, 'checkpoint.pth_1.tar')
    print('checkpoint is saved')

    # record training results
    file = open('.\_train_record1_1.txt', 'a', encoding='utf-8')
    file.write('Train epoch {} \n'.format(i))
    file.write('test_loss : {} \n'.format(total_test_loss))
    file.write('test_accuracy : {} \n'.format(total_accuracy / test_data_size))
    file.write('\n')
    file.close()

writer.close()