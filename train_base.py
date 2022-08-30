
# File : train_gpu.py
import torchvision
from torch.utils.tensorboard import SummaryWriter
from baseline import *
#from model import *
from torch.utils.data import DataLoader

file = open('.\_train_record3_1.txt', 'w', encoding='utf-8')
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

print("length of train_data ：{}".format(train_data_size))  # ctrl+d 复制
print("length of test_data ：{}".format(test_data_size))

# Load dataset
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64)

# call model
#dgnet = DGNet()
dgmodel = DGModel()
#jiaolong = Jiaolong()

if torch.cuda.is_available():
    print('GPU is available')
    dgmodel = dgmodel.cuda()

"""
#check requires_grad
for name, para in dgmodel.named_parameters():
    print('parameter name is {}, requires_grad is {}'.format(name, para.requires_grad))
"""

# loss function
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()


# optimizer
#learning_rate = 0.1  # 0.1
#optimizer = torch.optim.SGD(dgmodel.parameters(), lr=learning_rate)

# set some parameters
total_train_step = 0  # train step
total_test_step = 0  # test step
epoch = 500  # train epoch

# add tensorboard
writer = SummaryWriter('./logs_train')

for i in range(1, epoch+1):
    if i < 300:
        # optimizer
        learning_rate = 0.1  # 0.1
        optimizer = torch.optim.SGD(dgmodel.parameters(), lr=learning_rate)
    elif i < 400:
        # optimizer
        learning_rate = 0.01
        optimizer = torch.optim.SGD(dgmodel.parameters(), lr=learning_rate)
    else:
        # optimizer
        learning_rate = 0.001
        optimizer = torch.optim.SGD(dgmodel.parameters(), lr=learning_rate)

    print('--------Train epoch {}--------'.format(i))

    # train process
    dgmodel.train()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()

        #print('Input: {}'.format(imgs.shape))
        outputs = dgmodel(imgs)
        outputs = torch.squeeze(outputs)
        #print(outputs.shape)
        #print(targets.shape)
        loss = loss_fn(outputs, targets)
        #print("it is normal")

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        print('train_step：{}，Loss：{}'.format(total_train_step, loss.item()))
        writer.add_scalar('train_loss', loss.item(), total_train_step)

    # test process
    dgmodel.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = dgmodel(imgs)
            #print('Output of dgmodel : {}'.format(outputs))
            print('Output of targets : {}'.format(targets))
            print('argmax of outputs : {}'.format(outputs.argmax(1)))
            #outputs = torch.squeeze(outputs)  #keep 2 dimension in order to calculate loss function.
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()  # calculate accuracy
            print('The accuracy is : {}'.format(accuracy))
            total_accuracy = total_accuracy + accuracy  # Total_accuracy of test data

    print('total_test_loss：{}'.format(total_test_loss))
    print("test_accuracy: {}".format(total_accuracy / test_data_size))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1
    torch.save(dgmodel, 'dgmodel3_{}'.format(i))
    print('model is saved')

    # Set checkpoint
    checkpoint = {
        'epoch': i + 1,  # save train epoch
        'model_state_dict': dgmodel.state_dict(),  # save  model parameters
        'optimizer_state_dict': optimizer.state_dict(),  # save optimizer parameters
        'total_train_step': total_train_step,
        'total_test_step': total_test_step,
    }

    torch.save(checkpoint, 'checkpoint.pth2_1.tar')
    print('checkpoint is saved')

    file = open('.\_train_record3_1.txt', 'a', encoding='utf-8')
    file.write('Train epoch {} \n'.format(i))
    file.write('test_loss : {} \n'.format(total_test_loss))
    file.write('test_accuracy : {} \n'.format(total_accuracy / test_data_size))
    file.write('\n')
    file.close()

writer.close()