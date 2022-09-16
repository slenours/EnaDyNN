"""
This is a prediction file for baseline model. We will reuse the trained model for the purpose of prediction
"""
# Import packages
import torch
import torchvision
import time
from DGNet import *
from PIL import Image
from torch import nn

total_time = 0  #Initialize the total inference time

# This is a class dictionary of cifar-10
class_dict = {0: 'airplane',
              1: 'automobile',
              2: 'bird',
              3: 'cat',
              4: 'deer',
              5: 'dog',
              6: 'frog',
              7: 'horse',
              8: 'ship',
              9: 'truck',
              }

# prediction process
for i in range(10):
    image_path = "./image{}.png".format(i)  #load image
    image = Image.open(image_path)
    #print(image)

    # We need to transform the image to size 32x32x3 to fit the image size of  dataset cifar-10
    image = image.convert('RGB')
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                torchvision.transforms.ToTensor()])

    image = transform(image)
    #print(image.shape)


    #load model
    # If the model is trained with GPU but we want to use CPU to predict, we need add a map_location.
    model = torch.load("dgmodel2_254", map_location=torch.device('cpu'))

    # calculate the number of parameters which need gradients
    num = 0
    for param in model.parameters():
        print("The parameter is : {}".format(param))
        if param.requires_grad:
            num += param.numel()
    print("param is %.2fM" % (num / 1e6))



    #print(model)
    image = torch.reshape(image, (1, 3, 32, 32))
    start = time.time()
    model.eval()
    with torch.no_grad():
        output = model(image)
    output = output.view(1, 10)
    print(output)
    #print('The size of output is : {}'.format(output.shape))
    end = time.time()
    inference_time = (end - start)*1000  #inference time for one image
    total_time = total_time + inference_time
    print('inference time is : {} ms'.format(inference_time))
    label = int(output.argmax(1))  # predicted label
    #print('label is : {}'.format(label))
    #print(class_dict.get(label))
    print('This is a/an {} \n'.format(class_dict.get(label)))

average_time = total_time / 10  #average inference time
print("average_time is : {} ms".format(average_time))
