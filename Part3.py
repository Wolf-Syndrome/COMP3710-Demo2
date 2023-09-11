import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")
    exit()


# Hyper parameters
num_epochs = 35
learning_rate = 0.1
num_classes = 10
model_name = "unet"
path = ''
