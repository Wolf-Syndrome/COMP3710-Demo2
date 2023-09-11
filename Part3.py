import torch
import torch.nn as nn
from torch.utils.data import Dataset ,DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
from os import listdir
import os
import natsort
from PIL import Image
import matplotlib.pyplot as plt


trainning = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")
    exit()

# Hyper parameters
num_epochs = 35
learning_rate = 0.1
num_classes = 10
model_name = "unet"
path = 'data/keras_png_slices_data/'
batch_size = 128

# Dataloader

"""Dataloader to load up the preprocessed dataset for enumeartion and trainning"""
class OASISDataset(Dataset):

    def __init__(self, XLocation, YLocation, transform) -> None:
        self.XLocation = XLocation
        self.YLocation = YLocation
        self.transform = transform
        XImgs = listdir(XLocation)
        self.XImgs = natsort.natsorted(XImgs)
        YImgs = listdir(YLocation)
        self.YImgs = natsort.natsorted(YImgs)
        if len(self.XImgs) != len(self.YImgs):
            ValueError("Mismatch size of images")

    def __len__(self):
        return len(self.XImgs)
    
    def __getitem__(self, index):
        XImg_loc = os.path.join(self.XLocation, self.XImgs[index])
        YImg_loc = os.path.join(self.YLocation, self.YImgs[index])
        XImg = self.transform(Image.open(XImg_loc))
        YImg = self.transform(Image.open(YImg_loc))
        return (XImg, YImg)



transform = transforms.Compose([
    transforms.ToTensor(),
])

torch.manual_seed(47)
train_dataset = OASISDataset(path + 'keras_png_slices_train', path + 'keras_png_slices_seg_train', transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = OASISDataset(path + 'keras_png_slices_test', path + 'keras_png_slices_seg_test', transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

# Display 5 preview images
def show_samples(total):
    fig, axes = plt.subplots(2, total)
    for X, Y in train_loader:
        for i in range(total):
            img1 = X[i][0, :, :]
            axes[0, i].imshow(X[i][0, :, :], cmap='gray')
            axes[1, i].imshow(Y[i][0, :, :], cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].axis('off')
            axes[0, i].set_title(f"Image {i+1} input")
            axes[1, i].set_title(f"Image {i+1} seg")
        break
    plt.suptitle("Sample of Dataset")
    plt.tight_layout()
    plt.show()


if not trainning:
    show_samples(5)