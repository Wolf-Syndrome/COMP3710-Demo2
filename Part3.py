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

#Model
#https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
class UNet(nn.Module):
    def __init__(self, in_planes) -> None:
        super(UNet, self).__init__()
        self.sec1 = self._make_layer(in_planes, 64, 64, True)
        planes = (in_planes) / 2
        self.sec2 = self._make_layer(planes, 128, 128, True)
        planes = (planes) / 2
        self.sec3 = self._make_layer(planes, 256, 256, True)
        planes = (planes) / 2
        self.sec4 = self._make_layer(planes, 512, 512, True)
        planes = (planes) / 2
        self.sec5 = self._make_layer(planes, 1024, 1024, False)
        planes = planes * 2
        self.sec6 = self._make_layer(planes, 512, 512, False)
        planes = planes * 2
        self.sec7 = self._make_layer(planes, 256, 256, False)
        planes = planes * 2
        self.sec8 = self._make_layer(planes, 128, 128, False)
        planes = planes * 2
        self.sec9 = nn.Sequential([
            nn.Conv2d(in_planes, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_planes-2, 64),
            nn.ReLU,
        ])
        self.sec10 = nn.Conv1d(planes, 1)

    def _make_layer(self, in_planes, filter1, filter2, down):
        if down:
            lastLayer = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            lastLayer = nn.ConvTranspose2d(kernel_size=2, stride=2)
        return nn.Sequential([
            nn.Conv2d(in_planes, filter1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_planes-2, filter2),
            nn.ReLU,
            lastLayer
        ])
    
    def forward(self, out):
            out = self.sec1(out)
            out = self.sec2(out)
            out = self.sec3(out)
            out = self.sec4(out)
            out = self.sec5(out)
            out = self.sec6(out)
            out = self.sec7(out)
            out = self.sec8(out)
            out = self.sec9(out)
            out = self.sec10(out)
            return out