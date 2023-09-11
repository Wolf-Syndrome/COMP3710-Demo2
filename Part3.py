import torch
import torch.nn as nn
from torch.utils.data import Dataset ,DataLoader
import torchvision.transforms as transforms
from time import perf_counter
from os import listdir, path as ospath
import natsort
from PIL import Image
import matplotlib.pyplot as plt


trainning = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")
    exit()

# Hyper parameters
num_epochs = 1
learning_rate = 0.1
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
        XImg_loc = ospath.join(self.XLocation, self.XImgs[index])
        YImg_loc = ospath.join(self.YLocation, self.YImgs[index])
        XImg = self.transform(Image.open(XImg_loc))
        YImg = self.transform(Image.open(YImg_loc))
        return (XImg, YImg)



transform = transforms.Compose([
    transforms.ToTensor(),
])

torch.manual_seed(47)
train_dataset = OASISDataset(path + 'keras_png_slices_train', path + 'keras_png_slices_seg_train', transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
total_step = len(train_loader)

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

        self.test1 = nn.Conv2d(in_planes, 64, kernel_size=3, stride=1, padding=1)
        self.test2 = nn.ReLU(inplace=True)
        self.test3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.test4 = nn.ReLU(inplace=True)
        self.test5 = nn.MaxPool2d(kernel_size=2, stride=2)
        

        self.sec1 = self._make_layer(in_planes, 64, 64, True)
        self.sec2 = self._make_layer(64, 128, 128, True)
        self.sec3 = self._make_layer(128, 256, 256, True)
        self.sec4 = self._make_layer(256, 512, 512, True)
        self.sec5 = self._make_layer(512, 1024, 1024, False)
        self.sec6 = self._make_layer(1024, 512, 512, False)
        self.sec7 = self._make_layer(512, 256, 256, False)
        self.sec8 = self._make_layer(256, 128, 128, False)
        self.sec9 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.sec10 = nn.Conv2d(64, 1, kernel_size=1, stride=1)

    def _make_layer(self, in_planes, filter1, filter2, down):
        if down:
            lastLayer = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            lastLayer = nn.ConvTranspose2d(in_channels=in_planes, 
                                           out_channels=in_planes//2, 
                                           kernel_size=2, 
                                           stride=2,
                                           padding=1)
        return nn.Sequential(
            nn.Conv2d(in_planes, filter1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter1, filter2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            lastLayer
        )
    
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

# Initialise model 
model = UNet(1)
model = model.to(device)

#model info
print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
#print(model)

# The SDC criterion 
# Based on https://github.com/pytorch/pytorch/issues/1249
def SDC_loss(input, target):
    smooth = 1. # Used to help overfitting + /0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

criterion = SDC_loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

#--------------
# Train the model
model.train()
print("> Training")
start = perf_counter() #time generation
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): #load a batch
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

end = perf_counter()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

# Test the model
print("> Testing")
start = perf_counter() #time generation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))
end = perf_counter()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

print('END')

