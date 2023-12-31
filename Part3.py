import torch
import torch.nn as nn
from torch.utils.data import Dataset ,DataLoader
import torchvision.transforms as transforms
from time import perf_counter
from os import listdir, path as ospath
import natsort
from PIL import Image

trainning = True
if not trainning:
    import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")
    exit()

# Hyper parameters
num_epochs = 1
learning_rate = 0.01
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



aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.9, 1.0), antialias=True),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

transform = transforms.Compose([
    transforms.Resize(size=(128, 128), antialias=True),
    transforms.ToTensor(),
])

torch.manual_seed(47)
train_dataset = OASISDataset(path + 'keras_png_slices_train', path + 'keras_png_slices_seg_train', aug_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
total_step = len(train_loader)

test_dataset = OASISDataset(path + 'keras_png_slices_test', path + 'keras_png_slices_seg_test', transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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

        self.sec1 = self._make_layer(in_planes, 64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sec2 = self._make_layer(64, 128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sec3 = self._make_layer(128, 256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sec4 = self._make_layer(256, 512, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sec5 = self._make_layer(512, 1024, 1024)

        self.upconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=1024//2,
                                kernel_size=2, stride=2)
        self.sec6 = self._make_layer(1024, 512, 512)
        self.upconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=512//2,
                                kernel_size=2, stride=2)
        self.sec7 = self._make_layer(512, 256, 256)
        self.upconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=256//2,
                                kernel_size=2, stride=2)
        self.sec8 = self._make_layer(256, 128, 128)
        self.upconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=128//2,
                                kernel_size=2, stride=2)
        self.sec9 = self._make_layer(128, 64, 64)
        self.sec10 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        self.sec11 = nn.ReLU()

    def _make_layer(self, in_planes, filter1, filter2):
        return nn.Sequential(
            nn.Conv2d(in_planes, filter1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter1, filter2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, out):
        # Encoder
        xs1 = self.sec1(out)
        xs2 = self.sec2(self.pool1(xs1))
        xs3 = self.sec3(self.pool2(xs2))
        xs4 = self.sec4(self.pool3(xs3))
        xs5 = self.sec5(self.pool4(xs4))

        # Decoder
        xu5 = self.upconv1(xs5)
        out = torch.cat([xu5, xs4], dim=1)
        out = self.sec6(out)

        xu6 = self.upconv2(out)
        out = torch.cat([xu6, xs3], dim=1)
        out = self.sec7(out)
        
        out = self.upconv3(out)
        out = torch.cat([out, xs2], dim=1)
        out = self.sec8(out)
        
        out = self.upconv4(out)
        out = torch.cat([out, xs1], dim=1)
        out = self.sec9(out)
        
        out = self.sec11(self.sec10(out))
        return out

# Initialise model 
model = UNet(1)
model = model.to(device)

#model info
print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
#print(model)

# The SDC criterion 
# Based on https://github.com/pytorch/pytorch/issues/1249
class SDC_loss(nn.Module):
    def __init__(self):
        super(SDC_loss, self).__init__()

    def forward(self, inputs, targets):
        smooth = 1. # Used to help overfitting + /0

        iflat = inputs.contiguous().view(-1)
        tflat = targets.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        union = tflat.sum() + iflat.sum()
        
        return 1.0 - ((2. * intersection + smooth) /
                (union + smooth))

criterion = SDC_loss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

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
    total_loss = 0.0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        total_loss += loss.item()
        
    print('Test Accuracy: {} %'.format(100 * (1 - total_loss / len(test_loader))))
end = perf_counter()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

print('END')

