''''
COMP3710 D2 Eigenfaces Lachlan Bunt
Not all original code
'''
import torch
import torchvision.transforms as transforms
from time import perf_counter

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np

from torch.nn import Conv2d, Module, ReLU, MaxPool2d, Linear, CrossEntropyLoss, BatchNorm2d
from torch.utils.data import TensorDataset, DataLoader

'''Check Cuda'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning couldn't find cuda using cpu.")

'''Hyper-parameters'''
num_epochs = 5
learning_rate = 5e-3
channels = 1

'''Data loading + preprocessing'''
start_time = perf_counter()
# Download the data, if not already on disk and load it as numpy arrays
#print(get_data_home())
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.images
y = lfw_people.target
print("X_min:",X.min(),"X_train_max:", X.max())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train = torch.from_numpy(X_train[:, np.newaxis, :, :])
X_test = torch.from_numpy(X_test[:, np.newaxis, :, :])
y_test = torch.from_numpy(y_test)
y_train = torch.from_numpy(y_train)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=128)
test_dataloader = DataLoader(test_dataset, batch_size=128)

# Extra details
n_classes = lfw_people.target_names.shape[0]
total_step = len(X_train)

''''Model'''
#CNN using pytorch
class CNNModel(Module):
    def __init__(self, numChannels, num_classes):
        super(CNNModel, self).__init__()

        # Conv 1
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=(3, 3), padding=1)
        self.bn1 = BatchNorm2d(32)
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Conv 2
        self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
        self.bn2 = BatchNorm2d(32)
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Fully connected
        self.fc1 = Linear(in_features=3456, out_features=128)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        out = self.maxpool1(self.bn1(self.conv1(x)))
        out = self.maxpool2(self.bn2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        return out


model = CNNModel(1, n_classes)
model = model.to(device)

#model info
print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
print(model)

criterion = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



#--------------
# Train the model
model.train()
print("> Training")
start = perf_counter() #time generation
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader): #load a batch
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
    for images, labels in test_dataloader:
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
