import torch
print(torch.has_mps)
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as torch_datasets
import torchvision.transforms as transforms
from torchinfo import summary
from poutyne import Model, Experiment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data
# https://pytorch.org/vision/stable/datasets.html
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 4
trainset = torch_datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
testset = torch_datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# validate image
def get_item(datasets, idx, transform=True):
    img, lb = datasets.__getitem__(idx)
    if transform:
        return ((img / 2 + 0.5).numpy().transpose((1, 2, 0)), lb)
    else:
        return (img, lb)

n = len(trainset.data)
nimg = 9
fig = plt.figure(figsize=(10, 10))
for i in range(nimg):
    idx = np.random.randint(n)
    img, lb = get_item(trainset, idx)
    axs = fig.add_subplot(3, 3, i + 1)
    axs.imshow(img)
    axs.set_title(classes[lb])

# define network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# GPU

net = Net()
device = torch.device("mps")
net.to(device)
summary(net, (30, 3, 32, 32))

# define loss
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# callback
import os
from poutyne import set_seeds, Model, ModelCheckpoint, CSVLogger, Callback, ModelBundle, SKLearnMetrics, plot_history
save_path = "save_model/model"
os.makedirs(save_path, exist_ok=True)
callbacks = [
    # Save the latest weights to be able to continue the optimization at the end for more epochs.
    ModelCheckpoint(os.path.join(save_path, 'last_epoch.ckpt')),

    # Save the weights in a new file when the current model is better than all previous models.
    ModelCheckpoint(os.path.join(save_path, 'best_epoch_{epoch}.ckpt'), monitor='val_loss', mode='min',
                    save_best_only=True, restore_best=True, verbose=True),

    # Save the losses and accuracies for each epoch in a TSV.
    CSVLogger(os.path.join(save_path, 'log.tsv'), separator='\t'),
]

# training
# https://github.com/GRAAL-Research/poutyne
model = Model(net, optimizer, loss_function,
              device=device,
              batch_metrics=['cross_entropy'])
model.fit_generator(trainloader, testloader, epochs=20, callbacks=callbacks)
# test_loss, test_acc = model.evaluate_generator(testloader)
# model = Experiment(".", net, optimizer=optimizer, loss_function=loss_function, batch_metrics=['cross_entropy'])
# model.train(trainloader, testloader, epochs=1)

# # visualization of test
# logs = pd.read_csv("log_10e.tsv", sep="\t")
# plt.plot(logs["loss"], label="train")
# plt.plot(logs["val_loss"], label="test")
# plt.legend()

# # predict
# img, truth = get_item(trainset, 3, transform=False)

# ls_train = list(iter(trainloader))
# ip, lb = ls_train[0]
# ip.shape
# lb.shape
# net(ip)

# img.shape

# # training
# for epoch in range(2):  # loop over the dataset multiple times
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = loss_function(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss = 0.0
# print('Finished Training')



# # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# # https://colab.research.google.com/github/pranjalchaubey/Deep-Learning-Notes/blob/master/PyTorch%20Image%20Classification%20in%202020/Image_Classification_practice.ipynb#scrollTo=ut1uDoXJzCTu