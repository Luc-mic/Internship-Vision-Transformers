from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from pprint import pprint
import statistics
import time
import timm
from torchvision.transforms.functional import center_crop
from torchvision.transforms.transforms import RandomHorizontalFlip, RandomRotation
from vision_transformer_pytorch import VisionTransformer, model
import torch
from torch import nn
from torch.nn.modules.activation import Softmax
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Lambda, Compose, Resize, AutoAugment, RandomCrop, CenterCrop
from FFVT.models.modeling import FFVT, FFVT_CONFIGS
from TransFG.models.modeling import TransFG, TransFG_CONFIGS

if torch.cuda.is_available() : print("Running on cuda")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

model_name = "FFVT" # ViT-tiny, ViT-small, ViT-base, TransFG or FFVT

pretrained_vit_path = r"ViTs\imagenet21k+imagenet2012_ViT-B_16-224.npz"
data_root = r"data\birds_15"
classes = 150

only_head = True # set to True to let only the classification head learn

batch_size = 20
epochs = 50

learning_rate = 0.02
scheduler_step = 5
gamma = 1

if model_name == "ViT-tiny":
    base_model_outputs = 192
    ViT_base = timm.create_model(f'vit_tiny_patch16_224', pretrained=True)
    ViT_base.head = nn.Identity()
    if only_head:
        for param in ViT_base.parameters():
            param.requires_grad = False
elif model_name == "ViT-small":
    base_model_outputs = 384
    ViT_base = timm.create_model(f'vit_small_patch16_224', pretrained=True)
    ViT_base.head = nn.Identity()
    if only_head:
        for param in ViT_base.parameters():
            param.requires_grad = False
elif model_name == "ViT-base":
    base_model_outputs = 768
    ViT_base = timm.create_model(f'vit_base_patch16_224', pretrained=True)
    ViT_base.head = nn.Identity()
    if only_head:
        for param in ViT_base.parameters():
            param.requires_grad = False
elif model_name == "FFVT":
    base_model_outputs = 768
    config = FFVT_CONFIGS["ViT-B_16"]
    ViT_base = FFVT(config, 224, zero_head=True, num_classes=classes, vis=True, smoothing_value=0, dataset="")
    ViT_base.load_from(np.load(pretrained_vit_path))
    ViT_base.head = nn.Identity()
    if only_head:
        for param in ViT_base.parameters():
            param.requires_grad = False
elif model_name == "TransFG":
    base_model_outputs = 768
    config = TransFG_CONFIGS["ViT-B_16"]
    ViT_base = TransFG(config, 224, zero_head=True, num_classes=classes, smoothing_value=0)
    ViT_base.load_from(np.load(pretrained_vit_path))
    ViT_base.part_head = nn.Identity()
    if only_head:
        for param in ViT_base.parameters():
            param.requires_grad = False


class ClassificationHead(nn.Module):
    def __init__(self, base_model_outputs, classes):
        super(ClassificationHead, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_model_outputs, classes)
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class CustomViT(nn.Module):
    def __init__(self, model_name, ViT_base, classification_head):
        super(CustomViT, self).__init__()
        self.ViT_base = ViT_base
        self.classification_head = classification_head
        self.model_name = model_name
    def forward(self, x):
        if model_name == "FFVT":
            x1 = self.ViT_base(x)[0]
        else:
            x1 = self.ViT_base(x)
        x2 = self.classification_head(x1)
        return x2

# GENERATES THE CLASSIFICATION HEAD

classification_head = ClassificationHead(base_model_outputs, classes)

# GENERATES THE MODEL AND OTHER FUNCTIONS

model = CustomViT(model_name, ViT_base, classification_head).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = ExponentialLR(optimizer, gamma=gamma)

# DEFINES THE TRAINING AND TESTING

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    train_loss, correct = 0, 0
    start_time = time.time()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss, current = loss.item(), batch * len(X)
        # print(f"loss: {loss:>7f}  [{current}/{size}]")
    train_loss /= len(dataloader)
    correct /= size
    epoch_time = time.time() - start_time
    print(f"Train : \n Acc: {(100*correct):>0.1f}%, avg loss: {train_loss:>8f}, time: {epoch_time:>0.3f}s \n")
    return (100*correct, train_loss)

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return (100*correct, test_loss)


    # Download training data from open datasets.

# DATA LOADING

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        image = image.float()
        return image, label

training_data = CustomImageDataset(
    annotations_file=f"{data_root}\\train.csv", 
    img_dir=f"{data_root}\\train\\"
)

test_data = CustomImageDataset(
    annotations_file=f"{data_root}\\test.csv",
    img_dir=f"{data_root}\\test\\")

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# ACTUAL TRAINING

train_accuracy = []
train_loss = []
test_accuracy = []
test_loss = []


epoch_times = []
for t in range(epochs):
    start_time = time.time()
    print(f"Epoch {t+1}, \n-------------------------------")
    for param_group in optimizer.param_groups:
        print(f"lr = {param_group['lr']}")

    res = train(train_dataloader, model, loss_fn, optimizer)
    train_accuracy.append(res[0])
    train_loss.append(res[1])
    
    res = test(test_dataloader, model, loss_fn) 
    test_accuracy.append(res[0])
    test_loss.append(res[1])

    if t % scheduler_step == scheduler_step - 1:
        scheduler.step()
    
    epoch_times.append(time.time() - start_time)
    
print(f"Done! Train time: {sum(epoch_times):>0.3f}s, avg epoch time: {sum(epoch_times)/len(epoch_times):>0.3f}s.")

# RESULTS PLOT

x = np.linspace(1, epochs, epochs)

plt.subplot(1, 2, 1)
plt.plot(x, test_accuracy, label="test_accuracy", color="green")
plt.plot(x, train_accuracy, label="train_accuracy", color="blue")
plt.ylim((-5, 105))
resolution = 20 
for i in range(len(x)//resolution + 1):
    plt.hlines(np.mean(test_accuracy[i * resolution : min((i + 1) * resolution, len(x))]), xmin=x[0], xmax=x[-1], color="black", linestyles="dotted")
    plt.hlines(np.mean(test_accuracy[i * resolution : min((i + 1) * resolution, len(x))]), xmin=i * resolution, xmax=min((i + 1) * resolution, len(x)), color="red")
    print(f"[{i * resolution} - {min((i + 1) * resolution, len(x))}] : {statistics.mean(test_accuracy[i * resolution : min((i + 1) * resolution, len(x))])} +- {statistics.stdev(test_accuracy[i * resolution : min((i + 1) * resolution, len(x))])}")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, test_loss, label="test_loss", color="green")
plt.plot(x, train_loss, label="train_loss", color="blue")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.legend()
plt.show()

