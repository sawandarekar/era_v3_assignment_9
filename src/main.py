import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from model import ResNet50_Model
from train import train
from validate import test
import torch.optim as optim
from torchsummary import summary
from dataloader import get_imagenet_data_loaders
from checkpoint import save_model
import config

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = ResNet50_Model(1000).to(device)
summary(model, input_size=(3, 224, 224))

# train dataloader
# train_loader, test_loader = get_tiny_imagenet_dataset(config.BATCH_SIZE)
train_loader, test_loader = get_imagenet_data_loaders("/mnt/s3bucket/imagenet_dataset/", config.BATCH_SIZE)
# num_classes = len(train_loader.classes)

device =  torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else "cpu"

model =  ResNet50_Model(1000).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=6, gamma=0.1)

criterion = nn.CrossEntropyLoss()

EPOCHS = 5
for epoch in range(EPOCHS):
    train(model, device, train_loader, optimizer, epoch, criterion)
    scheduler.step()
    test(model, device, test_loader)

save_model(model=model, filename="/mnt/s3bucket/imagenet_dataset/resnet50_imagenet.pt")
