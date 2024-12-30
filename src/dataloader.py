import os
from datasets import load_dataset
from torch.utils.data import DataLoader
import multiprocessing  # Import for getting the number of CPU cores
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def get_train_transformers():
    train_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
                                       # Note the difference between (0.1307) and (0.1307,)
                                       ])
    return train_transforms
    
    
def get_test_transformers():
    test_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                       ])
    return test_transforms


def get_tiny_imagenet_dataset(batch_size):
    dataset = load_dataset("Maysee/tiny-imagenet")
    print(f"Dataset keys: {dataset.keys()}")
 
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
 
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
 
    class TinyImageNetDataset:
        def __init__(self, dataset_split, transform):
            self.data = dataset_split
            self.transform = transform
 
        def __len__(self):
            return len(self.data)
 
        def __getitem__(self, idx):
            image = self.data[idx]["image"]
            label = self.data[idx]["label"]
            if self.transform:
                image = self.transform(image)
            return image, label
 
    train_loader = DataLoader(
        TinyImageNetDataset(dataset["train"], transform_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TinyImageNetDataset(dataset["valid"], transform_val), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader
    
def get_MNIST_data_loader():
    train = datasets.MNIST('./data', train=True, download=True, transform=get_train_transformers())
    test = datasets.MNIST('./data', train=False, download=True, transform=get_test_transformers())

    SEED = 1

    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
    # test dataloader
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

    return train_loader, test_loader



def get_data_loaders(data_dir, batch_size=32):
    # Define transformations for training and validation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),  # Randomly crop and resize to 224x224
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),  # Resize images to 256x256
        transforms.CenterCrop(224),  # Center crop to 224x224
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
    ])

    # Determine the number of workers based on CPU cores
    num_workers = multiprocessing.cpu_count() - 4 # Get the number of CPU cores
    print(f"Number of workers for dataloader: {num_workers}")

    # Load the training data using ImageFolder
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'ILSVRC/Data/CLS-LOC/train'), transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Load the validation data using ImageFolder
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'ILSVRC/Data/CLS-LOC/val'), transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader  # Return the data loaders for training and validation