# utils.py

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
import os

def load_data(data_dir, batch_size):
    # Define your data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {x: ImageFolder(root=os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}

    # Using the image datasets and the transforms, define the dataloaders
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'val']}

    # Size of the training dataset for calculating the validation split
    dataset_size = len(image_datasets['train'])
    indices = list(range(dataset_size))
    split = int(0.2 * dataset_size)

    # Define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(indices[split:])
    val_sampler = SubsetRandomSampler(indices[:split])

    # Define dataloaders with samplers
    train_loader = DataLoader(image_datasets['train'], batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(image_datasets['val'], batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader, image_datasets['train'].classes, image_datasets['train'].class_to_idx

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    model.to(device)
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        train_loss.append(epoch_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()

        epoch_val_loss /= len(val_loader)
        val_loss.append(epoch_val_loss)

        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}')

    return train_loss, val_loss

def save_checkpoint(model, save_dir, arch, hidden_units):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))
