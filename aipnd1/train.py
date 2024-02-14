import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from model_utils import create_model, save_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Train a deep learning model.")
    parser.add_argument('data_dir', help='Path to the dataset')
    parser.add_argument('--arch', type=str, default='resnet34', help='Model architecture (default: resnet34)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')

    return parser.parse_args()

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    # Define your data transforms for training and validation
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    batch_size = 32    
    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    # Using DataLoader for batching and shuffling
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
    
    return {'train': trainloader, 'valid': validloader}

def train_model(model, criterion, optimizer, dataloaders, epochs, device):
    # Training loop
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloaders[phase])
            print(f'Epoch {epoch + 1}/{epochs} | {phase} Loss: {epoch_loss:.4f}')

    print('Training complete.')

def main():
    args = parse_args()
    
    # Use args.arch, args.learning_rate, args.hidden_units, args.epochs in your training loop
    
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = create_model(arch=args.arch, hidden_units=args.hidden_units)
    model.to(device)

    dataloaders = load_data(args.data_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_model(model, criterion, optimizer, dataloaders, args.epochs, device)

    # Save the trained model checkpoint
    # After training, save the model checkpoint
    # After training, save the model checkpoint
    save_checkpoint(model, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.gpu)

if __name__ == "__main__":
    main()

