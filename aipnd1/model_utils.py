import torch
from torchvision import transforms
from PIL import Image
import json
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
def process_image(image_path):
    # Load and preprocess the image
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = preprocess(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img


def predict(model, image, topk=1, device='cpu'):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    Returns the top K classes and associated probabilities.

    Parameters:
        model (torch.nn.Module): Trained deep learning model
        image (torch.Tensor): Processed input image tensor
        topk (int): Number of top classes to return (default: 1)
        device (str): Device to perform inference on (default: 'cpu')

    Returns:
        top_classes (list): List of top K classes
        top_probabilities (list): List of associated probabilities
    """
    model.eval()
    
    # Move the image tensor to the selected device
    image = image.to(device)
    
    # Perform forward pass
    with torch.no_grad():
        output = model(image.unsqueeze(0))
    
    # Calculate probabilities and top classes using torch.topk
    probabilities, indices = torch.topk(torch.nn.functional.softmax(output, dim=1), topk)
    
    # Convert indices to class labels
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx.item()] for idx in indices[0]]
    
    # Convert probabilities to a list
    top_probabilities = probabilities[0].tolist()
    
    return top_classes, top_probabilities

# Function to load a checkpoint and rebuild the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

    # Check if 'state_dict' key is present in the checkpoint
    if 'state_dict' not in checkpoint:
        raise KeyError("Checkpoint does not contain 'state_dict' information.")

    model = create_model()

    # Load the model's state_dict
    model.load_state_dict(checkpoint['state_dict'])

    # Check if other expected keys are present
    possible_keys = ['class_to_idx', 'idx_to_class']
    for key in possible_keys:
        if key not in checkpoint:
            raise KeyError(f"Checkpoint does not contain the key: {key}")

    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']

    return model
def save_checkpoint(model, arch, hidden_units, learning_rate, epochs, use_gpu):
    checkpoint = {
        'arch': arch,
        'state_dict': model.state_dict(),
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'class_to_idx': model.class_to_idx
    }

    # Additional code for GPU compatibility if needed
    if use_gpu:
        checkpoint['gpu'] = True

    torch.save(checkpoint, 'checkpoint.pth')


def create_model(arch='resnet34', hidden_units=512, class_to_idx=None):
    if arch == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError("Invalid architecture choice. Supported architectures: 'resnet34', 'alexnet'")

    # Customize the classifier
    if arch == 'resnet34':
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    elif arch == 'alexnet':
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )

    return model

def train_model(model, dataloaders, criterion, optimizer, device, epochs=10):
    """
    Train the model.

    Parameters:
    - model: The PyTorch model to be trained.
    - dataloaders: A dictionary containing training and validation dataloaders.
    - criterion: The loss criterion for training.
    - optimizer: The optimizer used for training.
    - device: The device to which the model and data should be moved.
    - epochs: The number of training epochs (default: 10).
    """
    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print training loss
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {running_loss/len(train_loader)}")

        # Validation phase
        model.eval()
        valid_loss = 0.0
        accuracy = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                # Calculate accuracy
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        # Print validation loss and accuracy
        print(f"Validation Loss: {valid_loss/len(valid_loader)}")
        print(f"Validation Accuracy: {accuracy/len(valid_loader)}")

    print("Training complete.")
import torch
from torchvision import datasets, transforms

def load_data(data_dir):
    """
    Load and preprocess the image dataset.

    Parameters:
    - data_dir: The path to the dataset.

    Returns:
    - dataloaders: A dictionary containing training, validation, and test dataloaders.
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and test sets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    dataloaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
    return dataloaders
