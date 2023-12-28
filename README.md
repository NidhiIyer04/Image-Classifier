```markdown
# Flower Classification Project

This project is being done as a part of the AI Programming with Python Nanodegree by Udacity.
Attached below is the project details with reference to the rubric provided.
This is an ongoing project.
## Project Overview

This project involves training a deep learning model to classify images of flowers. The implementation consists of two main parts: a>

### Part 1 - Development Notebook

#### Submission Files

Ensure that all required files are included in the submission. Model checkpoints are not required for submission.

#### Package Imports

The development notebook should start with a cell that imports all necessary packages and modules.

```python
# Example of package imports
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
# Add other necessary imports
```
#### Training Data Augmentation
```
Utilize torchvision transforms to augment the training data with random scaling, rotations, mirroring, and/or cropping.

```python
# Example of training data augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    # Add other augmentation techniques
])
```

#### Data Normalization
```
Appropriately crop and normalize the training, validation, and testing data.

```python
# Example of data normalization
normalize_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

#### Data Batching
```
Load data for each set (train, validation, test) using torchvision's DataLoader.

```python
# Example of data batching
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# Add loaders for validation and test sets
```

#### Data Loading
```
Load data for each set using torchvision's ImageFolder.

```python
# Example of data loading
train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
# Add datasets for validation and test sets
```

#### Pretrained Network
```
Load a pretrained network (e.g., VGG16) from torchvision.models, and freeze its parameters.

```python
# Example of loading a pretrained network
pretrained_model = models.vgg16(pretrained=True)
for param in pretrained_model.parameters():
    param.requires_grad = False
```

#### Feedforward Classifier
```
Define a new feedforward network for use as a classifier using the features as input.

```python
# Example of defining a feedforward classifier
classifier = torch.nn.Sequential(
    torch.nn.Linear(25088, 4096),
    torch.nn.ReLU(),
    # Add other layers
)
```

#### Training the Network
```
Train the feedforward classifier while keeping the feature network parameters static.

```python
# Example of training the network
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
# Add training loop
```

#### Testing Accuracy
```Measure the network's accuracy on the test data.

```python
# Example of testing accuracy
# Evaluate the model on the test set
```

#### Validation Loss and Accuracy
```
Display validation loss and accuracy during training.

```python
# Example of displaying validation loss and accuracy
# Add validation loop with loss and accuracy calculation
```

#### Loading Checkpoints
```
Implement a function that successfully loads a checkpoint and rebuilds the model.

```python
# Example of loading checkpoints
# Implement a function to load checkpoints
```

#### Saving the Model
```
Save the trained model as a checkpoint with associated hyperparameters and the class_to_idx dictionary.
```python
# Example of saving the model
# Save model checkpoint with hyperparameters and class_to_idx
```

#### Image Processing
```
Implement the `process_image` function that converts a PIL image into an object usable as input to a trained model.

```python
# Example of image processing
# Implement the process_image function
```

#### Class Prediction
```
Implement the `predict` function that takes the path to an image and a checkpoint, then returns the top K most probable classes.

```python
# Example of class prediction
# Implement the predict function
```#### Sanity Checking with Matplotlib
```
Create a matplotlib figure displaying an image and its top 5 most probable classes with actual flower names.

```python
# Example of sanity checking with matplotlib
# Create a matplotlib figure for sanity checking
```

### Part 2 - Command Line Application

#### Submission Files
```
Ensure that all required files are included in the submission. Model checkpoints are not required for submission.

#### Training a Network

Ensure that `train.py` successfully trains a new network on a dataset of images and saves the model to a checkpoint.

```bash
python train.py --data_dir 'path/to/data' --arch 'vgg16' --learning_rate 0.001 --hidden_units 4096 --epochs 10 --gpu
```

#### Training Validation Log
```
Print out the training loss, validation loss, and validation accuracy as the network trains.
```python
# Example of printing training/validation log
# Print out training/validation loss and accuracy during training
```

#### Model Architecture
```
Allow users to choose from at least two different architectures available from torchvision.models.

```bash
python train.py --data_dir 'path/to/data' --arch 'resnet50' --learning_rate 0.001 --hidden_units 4096 --epochs 10 --gpu
```

#### Model Hyperparameters
```
Allow users to set hyperparameters for learning rate, number of hidden units, and training epochs.

```bash
python train.py --data_dir 'path/to/data' --arch 'vgg16' --learning_rate 0.01 --hidden_units 512 --epochs 15 --gpu
```

#### Training with GPU
```
Allow users to choose training the model on a GPU.

```bashpython train.py --data_dir 'path/to/data' --arch 'vgg16' --learning_rate 0.001 --hidden_units 4096 --epochs 10 --gpu
```

#### Predicting Classes
```
Ensure that `predict.py` successfully reads in an image and a checkpoint, then prints the most likely image class and its associated>

```bash
python predict.py --image_path 'path/to/image.jpg' --checkpoint 'path/to/checkpoint.pth'
```

#### Top K Classes
```
Allow users to print out the top K classes along with associated probabilities.

```bash
python predict.py --image_path 'path/to/image.jpg' --checkpoint 'path/to/checkpoint.pth' --top_k 3
```

#### Displaying Class Names
```
Allow users to load a JSON file that maps class values to other category names.

```bash
python predict.py --image_path 'path/to/image.jpg' --checkpoint 'path/to/checkpoint.pth' --category_names 'path/to/cat_to_name.json'
```
#### Predicting with GPU
```
Allow users to use the GPU to calculate predictions.

```bash
python predict.py --image_path 'path/to/image.jpg' --checkpoint 'path/to/checkpoint.pth' --gpu
```
