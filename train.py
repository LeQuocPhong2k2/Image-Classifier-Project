# Programmer: PhongLQ6
# Date: 2024/09/07
# version: 1.0

from get_input_args import get_input_args

import json
import numpy as np
import ast
import os

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision

from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from collections import OrderedDict
from PIL import Image
from torch import __version__

def main():
    
    in_arg = get_input_args()
    
    # Load the data
    data_dir = in_arg.data_dir
    
    save_dir = in_arg.save_dir
    
    train_dir = data_dir + '/train'
    
    valid_dir = data_dir + '/valid'
    
    test_dir = data_dir + '/test'
    
    # Define your transforms for the training, validation, and testing sets
    the_means = [0.485, 0.456, 0.406]
    the_standard_deviations = [0.229, 0.224, 0.225]
    max_images_size = 224
    
    data_transforms = {
        "training": transforms.Compose([
                                        transforms.RandomRotation(30),
                                        transforms.RandomHorizontalFlip(p=0.25),
                                        transforms.RandomGrayscale(p=0.02),
                                        transforms.RandomResizedCrop(max_images_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(the_means, the_standard_deviations)]),

        "validation": transforms.Compose([transforms.Resize(max_images_size + 1),
                                          transforms.CenterCrop(max_images_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(the_means, the_standard_deviations)]),

        "testing": transforms.Compose([transforms.Resize(max_images_size + 1),
                                       transforms.CenterCrop(max_images_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize(the_means, the_standard_deviations)])
    }
    
    # Load the datasets with ImageFolder
    image_datasets = {
        "training": datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        "validation": datasets.ImageFolder(valid_dir, transform=data_transforms["validation"]),
        "testing": datasets.ImageFolder(test_dir, transform=data_transforms["testing"])
    }
    
    # Using the image datasets and the transforms, define the dataloaders
    dataloaders = {
        "training": torch.utils.data.DataLoader(image_datasets["training"], batch_size=32, shuffle=True),
        "validation": torch.utils.data.DataLoader(image_datasets["validation"], batch_size=32),
        "testing": torch.utils.data.DataLoader(image_datasets["testing"], batch_size=32)
    }
    
    # Load the label names
    with open(in_arg.label_name, 'r') as imagenet_classes_file:
        cat_to_name = json.load(imagenet_classes_file)
    
    # Select the model architecture
    
    if in_arg.arch.startswith('vgg'):
        model = models.vgg16(pretrained=True)
        input_features = 25088
    elif in_arg.arch.startswith('resnet'):
        model = models.resnet50(pretrained=True)
        input_features = 512
    elif in_arg.arch.startswith('alexnet'):
        model = models.alexnet(pretrained=True)
        input_features = 9216
    else:
        raise ValueError("Unsupported architecture. Choose from 'vgg', 'resnet', or 'alexnet'.")
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Define a new, untrained feed-forward network as a classifier
    classifier = nn.Sequential(
        nn.Linear(input_features, in_arg.hidden_units),
        nn.ReLU(),
        nn.Dropout(in_arg.dropout),
        nn.Linear(in_arg.hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
     # Replace the pre-trained network's classifier with our own
    if in_arg.arch.startswith('resnet'):
        model.fc = classifier
    else:
        model.classifier = classifier
        
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters() if not in_arg.arch.startswith('resnet') else model.fc.parameters(), lr=in_arg.learning_rate)
    
    # Define the loss function
    criteria = nn.NLLLoss()
    
    # Move the model to the GPU if available
    device = torch.device("cuda" if in_arg.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training the network
    epochs = in_arg.epochs
    steps = 0
    print_every = in_arg.print_every
    
    print("\n\n*** Results Summary for CNN Model Architecture",in_arg.arch.upper(),"***")
    for epoch in range(epochs):
        running_loss = 0
        running_accuracy = 0
        batch_count = 0
        
        print("\n----------------------------")
        print(f"Epoch {epoch + 1} of {epochs}")

        for images, labels in dataloaders['training']:
            steps += 1
            batch_count += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(images)
            loss = criteria(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            if steps % print_every == 0:
                avg_loss = running_loss / print_every
                avg_accuracy = running_accuracy / print_every * 100
                print(f"Batches {steps - print_every:03d} to {steps:03d}: avg. loss: {avg_loss:.4f}, accuracy: {avg_accuracy:.2f}%.")
                running_loss = 0
                running_accuracy = 0

        # Validation phase
        validation_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for images, labels in dataloaders['validation']:
                images, labels = images.to(device), labels.to(device)
                logps = model.forward(images)
                batch_loss = criteria(logps, labels)

                validation_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"\tValidating for epoch {epoch + 1}...")
        print(f"\tAccurately classified {accuracy / len(dataloaders['validation']) * 100:.0f}% of {len(dataloaders['validation']) * dataloaders['validation'].batch_size} images.")
        model.train()
    
    # Save the checkpoint
    model.class_to_idx = image_datasets['training'].class_to_idx
    checkpoint = {
        'model': model,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs
    }
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    torch.save(checkpoint, in_arg.save_dir + '/checkpoint.pth')
    
    print("process completed...")
    print("*** CNN Model Architecture",in_arg.arch.upper(),"has been trained and saved to",in_arg.save_dir + '/checkpoint.pth',"***")

if __name__ == '__main__':
    main()