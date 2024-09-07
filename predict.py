# Programmer: PhongLQ6
# Date: 2024/09/07
# version: 1.0

from get_predict_input_args import get_predict_input_args

import json
import numpy as np
import ast

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
    in_arg = get_predict_input_args()
    
    # viết code predict.py
    with open(in_arg.category_names, 'r') as imagenet_classes_file:
        cat_to_name = ast.literal_eval(imagenet_classes_file.read())
        
    # Load the checkpoint
    model = load_checkpoint(in_arg.file_checkpoint)
    
    
    # Make a prediction
    probs, classes = predict(in_arg.file_image, model, in_arg.top_k, in_arg.gpu)
    
    # Print the results
    print('Predictions:')
    
    for i in range(len(probs)):
       print(f'{cat_to_name[classes[i]]}: {probs[i]:.4f}')
        
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epochs = checkpoint['epochs']
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    the_means = [0.485, 0.456, 0.406]
    the_standard_deviations = [0.229, 0.224, 0.225]
    # TODO: Process a PIL image for use in a PyTorch model
    img_pil = Image.open(image)
    
    # define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(the_means, the_standard_deviations)
    ])
    
    img_tensor = preprocess(img_pil)
    return img_tensor
    
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
    
    
def predict(image_path, model, topk=5, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # Load the image and process it
    img = process_image(image_path)
    
    # Add batch dimension
    img = img.unsqueeze(0)
    
    # Move the image tensor to the same device as the model
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    img = img.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Turn off gradients for prediction
    with torch.no_grad():
        output = model(img)
    
    # Get the topk probabilities and indices
    probabilities = torch.exp(output)
    top_probabilities, top_indices = probabilities.topk(topk, dim=1)
    
    # Convert to lists
    top_probabilities = top_probabilities.cpu().numpy().tolist()[0]
    top_indices = top_indices.cpu().numpy().tolist()[0]
    
    # Map indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[idx] for idx in top_indices]
    
    return top_probabilities, top_labels 

if __name__ == '__main__':
    main()