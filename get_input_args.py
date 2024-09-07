# Programmer: PhongLQ6
# Date: 2024/09/07
# version: 1.0

import argparse

def get_input_args():
    parser = argparse.ArgumentParser()
    
    # data directory
    parser.add_argument('data_dir', action="store",  help = 'path to the folder of images')
    
    # save directory
    parser.add_argument('--save_dir', type = str, default = 'save_directory', help = 'path to save the model')
    
    # hyperparameters
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'learning rate')
    
    parser.add_argument('--epochs', type = int, default = 5, help = 'number of epochs')
    
    parser.add_argument('--gpu', type = bool, default = False, help = 'use GPU for training')
    
    # file label name
    parser.add_argument('--label_name', type = str, default = 'cat_to_name.json', help = 'path to file containing the flower name.')
    
    # top k
    parser.add_argument('--topK', type = int, default = 5, help = 'top k')
    
    # path file images
    parser.add_argument('file_image', action="store", help = 'path file images')
    # default = 'flowers/test/5/image_05169.jpg
    
    # file model_checkpoint.pth
    parser.add_argument('--file_checkpoint', type = str, default = 'save_directory/checkpoint.pth', help = 'file checkpoint.pth')
    
    # CNN Model Architecture
    parser.add_argument('--arch', type = str, default = 'vgg', help = 'CNN Model Architecture resnet or vgg or alexnet')
    
    # Hidden units
    parser.add_argument('--hidden_units', type = int, default = 512, help = 'Number of hidden units in the new classifier')
    
    # Dropout probability
    parser.add_argument('--dropout', type = float, default = 0.5, help = 'Dropout probability')
    
    # print_every
    parser.add_argument('--print_every', type = int, default = 2, help = 'Print every n steps')
    
    in_args = parser.parse_args()
    
    return in_args