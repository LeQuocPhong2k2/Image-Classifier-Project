# Programmer: PhongLQ6
# Date: 2024/09/07
# version: 1.0

import argparse

def get_predict_input_args():
    parser = argparse.ArgumentParser()
    
    # path file images
    parser.add_argument('file_image', action="store", help = 'path file images')
    # default = 'flowers/test/5/image_05169.jpg
    
    # file model_checkpoint.pth
    parser.add_argument('file_checkpoint', action="store", help = 'file checkpoint.pth')
    # default = 'save_directory/checkpoint.pth'
    
    # top k
    parser.add_argument('--top_k', type = int, default = 5, help = 'top k')
    
    # file label name
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'path to file containing the flower name.')
    
    parser.add_argument('--gpu', action="store_true", dest="gpu", default=False, help='use GPU for training')
    
    # save directory
    parser.add_argument('--save_dir', type = str, default = 'save_directory', help = 'path to save the model')
    
    # hyperparameters
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'learning rate')
    
    parser.add_argument('--epochs', type = int, default = 5, help = 'number of epochs')
    
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