import sys
import copy
import os
import time
import argparse
import utils
import json
import torch
from torch import optim, nn
from torchvision import datasets, transforms, models
import torchvision
import torch.nn.functional as F
import numpy as np

def train_and_test_model(model, save_dir, data_loader, learning_rate=0.01, 
                       epochs=2, gpu=False, plot=False):

    # initialize weights and biases
    for layer in iter(model.classifier.children()):
        if len(layer.state_dict()) > 0:
            layer.bias.data.fill_(0)
            layer.weight.data.normal_(std=0.01)

    # freeze vgg's parameters so that we can initialize the last layer's weights
    for param in model.parameters():
        param.requires_grad = False

    # enable training the classifier with the new output
    for param in model.classifier[-1].parameters():
        param.requires_grad_()

    best_acc_epoch = 0
    best_acc = 0.0
    avg_loss_val = 0
    avg_acc_val = 0
    train_losses = []
    train_accuracy = []
    valid_losses = []
    valid_accuracy = []
    best_model_wts = copy.deepcopy(model.state_dict())

    print("TRAINING MODEL with following parameters")
    print("\tsave_dir (for checkpoints):", save_dir)
    print("\tlearning_rate:", learning_rate)
    print("\tepochs:", epochs)
    print("\tgpu:", gpu)

    if gpu and not torch.cuda.is_available():
        print("Sorry, no cuda enabled gpu is available.")
        print("Training on the cpu...")
        gpu = False

    if gpu:
        model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.classifier[-1].parameters(), lr=0.01, momentum=0.9)

    since = time.time()
    for e in range(epochs):
        for phase in ['train', 'valid']:
            correct = 0
            running_loss = 0
            for inputs,labels in data_loader[phase]:
                batch_loss, batch_correct = run_batch(phase, gpu, inputs, labels, 
                                             optimizer_ft, model, criterion)
                running_loss += batch_loss # loss.item()
                # for computing accuracy
                correct += batch_correct # torch.sum(preds == labels.data)
                # free gpu resources
                torch.cuda.empty_cache()

            epoch_acc = float(correct) / float(dataset_sizes[phase])
                
            if phase == "train":
                train_losses.append(running_loss)
                train_accuracy.append(epoch_acc)
            else: # "valid" 
                valid_losses.append(running_loss)
                valid_accuracy.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_acc_epoch = e
                    best_model_wts = copy.deepcopy(model.state_dict())
            
            print("Epoch {}: {}\n\tTotal Loss: {:4f}\n\tAccuracy: {:d}/{:d} = {:4f}\n".format(
                e, phase.upper(), running_loss, correct, dataset_sizes[phase], epoch_acc))                                                         
             
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Best Accuracy: {} in epoch {}".format(round(best_acc, 3), best_acc_epoch))
    
    if plot:
        utils.plot_metrics(train_losses, valid_losses, train_accuracy, valid_accuracy, e) 

    # Test data
    phase = 'test'
    correct = 0
    for inputs,labels in data_loader[phase]:
        batch_loss, batch_correct = run_batch(phase, gpu, inputs, labels, 
                                     optimizer_ft, model, criterion)
        running_loss += batch_loss 
        # for computing accuracy
        correct += batch_correct 
        # free gpu resources
        torch.cuda.empty_cache()

    test_acc = float(correct) / float(dataset_sizes[phase])
    print("{}\nTotal Loss: {:4f}\nAccuracy: {:d}/{:d} = {:4f}".format(phase.upper(), 
        running_loss, correct, dataset_sizes[phase], test_acc))  

    return model, optimizer_ft, e

def run_batch(phase, gpu, inputs, labels, optimizer_ft, model, criterion):
    '''Runs a forward pass through the model and backpropagates
        if we are in the training phase.
       Outputs: the loss and the accuracy for this batch '''
    if phase == "train":
        model.train(True)
    else:
        model.train(False)
        model.eval()
    if gpu:
        # send data to the gpu
        inputs, labels = inputs.cuda(), labels.cuda()
    # reset the parameter's gradients
    optimizer_ft.zero_grad()
    # feed inputs forward
    outputs = model(inputs)
    
    # get the transfer model's predictions
    _, preds = torch.max(outputs.data, 1)
    # compute cross entropy error
    losses = criterion(outputs, labels)
    if phase == "train": 
        # backpropagate
        losses.backward()
        # gradient descend
        optimizer_ft.step()
    
    return losses.item(), torch.sum(preds == labels.data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='training/validation/test data')
    parser.add_argument('--save_dir', action='store', dest='save_dir', 
        default='checkpoint.pth', help='the directory to save the trained model to')
    parser.add_argument('--arch', action='store', dest='arch', 
        default="vgg16_bn", help='the model architecture')
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', 
        default=0.005, help='the learning rate')
    parser.add_argument('--hidden_units', action='store', dest='hidden_units', 
        default=0, help='number of the hidden units prior to output')
    parser.add_argument('--epochs', action='store', dest='epochs', 
        default=2, help='the number of training epochs')
    parser.add_argument('--gpu', action='store', dest='gpu', 
        default=False, help='the model architecture')    
    results = parser.parse_args()
    data_dir = results.data_dir
    save_dir = results.save_dir
    model_arch = results.arch
    learning_rate = float(results.learning_rate)
    hidden_units = int(results.hidden_units)
    epochs = int(results.epochs)
    gpu = results.gpu
    
    # load and transform images, returning label names and pytorch data loaders
    data_loader, dataset_sizes, class_names, class_2_idx = utils.load_and_transform_data(data_dir)

    # get mapping from category label to category name
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # utils.show_image_grid(data_loader, cat_to_name)

    if model_arch == "vgg16_bn":
        since = time.time()
        print("LOADING vgg16_bn CLASSIFIER (might take a few minutes)")
        model = models.vgg16_bn(pretrained=True)
        time_elapsed = time.time() - since
        print('Loading complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print(model.classifier[6].out_features)
    elif model_arch == "alexnet":
        since = time.time()
        print("LOADING alexnet CLASSIFIER (might take a few minutes)")
        model = models.alexnet(pretrained=True)
        time_elapsed = time.time() - since
        print('Loading complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print(model.classifier[6].out_features)
    else: 
        print("IMPLEMENT MORE CLASSIFIERS HERE")
        sys.exit(0)
        
    in_dim = model.classifier[-1].in_features
    out_dim = len(class_names)

    if hidden_units == 0: # just append directly to this classifier
        model.classifier[-1] = nn.Linear(in_features = in_dim, 
                                        out_features = out_dim)
    else: # add the hidden layer and then append our output classifier
        model.classifier[-1] = nn.Linear(in_features = in_dim, 
                                        out_features = hidden_units)
        
        model.add_module("relu", nn.ReLU())
        model.add_module("dropout", nn.Dropout(p=0.5))
        model.add_module("output", nn.Linear(in_features = hidden_units,
                                  out_features = out_dim))

    trained_model, optimzer, epoch = train_and_test_model(model, save_dir, data_loader, learning_rate, 
                                      epochs, gpu, plot=True)

    utils.save(trained_model, optimzer.state_dict(), 
                       epoch, class_2_idx, save_dir)   