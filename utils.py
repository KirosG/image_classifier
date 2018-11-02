import os
import sys
import torch
import torchvision
from torchvision import datasets, transforms, models
from scipy.io import loadmat
from PIL import Image
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_transform_data(data_dir, batch_sz=16):
    '''Inputs:
        Takes the data directory, applies a transform and 
       Outputs: 
     	a dictionary of pytorch DataLoaders for the training 
     	set, validation set, and test set. It also returns 
     	the dataset sizes and class/label names.'''

    print("LOADING from ", data_dir, " directory")

    data_dirs = {
    	"train" : data_dir + '/train', 
    	"valid" : data_dir + '/valid', 
   	 	"test"  : data_dir + '/test'
	}
    
    data_transforms = {
        "train" : transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
        ]),
        "valid" : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
        ]),
        "test" : transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
        ])
    }

    images = datasets.ImageFolder(data_dir)
    # Load the datasets with ImageFolder
    data = {x: datasets.ImageFolder(data_dirs[x], data_transforms[x]) 
            for x in ['train', 'valid', 'test']}
    # dictionary mapping from class label to class index
    class_2_idx = data['train'].class_to_idx
    # useful for computing accuracy later
    dataset_sizes = {x: len(data[x]) for x in ['train', 'valid', 'test']}
    # get all of the different labels
    class_names = data['train'].classes
    # define the dataloaders
    data_loaders = {x: torch.utils.data.DataLoader(data[x], batch_size=batch_sz, 
                        shuffle=True) for x in ['train', 'valid', 'test']}
    return data_loaders, dataset_sizes, class_names, class_2_idx

def save(trained_model, optimzer, epoch, 
         class_2_idx, save_dir='checkpoint.pth'):
	state = {
		'epoch' : epoch,
		'class_to_idx' : class_2_idx,
    	'classifier' : trained_model.classifier,
    	'state_dict': trained_model.state_dict(),
    	'optimizer': optimzer
	}
	torch.save(state, save_dir)

def load_from_checkpoint(filepath, model_arch="vgg16_bn", gpu=False):
    '''load the pre-trained model, and return it with 
        desired transfer learned weights '''
    print("LOADING FROM", filepath, "directory")
    if gpu:
    	state = torch.load(filepath)
    else:
    	state = torch.load(filepath, lambda storage, loc: storage)#map_location=torch.device('cpu'))
    if model_arch == "vgg16_bn":
        model = models.vgg16_bn(pretrained=True)
        model.classifier = state['classifier']
        model.load_state_dict(state['state_dict'])
        return model
    elif model_arch == "alexnet":
        model = models.alexnet(pretrained=True)
        model.classifier = state['classifier']
        model.load_state_dict(state['state_dict'])
        return model
        
    else:
        print("NOT VGG 16 BN")
        return None

def softmax(x):
    '''used to squash probabilities during prediction'''
    return torch.exp(x) / torch.sum(torch.exp(x))

def plot_metrics(train_losses, valid_losses, train_accuracy, valid_accuracy, last_epoch):
	sns.set_style('white')
	fig, ax1 = plt.subplots()
	t = np.arange(1, len(train_losses)+1)

    # Plot losses
	ax1.plot(t, train_losses, 'b--', label='train losses')
	ax1.plot(t, valid_losses, 'b', label='valid losses')
	ax1.set_xlabel('EPOCH')
	ax1.set_ylabel('LOSS', color='b')
	ax1.tick_params('y', colors='b')
	ax1.spines['top'].set_visible(False)
	ax1.legend(bbox_to_anchor=(-0.15, 0.6), fancybox=True, shadow=True) # center

    # Plot accuracy on the same figure
	ax2 = ax1.twinx()
	ax2.plot(t, train_accuracy, 'r--', label='train accuracy')
	ax2.plot(t, valid_accuracy, 'r', label='valid accuracy')
	ax2.set_ylabel('ACCURACY', color='r')
	ax2.tick_params('y', colors='r')
	ax2.spines['top'].set_visible(False)
	ax2.legend(bbox_to_anchor=(1.45, 0.6), fancybox=True) # center right
	plt.show()

def show_image_grid(data_loader, cat_to_name):
    inputs, labels = next(iter(data_loader['train']))
    out = torchvision.utils.make_grid(inputs)
    print([cat_to_name[str(int(x)+1)] for x in labels])
    show_batch(out)

def show_batch(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def imshow(image, ax=None, title=None):
    '''given an image as a numpy array, this function
        undoes the normalization preprocessing and shows
        the image. NOTE: see preprocess_image below for how
        to convert a PIL image to a format that 
        can be used as input to imshow'''
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax
	
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''
    width, height = image.size
    # rescale image with smallest dim as 256 pixels
    # and no change to the aspect ratio of the photo
    if width > height:
        width  = round(256*(width/height))
        height = 256
    elif height > width:
        height = round(256*(height/width))
        width  = 256
    else:
        width = height = 256
   
    image = image.resize(size = (width,height))
    
    # crop center 224X224 of image 
    if width > height:
        dist = int(width - 224)/2
        image = image.crop(box = (dist, 16, width-dist, 240))
    elif height > width:
        dist = int(height - 224)/2
        image = image.crop(box = (16, dist, 240, height-dist))
    else:
        image = image.crop(box = (16, 16, 240, 240))
        
    # convert newly formatted PIL image to numpy array
    np_image = np.array(image)
    np_image = np_image/256
    # apply network's normalization expectations
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    np_image[:,:,0] = (np_image[:,:,0] - means[0])/stds[0]
    np_image[:,:,1] = (np_image[:,:,1] - means[1])/stds[1]
    np_image[:,:,2] = (np_image[:,:,2] - means[2])/stds[2]
    # pytorch places color channel last so we have to permute dims
    preprocessed_image = np.transpose(np_image, (2,0,1))
    return preprocessed_image