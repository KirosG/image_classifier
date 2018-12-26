Implementation of a transfer learning image classifier in pytorch. Models supported by the program include: vgg16_bn (SEE: [Very Deep Convolutional Networks For Large Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)) or alexnet (SEE: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)). The model is able to be run in a jupyter notebook, or as a python command line script. For running from command line, the only files that the user needs to be aware of are train.py and predict.py. Using train.py, the user is able to specify the model architecture (vgg16_bn or alexnet), the learning rate, number of epochs, and number of hidden units (if any) before the output layer. This command then loads the pretrained model of choice and if no hidden layer was specified replaces the output layer of the model with one that fits for the class dimensions of the training dataset supplied by the user. This model is then trained and the best performing version of the model is saved. Using predict.py the user is able to load a model (checkpoint.pth is an example of what train.py saves to load at this time) and feed a picture through it to receive a prediction from the trained model. 
