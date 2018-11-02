import argparse
import sys
import utils
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

def predict(image_path, model, gpu, cat_to_name=None, topk=5):
    ''' Predict the class (or classes) of an image 
            using a trained deep learning model.
        Prints a bar chart of the top 5 most likely 
            labels for the given input image. '''
    model.train(False)
    model.eval()
    np_image = utils.process_image(image_path)
    image = torch.from_numpy(np_image)
    image = image.type(torch.FloatTensor)
    image = image.unsqueeze(0)

    if gpu and not torch.cuda.is_available():
        print("Sorry, no cuda enabled gpu is available.")
        print("Predicting on the cpu...")
        gpu = False

    if gpu:
        model.cuda()
        image.cuda()

    output = model(image)
    prob_output = utils.softmax(output)
    probs, classes = prob_output.topk(topk)

    probs = probs.cpu().detach().numpy()[0]
    
    fig, ax = plt.subplots()
    ax.barh(np.arange(5), probs, align='center')
    ax.set_yticks(np.arange(5))

    if cat_to_name is not None:
        image_names = []
        for image_name_idx in classes[0]:
            image_names.append(cat_to_name[str(int(image_name_idx) + 1)])
        ax.set_yticklabels(image_names)

    ax.invert_yaxis()

    for i, prob in enumerate(probs):
        ax.text(prob + (np.max(probs) / 100), i+.1, 
                round(prob,3), color='black', fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('SOFTMAX PROBABILITY')
    ax.set_title('PREDICTED PROBABILITIES')
    ax.set_xlim(0)
    ax.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='picture to predict')
    parser.add_argument('checkpoint', help='model checkpoint to upload')
    parser.add_argument('--top_k', action='store', dest='top_k', default=5, 
                         help='the number of probabilities to show')
    parser.add_argument('--category_names', action='store', dest='category_names', 
                         default=None, help='the label names')
    parser.add_argument('--gpu', action='store', dest='gpu', default=True, 
                         help='the model architecture')	
    results = parser.parse_args()
    image_path = results.input
    checkpoint = results.checkpoint
    top_k = int(results.top_k)
    category_names_json = results.category_names
    gpu = results.gpu

    category_names = None
    if category_names_json == None:
        print("No label to index mapping provided")
    else: 
        with open(str(category_names_json), 'r') as f:
            category_names = json.load(f)

    model = utils.load_from_checkpoint(checkpoint)
    pil_image = Image.open(image_path, 'r')
    predict(pil_image, model, gpu, category_names, topk=top_k)	