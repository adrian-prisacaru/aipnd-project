import argparse
import json
import torch
import numpy as np
from PIL import Image
from utils import build_model, determine_device


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = build_model(checkpoint.arch, checkpoint.hidden_units, checkpoint.dropout)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image.thumbnail((256, 256))
    width, height = image.size
    new_width, new_height = (224, 224)
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))
    np_image = np.array(image) / 255

    # normalize the image
    mean = [0.485, 0.456, 0.406]
    sd = [0.229, 0.224, 0.225]
    np_image = np_image - mean
    np_image = np_image / sd

    # transpose the image as expected by pytorch
    np_image = np_image.transpose(2, 0, 1)
    return torch.from_numpy(np_image)


def predict(image_path, checkpoint, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with Image.open(image_path) as image:
        processed = process_image(image)
        model = load_checkpoint(checkpoint)
        model.eval()
        img = processed.unsqueeze(0)
        img.to(device)
        # Calculate the class probabilities (softmax) for img
        with torch.no_grad():
            output = model.forward(img.float())
            ps = torch.exp(output)
            top_p, top_class = ps.topk(topk, dim=1)
            top_p = top_p.data.numpy().squeeze()
            top_class = top_class.numpy()[0]
            return top_p, top_class


def parse_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    parser.add_argument('input', help='Path to image')
    parser.add_argument('checkpoint', help='Checkpoint of saved model')
    parser.add_argument('--top_k', default=5, type=int, help='Number of top classes to show')
    parser.add_argument(
        '--category_names',
        default='cat_to_name.json',
        help='Mapping from category label to category name'
    )
    parser.add_argument('--gpu', action='store_true', help='Enable GPU training')
    return parser.parse_args()


def main():
    print('predict started')
    args = parse_args()
    # mapping from category label to category name
    with open(args.cat_to_name, 'r') as f:
        cat_to_name = json.load(f)

    device = determine_device(args.gpu)

    # predict
    top_p, top_class = predict(input, args.checkpoint, device, args.topk)
    top_class = list(map(lambda cl: cat_to_name[str(cl)], top_class))
    print(top_class)
    print(top_p)


if __name__ == "__main__":
    main()
