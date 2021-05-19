import argparse
import json
import torch
import numpy as np
from PIL import Image
from utils import build_model, determine_device


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = build_model(checkpoint['arch'], checkpoint['hidden_units'], checkpoint['dropout'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model


def resize_image(image):
    width, height = image.size
    aspect_ratio = width / height
    if width < height:
        new_width = 256
        new_height = int(new_width / aspect_ratio)
    elif height < width:
        new_height = 256
        new_width = int(width * aspect_ratio)
    else: # when both sides are equal
        new_width = 256
        new_height = 256
    return image.resize((new_width, new_height))


def crop_image(image):
    ''' Crop the center of the image
    '''
    width, height = image.size
    new_width, new_height = (224, 224)
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    return image.crop((left, top, right, bottom))


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image = resize_image(image)
    image = crop_image(image)

    np_image = np.array(image) / 255

    # normalize the image
    mean = [0.485, 0.456, 0.406]
    sd = [0.229, 0.224, 0.225]
    np_image = np_image - mean
    np_image = np_image / sd

    # transpose the image as expected by pytorch
    np_image = np_image.transpose(2, 0, 1)
    return torch.from_numpy(np_image)


def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with Image.open(image_path) as image:
        processed = process_image(image)
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


def print_result(top_class, top_p):
    result = list(zip(top_class, top_p))
    result.sort(key=lambda x: x[1], reverse=True)
    print("\nTop classes")
    print("-----------")
    for top_class, top_p in result:
        print("{}: {}".format(top_class, top_p))


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
    args = parse_args()
    # mapping from category label to category name
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    device = determine_device(args.gpu)
    model = load_checkpoint(args.checkpoint)
    model.to(device)

    # predict
    top_p, top_class = predict(args.input, model, device, args.top_k)

    # top_class is the index, not the actual class
    # convert to classes and use cat_to_name to get the labels
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_categories = [idx_to_class[idx] for idx in top_class]
    labels = [cat_to_name[cat] for cat in top_categories]

    print_result(labels, top_p)


if __name__ == "__main__":
    main()
