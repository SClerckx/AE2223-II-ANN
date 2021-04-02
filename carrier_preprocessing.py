import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from training import get_paths_and_names

def save_images(images, image_filenames):
    """Save processed images in folder."""
    for i in range(len(images)):
        cv2.imwrite(f'images/carrier_preprocessed/{image_filenames[i]}', images[i])

def preprocess_image(image):
    """Function that resizes image to (320, 180), blurs it and returns the result."""
    dim = (320, 180)

    result = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    result = cv2.GaussianBlur(result, (5, 5), 0)
    
    return result

def processing(image_paths, image_filenames):
    """Main processing function which runs preprocess_image function for all images."""
    # Load image
    images = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in image_paths]

    processed_images = []
    for i in range(len(images)):
        result = preprocess_image(images[i])
        processed_images.append(result)

    save_images(processed_images, image_filenames)  
    
if __name__ == "__main__":
    image_paths, image_filenames = get_paths_and_names("images//carrier_preprocessed")
    processing(image_paths, image_filenames)