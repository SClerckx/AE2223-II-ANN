### type_preprocessing.py ###
import cv2
from random import randint
from training import get_paths_and_names

def save_images(images, image_filenames, image_type):
    for i in range(len(images)):
        cv2.imwrite(f'images/type_preprocessed_{image_type}/{image_filenames[i]}', images[i])

def preprocess_image(image, image_type):
    dim = (320, 180)

    result = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    result = cv2.GaussianBlur(result, (5, 5), 0)
    if image_type == "outline":
        result = cv2.Canny(result, 0, 200)
    
    return result

def processing(image_paths, image_filenames):
    image_type = "grey" if int(input("Type? (1 for gray, 2 for outline): ")) == 1 else "outline"
    images = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in image_paths]

    processed_images = []
    for i in range(len(images)):
        result = preprocess_image(images[i], image_type)
        processed_images.append(result)

    save_images(processed_images, image_filenames, image_type)  

if __name__ == "__main__":
    image_paths, image_filenames = get_paths_and_names("images//type_preprocessed_grey")
    processing(image_paths, image_filenames)