import os, pickle, cv2
import matplotlib.pyplot as plt
import numpy as np

def get_paths_and_names(directory):
    """
    directory (string): local reference to desired directory to scout. Example: "images\\carrier_preprocessed"

    returns:
    image_paths (list(string)): list of the paths of the files in the scouted directory
    image_filenames (list(string)): list of the names of the files in the scouted directory
    """
    cur_dir = os.getcwd()
    image_paths = [os.path.join(cur_dir, directory, file) for file in os.listdir(os.path.join(cur_dir, directory)) if file.endswith('.jpg')]
    image_filenames = [name for name in os.listdir(os.path.join(cur_dir, directory)) if name.endswith('.jpg')]

    return image_paths, image_filenames

def get_names(filenames, location):
    """
    filenames (list(string)): list of the names of the files
    location (int): location to find the name in the filename

    returns:
    names (list(string)): list of found names
    """
    names = []
    for filename in filenames:
        name = filename.split("_")[location]
        if name not in names:
            names.append(name)
    return names

def get_data(image_paths, location, greyscale, names):
    """
    image_paths (list(string)): list of the paths of the files that should be converted into trainable data
    location (int): location to find the name in the filename
    greyscale (bool): read the images in greyscale or in rgb
    names (list(string)): list of names

    returns:
    data (list(tuple)): list of the trainable data
    """
    data = []
    for image_path in image_paths:

        if not greyscale:
            image_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        else:
            image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        image_filename = image_path.split("\\")[-1]
        name = image_filename.split("_")[location]
        
        x = np.array([[pixel.item()] for pixel in np.nditer(image_array)])
        y = np.array([[1] if name == t else [0] for t in names])
        data.append((x,y))

    return data

def save_network(network, name, epochs, learning_rate, mini_batch_size):
    """
    Saves the trained netowrk in a .pkl file in the models folder

    network (BigOofNetwork): network to save
    name (string): name for the network
    epochs, learning_rate, mini_batch_size (int): training settings of the network
    """
    save = input("Should I save? [y/N]: ").lower()
    if save == "y":
        with open(f'models/{name}_{epochs}_{learning_rate}_{mini_batch_size}.pkl' , 'wb') as output:
            pickle.dump(network, output, pickle.HIGHEST_PROTOCOL)
