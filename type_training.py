import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, pickle
from learning import BigOofNetwork
from training import get_paths_and_names, get_names, get_data, save_network

if __name__ == "__main__":
    
    image_type = "grey" if int(input("Type? (1 for gray, 2 for outline): ")) == 1 else "outline"

    image_paths, image_filenames = get_paths_and_names(f"images\\type_preprocessed_{image_type}")

    type_name_location = 0 #filenames are ordered as follows: type_carrier_number.jpg. Carrier is item 1 in list [type, carrier, number]
    types = get_names(image_filenames, type_name_location)

    data = get_data(image_paths, type_name_location, True, types)

    input_size, output_size = len(data[0][0]), len(types)
    network = BigOofNetwork([input_size, 100, 75, 25, output_size])

    epochs = 2
    learning_rate = 1
    mini_batch_size = 20
    network.train_network(data, epochs, mini_batch_size, learning_rate, test_data = data)
    network.plot_accuracies()

    save_network(network, f"type_{image_type}", epochs, learning_rate, mini_batch_size)