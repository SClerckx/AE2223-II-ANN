import pickle, cv2, os
import numpy as np
from type_training import BigOofNetwork
from type_preprocessing import preprocess_image
from training import get_paths_and_names, get_names

if __name__ == "__main__":
    image_type = "grey" if int(input("Type? (1 for gray, 2 for outline): ")) == 1 else "outline"

    file = open(f'models/type_{image_type}_600_1_20.pkl', 'rb')
    network = pickle.load(file)
    file.close()

    image_paths, image_filenames = get_paths_and_names(f"images\\type_preprocessed_{image_type}")
    type_name_location = 0
    types = get_names(image_filenames, type_name_location)

    external_image_paths, external_image_filenames = get_paths_and_names(f"images\\external_images\\types") #(f"images\\type_preprocessed_{image_type}")#

    correctly_identifieds = 0
    for external_image in external_image_paths:
        image_array = cv2.imread(external_image, cv2.IMREAD_GRAYSCALE)
        result = preprocess_image(image_array, image_type)

        x = np.array([[pixel.item()] for pixel in np.nditer(result)])
        y = network.feedforward(x)

        predicted_type = types[np.argmax(y)]
        image_filename = external_image.split("\\")[-1]
        actual_type = image_filename.split("_")[0]

        print(f"{actual_type} identified as {predicted_type}")
        if actual_type == predicted_type:
            correctly_identifieds += 1
    
    correctly_identified =  correctly_identifieds / len(external_image_paths)
    print(f"correctly identified {correctly_identified*100} % of aircraft")