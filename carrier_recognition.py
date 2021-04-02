import pickle, cv2, os
import numpy as np
from learning import BigOofNetwork
from carrier_preprocessing import preprocess_image
from training import get_paths_and_names, get_names

if __name__ == "__main__":
    file = open('models/carrier_200_1_20.pkl', 'rb')
    network = pickle.load(file)
    file.close()

    image_paths, image_filenames = get_paths_and_names("images\\carrier_preprocessed")
    carrier_name_location = 1
    carriers = get_names(image_filenames, carrier_name_location)

    external_image_paths, external_image_filenames = get_paths_and_names("images\\carrier_preprocessed") #(f"images\\type_preprocessed_{image_type}")#

    correctly_identifieds = 0
    for external_image in external_image_paths:
        image_array = cv2.imread(external_image, cv2.IMREAD_UNCHANGED)
        result = preprocess_image(image_array)

        x = np.array([[pixel.item()] for pixel in np.nditer(result)])
        y = network.feedforward(x)

        predicted_carrier = carriers[np.argmax(y)]
        image_filename = external_image.split("\\")[-1]
        actual_carrier = image_filename.split("_")[1]

        print(f"{actual_carrier} identified as {predicted_carrier}")
        if actual_carrier == predicted_carrier:
            correctly_identifieds += 1
    
    correctly_identified =  correctly_identifieds / len(external_image_paths)
    print(f"correctly identified {correctly_identified*100} % of aircraft")