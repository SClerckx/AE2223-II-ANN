from learning import BigOofNetwork
from training import get_paths_and_names, get_names, get_data, save_network

if __name__ == "__main__":
    image_paths, image_filenames = get_paths_and_names("images\\carrier_preprocessed")

    carrier_name_location = 1 #filenames are ordered as follows: type_carrier_number.jpg. Carrier is item 1 in list [type, carrier, number]
    carriers = get_names(image_filenames, carrier_name_location)

    data = get_data(image_paths, carrier_name_location, False, carriers)

    input_size, output_size = len(data[0][0]), len(carriers)
    network = BigOofNetwork([input_size, 100, 75, 25, output_size])

    epochs = 2
    learning_rate = 1
    mini_batch_size = 20
    network.train_network(data, epochs, mini_batch_size, learning_rate, test_data = data)
    network.plot_accuracies()

    save_network(network, "carrier", epochs, learning_rate, mini_batch_size)