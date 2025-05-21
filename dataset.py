import kagglehub

custom_download_path = "/Users/nayz/Desktop/Projects/neural_networks_from_scratch/"
returned_path = kagglehub.dataset_download('oddrationale/mnist_in-csv', path=custom_download_path, force_download=True)

print(returned_path)
