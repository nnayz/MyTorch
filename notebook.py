# %%
import pandas as pd

dataset = pd.read_csv('./archive/mnist_train.csv')
print(dataset)
# %%
from layer import Input_Layer, Dense_Layer
import numpy as np

input_layer = Input_Layer(n_neurons=768) # Number of features (pixel values)
input_layer.receive_data(np.array(dataset.iloc[0]))
# print(dataset.to_numpy().shape)
