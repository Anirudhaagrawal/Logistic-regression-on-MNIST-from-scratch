from data import load_data, z_score_normalize, min_max_normalize, onehot_encode, onehot_decode, shuffle, append_bias
from network import sigmoid, softmax, binary_cross_entropy
import idx2numpy
import numpy as np
import os
import data


vector, labels = load_data("data", False)

 
vector_5_2, label_5_2 = data.filter((vector, labels), 5, 2)

print(vector_5_2)
print(label_5_2)

print(data.rename(label_5_2, 5))

one_hot_decode = onehot_encode(vector_5_2,label_5_2)
prit

print(one_hot_decode)

X, y = shuffle((vector, labels))

print(y)

x_train = append_bias(X)

print(x_train)

arr=[1,0,1,0,1]
target=[0.9,0.1,0.9,0.1,0.9]

k=np.array(arr)
k1=np.array(target)
print(binary_cross_entropy(k1, k))



