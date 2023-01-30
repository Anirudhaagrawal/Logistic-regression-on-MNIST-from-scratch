import argparse
import network
import data
import image
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
import itertools
import pandas as pd
import random
from tqdm import tqdm
import streamlit

def print_projected_image(pca,vector_pca):
        vector_projected=pca.inverse_transform(vector_pca)
        vector_projected_reshaped=vector_projected.reshape(vector_projected.shape[0], 28,28)
        plt.figure()
        plt.imshow(vector_projected_reshaped[1]) 
        plt.show()

def print_original_image(vectors):
        plt.figure()
        plt.imshow(vectors[1].reshape(28,28)) 
        plt.show()

def test_on_test(vectors_test, labels_test, network, pca):
        vectors_test = StandardScaler().fit_transform(vectors_test)
        vector_test_pca = pca.transform(vectors_test)
        vectors_processed, labels_processed = process_based_on_dimensions(vector_test_pca, labels_test)
        vector_with_bias_normalized, labels_shuffled = preprocess_data(vectors_processed,labels_processed, hyperparameters)
        return network.test((vector_with_bias_normalized, labels_shuffled))

def set_parameters():
        if hyperparameters.number1 :
                activation = network.sigmoid
                loss_function = network.binary_cross_entropy 
                out_dimensions = 1 
        else:
                activation = network.softmax
                loss_function = network.multiclass_cross_entropy
                out_dimensions = 10
        return activation, loss_function, out_dimensions

def find_best_hyperparameters(hyperparameters):
        learning_rates = [0.1, 0.01, 0.001]
        epochs = [5, 10, 50, 100, 200, 500]
        batch_sizes = [64, 128, 256, 500, 1000, 3000]
        pca_components = [5, 10, 50, 100, 500]

        k_folds=[10]
        normalization=[data.min_max_normalize, data.z_score_normalize]
        all = [learning_rates, epochs, batch_sizes, pca_components, k_folds, normalization]
        all_combinations = list(itertools.product(*all))
        randomly_sampled = random.sample(all_combinations, 100)
        all_accuracies=[]
        for  combination in tqdm(randomly_sampled):
                parser = argparse.ArgumentParser()
                new_hyperparameters = parser.parse_args('')
                new_hyperparameters.learning_rate=combination[0]
                new_hyperparameters.epochs=combination[1]
                new_hyperparameters.batch_size=combination[2]
                new_hyperparameters.p=combination[3]
                new_hyperparameters.k_folds=combination[4]
                new_hyperparameters.number1=hyperparameters.number1
                new_hyperparameters.number2=hyperparameters.number2
                new_hyperparameters.normalization=combination[5]
                accuracy = k_fold_grid_search(new_hyperparameters)
                all_accuracies.append(accuracy)
        max_accuracy = max(all_accuracies)
        index = all_accuracies.index(max_accuracy)
        best_combination = randomly_sampled[index]
        dict = {'combination': randomly_sampled, 'accuracy': all_accuracies}
        df = pd.DataFrame(dict)
        df.to_csv(hyperparameters.filename+'.csv')
        print(best_combination)
        

def k_fold_grid_search(hyperparameters):       
        vectors, labels = data.load_data("data", True)
        activation, loss_function, out_dimensions =  set_parameters()
        pca, vectors_pca = data.do_pca(hyperparameters.p,  vectors)
        vectors_processed, labels_processed = process_based_on_dimensions(vectors_pca, labels)
        vector_with_bias_normalized, labels_shuffled = preprocess_data(vectors_processed,labels_processed, hyperparameters)
        no_examples = labels_shuffled.shape[0]
        fold_size=no_examples//hyperparameters.k_folds
        lower=0
        higher=lower + fold_size

        fold_accuracies = []
        for fold in range(hyperparameters.k_folds):
                net = network.Network(hyperparameters, activation, loss_function, out_dimensions)
                minibaches = data.generate_minibatches((np.concatenate((vector_with_bias_normalized[:lower],vector_with_bias_normalized[higher:])), np.concatenate((labels_shuffled[:lower],labels_shuffled[higher:]))), batch_size=hyperparameters.batch_size)
                for epoch in range(hyperparameters.epochs):
                        last_loss=math.inf
                        for minibatch in minibaches:
                                net.train((minibatch[0],minibatch[1]))  
                        test_loss, valid_accuracy = net.test((vector_with_bias_normalized[lower:higher], labels_shuffled[lower:higher]))  
                        if(abs(test_loss)>abs(last_loss)):
                                print("Stopped Early")
                                break
                        last_loss=test_loss
                fold_loss, fold_accuracy = net.test((vector_with_bias_normalized[lower:higher], labels_shuffled[lower:higher]))
                fold_accuracies.append(fold_accuracy)
                lower=higher
                higher=lower + fold_size

        return np.average(fold_accuracies)



def main(hyperparameters):
        if hyperparameters.grid_search==1:
                hyperparameters.filename="softmax"
                find_best_hyperparameters(hyperparameters)
                hyperparameters.number1=8
                hyperparameters.number2=5
                hyperparameters.filename="eight_five"
                find_best_hyperparameters(hyperparameters)
                hyperparameters.number1=2
                hyperparameters.number2=7
                hyperparameters.filename="two_seven"
                find_best_hyperparameters(hyperparameters)
        else:
                vectors, labels = data.load_data("data", True)
                print_original_image(vectors)
                vectors_test, labels_test = data.load_data("data", False)
                activation, loss_function, out_dimensions =  set_parameters()
                pca, vectors_pca = data.do_pca(hyperparameters.p,  vectors)
                print_projected_image(pca,vectors_pca)
                vectors_processed, labels_processed = process_based_on_dimensions(vectors_pca, labels)
                vector_with_bias_normalized, labels_shuffled = preprocess_data(vectors_processed,labels_processed, hyperparameters)  
                net = network.Network(hyperparameters, activation, loss_function, out_dimensions)
                minibaches = data.generate_minibatches((vector_with_bias_normalized,labels_shuffled), batch_size=hyperparameters.batch_size)
                for epoch in range(hyperparameters.epochs):
                        for minibatch in minibaches:
                                        net.train((minibatch[0],minibatch[1]))  
                print(test_on_test(vectors_test, labels_test, net, pca))


                
               
def preprocess_data(vectors,labels,hyperparameters): 
        vectors_shuffled, labels_shuffled = data.shuffle((vectors, labels))
        vector_with_bias=data.append_bias(vectors_shuffled)
        vector_with_bias_normalized, mean, sd = hyperparameters.normalization(vector_with_bias)
        return vector_with_bias_normalized, labels_shuffled

def process_based_on_dimensions(vectors, labels):
        if hyperparameters.number1:
                vectors_processed, labels= data.filter((vectors, labels),hyperparameters.number1 , hyperparameters.number2)
                labels_processed = data.rename(labels, hyperparameters.number1) 
        else:
                vectors_processed = vectors
                labels_processed = data.onehot_encode(labels)
        return vectors_processed, labels_processed

 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'CSE151B PA1')
    parser.add_argument('--batch-size', type = int, default = 1,
            help = 'input batch size for training (default: 1)')
    parser.add_argument('--epochs', type = int, default = 100,
            help = 'number of epochs to train (default: 100)')
    parser.add_argument('--learning-rate', type = float, default = 0.001,
            help = 'learning rate (default: 0.001)')
    parser.add_argument('--z-score', dest = 'normalization', action='store_const', 
            default = data.min_max_normalize, const = data.z_score_normalize,
            help = 'use z-score normalization on the dataset, default is min-max normalization')
    parser.add_argument('--k-folds', type = int, default = 5,
            help = 'number of folds for cross-validation')
    parser.add_argument('--p', type = int, default = 100,
            help = 'number of principal components')
    parser.add_argument('--number1', type = int, 
            help = 'If activation is sigmoid choose value 1 to differentiate with')
    parser.add_argument('--number2', type = int, 
            help = 'If activation is sigmoid choose value 2 to differentiate with')
    parser.add_argument('--grid-search', type=int, 
            help ="1 if you want to grid search for best hyperparameters 0 if you want to train model on set of given hyperparameters") 


    hyperparameters = parser.parse_args()
    main(hyperparameters)

