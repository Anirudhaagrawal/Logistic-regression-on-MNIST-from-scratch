import idx2numpy
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_data(data_directory, train = True):
    if train:
        images = idx2numpy.convert_from_file(os.path.join(data_directory, 'train_images'))
        labels = idx2numpy.convert_from_file(os.path.join(data_directory, 'train_labels'))
    else:
        images = idx2numpy.convert_from_file(os.path.join(data_directory, 'test_images'))
        labels = idx2numpy.convert_from_file(os.path.join(data_directory, 'test_labels'))

    vdim = images.shape[1] * images.shape[2]
    vectors = np.empty([images.shape[0], vdim])
    for imnum in range(images.shape[0]):
        imvec = images[imnum, :, :].reshape(vdim, 1).squeeze()
        vectors[imnum, :] = imvec
    
    return vectors, labels

def z_score_normalize(X, u = None, sd = None):
    """
    Performs z-score normalization on X. 

    f(x) = (x - μ) / σ
        where 
            μ = mean of x
            σ = standard deviation of x

    Parameters
    ----------
    X : np.array
        The data to z-score normalize
    u (optional) : np.array
        The mean to use when normalizing
    sd (optional) : np.array
        The standard deviation to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with mean 0 and stdev 1
            Computed statistics (mean and stdev) for the dataset to undo z-scoring.
    """
    
    mean = u if u else X.mean()
    sd = sd if sd else X.std()
    X_normalized = (X-mean)/sd
    return X_normalized, mean, sd

def min_max_normalize(X, _min = None, _max = None):
    """
    Performs min-max normalization on X. 

    f(x) = (x - min(x)) / (max(x) - min(x))

    Parameters
    ----------
    X : np.array
        The data to min-max normalize
    _min (optional) : np.array
        The min to use when normalizing
    _max (optional) : np.array
        The max to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with all values in [0,1]
            Computed statistics (min and max) for the dataset to undo min-max normalization.
    """

    min = _min if _min else X.min()
    max = _max if _max else X.max()

    X_normalized = (X-min)/(max-min) 

    return X_normalized, min, max

def onehot_encode(y):
    """
    Performs one-hot encoding on y.

    Ideas:
        NumPy's `eye` function

    Parameters
    ----------
    y : np.array
        1d array (length n) of targets (k)

    Returns
    -------
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.
    """
    res = np.eye(10)[np.array(y).reshape(-1)]
    return res.reshape(list(y.shape)+[10])
    

def onehot_decode(y):
    """
    Performs one-hot decoding on y.

    Ideas:
        NumPy's `argmax` function 

    Parameters
    ----------
    y : np.array
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.

    Returns
    -------
        1d array (length n) of targets (k)
    """
    return np.argmax(y, axis=1)

def shuffle(dataset):
    """
    Shuffle dataset.

    Make sure that corresponding images and labels are kept together. 
    Ideas: 
        NumPy array indexing 
            https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing

    Parameters
    ----------
    dataset
        Tuple containing
            Images (X)
            Labels (y)

    Returns
    -------
        Tuple containing
            Images (X)
            Labels (y)
    """
    idx = np.random.permutation(len(dataset[0]))
    x,y = dataset[0][idx], dataset[1][idx]  
    return x, y 

def append_bias(X):
    """
    Append bias term for dataset.

    Parameters
    ----------
    X
        2d numpy array with shape (N,d)

    Returns
    -------
        2d numpy array with shape (N,d+1)
    """
    m = X.shape[0]
    return np.c_[np.ones((m, 1)), X]

def generate_minibatches(dataset, batch_size=64):
    X, y = dataset
    minibatches=[]
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        minibatches.append(( X[l_idx:r_idx], y[l_idx:r_idx]))
        l_idx, r_idx = r_idx, r_idx + batch_size

    minibatches.append((X[l_idx:], y[l_idx:]))
    return minibatches

def generate_k_fold_set(dataset, k = 5): 
    X, y = dataset
    if k == 1:
        yield (X, y), (X[len(X):], y[len(y):])
        return

    order = np.random.permutation(len(X))
    
    fold_width = len(X) // k

    l_idx, r_idx = 0, fold_width

    for i in range(k):
        train = np.concatenate([X[order[:l_idx]], X[order[r_idx:]]]), np.concatenate([y[order[:l_idx]], y[order[r_idx:]]])
        validation = X[order[l_idx:r_idx]], y[order[l_idx:r_idx]]
        yield train, validation
        l_idx, r_idx = r_idx, r_idx + fold_width

def filter(dataset, label1, label2):
    X, y = dataset
    indexes=[]
    for i in range(y.shape[0]):
        if y[i]==label1 or y[i]==label2:
                indexes.append(i)
    vector_label1_and_label2 = X[indexes]
    labels_label1_and_label2 = y[indexes]
  
    return vector_label1_and_label2, labels_label1_and_label2.reshape(labels_label1_and_label2.shape[0],1)

def rename(labels,label1):
    for i in range(labels.shape[0]):
        if labels[i]==label1:
                labels[i]=0
        else:
            labels[i]=1
    return labels

def do_pca(n_components, data):
    X = StandardScaler().fit_transform(data)
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca