import pandas as pd
import os
import opendatasets as od
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import tensorflow
import keras
import glob
from skimage import io
import skimage
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from itertools import permutations
from sklearn.svm import LinearSVC
from skimage.feature import local_binary_pattern
from sklearn.model_selection import ParameterGrid

from sklearn.cluster import AgglomerativeClustering

od.download('https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset')


def get_dataset(path):
    """
        This function loads the images and returns a DataFrame with 2 columns: image and label (of the image).
    """
    image_paths = []
    labels = []

    # This for iterates through all images in the path directory
    for image in tqdm(sorted((Path(path).glob('*/*.*')))):
        image_paths.append(str(image))  # Here I save image path in the list.

        label = str(image).split('/')[-2]  # Here I extract the label from the image path.

        labels.append(label)  # Finally I add the label to the list

    # Create a pandas dataframe from the collected file names and labels
    return pd.DataFrame.from_dict({"image": image_paths, "label": labels})

def prepare_data(df):
    """
    This function drops Ahegao images and separates images from labels into X and y.
    """
    print(df['label'].unique())
    print(df[df['label'] == 'Ahegao'].count())
    df = df.drop(labels = range(1205), axis=0, inplace = False) #Here we drop Ahegao images as it is NSFW.

    y = df['label']
    X = df.drop(['label'],axis=1,)
    return X, y

def get_lbp(image_url, x_size=150, y_size=150, radius = 1, no_points=8):
    """
    This method rescales an image to 150x150, grayscales it then and extracts features from the image using LBP.
    """
    image = io.imread(image_url, as_gray=True)
    image = resize(image, (x_size,y_size))
    return local_binary_pattern(image, no_points, radius)

def get_LBP_features(X):
    """
    This method extracts features from a vector of images using LBP.
    """
    X_data = []
    for img in tqdm(X.image):
        image = get_lbp(img)
        X_data.append(image)
    return X_data

def transform_extracted_features(X):
    """
    This function reshapes NxM matrices into a vector of dimension N*M.
    """
    X = np.asarray(X)
    nsamples, nx, ny = X.shape
    X = X.reshape((nsamples,nx*ny))
    return X

def get_encoded_labels(y):
    """
    This method encodes the labels using LabelEncoder.
    """
    le = LabelEncoder()
    y = le.fit_transform(y)
    return y

def reduce_dimensions(X, n_dimensions=100):
    """
        This method reduces dimensions of a vector using PCA.
    """
    pca = PCA(n_components = n_dimensions)
    return pca.fit_transform(X)

def plot_confusion_matrix(y_true, predicted, title, labels=None, normalize=None):
    """
    This function plots a confusion matrix with 6x6 size and a 100 dpi resolution.
    """

    #I changed the figure size and increased the dpi for a better resolution.
    _, axes = plt.subplots(figsize=(6,6), dpi=100)

    conf_matrix = confusion_matrix(y_true, predicted, normalize=normalize)
    display = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)

    #I set the title using the axes object.
    axes.set(title=f'Confusion Matrix {title} Model')

    display.plot(ax=axes)
    return conf_matrix

def compute_accuracy(predictions, y_test, n_clusters=5):
    """
        This methods cycles through all the ways in which the predicted clusters
        could be assigned to the real labels until it finds and returns the best
        accuracy and permutation of predictions.
    """
    label_permutation = list(permutations(range(0, n_clusters)))
    best_predictions=[]
    accuracy=0
    for permutation in tqdm(label_permutation):
        dict_perm={0:permutation[0], 1:permutation[1], 2:permutation[2], 3:permutation[3], 4:permutation[4]}
        new_predictions=[]
        for pred in predictions:
            pred = dict_perm[pred]
            new_predictions.append(pred)
    acc = np.mean(new_predictions == y_test)
    if(acc>accuracy):
        accuracy = acc
        best_predictions = new_predictions

    print(f" Accuracy: {accuracy * 100:.2f}%")
    return best_predictions, accuracy

def compute_dummy(X_train, X_test, y_train, y_test):
    """
    This method computes a dummy classifier and returns its accuracy.
    """
    from sklearn.dummy import DummyClassifier
    dummy_clf = DummyClassifier()
    dummy_clf.fit(X_train, y_train)
    dummy_predictions = dummy_clf.predict(X_test)

    accuracy = np.mean(y_test == dummy_predictions)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def compute_SVM(X_train, X_test, y_train, y_test):
    """
        This function computes a default LinearSVC classifier and returns its accuracy.
    """
    
    model = LinearSVC(max_iter=10000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)

    print(f"Accuracy: {accuracy * 100:.2f}%")


def compute_grid_search(X):
    """
    This method does a grid search for Agglomerative Clustering model returning 
    the parameters for which the Silhouette score is the highest.
    """
    param_grid = {
        'n_clusters': [5, 6, 7],
        'linkage': ['complete', 'average', 'single'],
        'metric':['cosine', 'minkowski', 'chebyshev', 'sqeuclidean',
                'precomputed', 'l1', 'manhattan', 'l2', 'wminkowski',
                'euclidean', 'hamming']
    }
    best_score = -1  
    best_params = None

    for params in tqdm(ParameterGrid(param_grid)):
    
        model = AgglomerativeClustering(**params)
        labels = model.fit_predict(X)

        score = silhouette_score(X, labels)
        
        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score


path = f'{os.getcwd()}/emotion-recognition-dataset/dataset'

df = get_dataset(path)

X, y = prepare_data(df)

X = get_LBP_features(X)
y = get_encoded_labels(y)

X = transform_extracted_features(X)
X = reduce_dimensions(X)

best_params, best_score = compute_grid_search(X)
print("Best Parameters:", best_params)
print("Best Silhouette Score:", best_score)

cluster = AgglomerativeClustering(n_clusters=5, metric='sqeuclidean', linkage='average')
cluster.fit(X)
predictions = cluster.fit_predict(X)
silhouette = silhouette_score(X, predictions)

best_predictions, _ = compute_accuracy(predictions, y)
plot_confusion_matrix(y, best_predictions,"Agglomerative Clustering LBP")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=420)
compute_dummy(X_train, X_test, y_train, y_test)
compute_SVM(X_train, X_test, y_train, y_test)