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
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from itertools import permutations
from sklearn.svm import LinearSVC

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

def get_HOG_features(X):
    """
    This method rescales all images to 150x150, grayscales then and extracts features from images using HOG.
    """
    X_data = []
    for img in tqdm(X.image):
        image = resize(io.imread(img, as_gray=True), (150,150))
        fd, _ = hog(image, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True)
        X_data.append(fd)
    return X_data

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





path = f'{os.getcwd()}/emotion-recognition-dataset/dataset'

df = get_dataset(path)

X, y = prepare_data(df)

X = get_HOG_features(X)
y = get_encoded_labels(y)

X = reduce_dimensions(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=420)


gmm = GaussianMixture(n_components = 5, verbose=1, init_params='kmeans',
                        tol=3.334e-02)
gmm.fit(X_train)
predictions = gmm.predict(X_test)
silhouette = silhouette_score(X_test, predictions)

best_predictions, _ = compute_accuracy(predictions, y_test)
plot_confusion_matrix(y_test, best_predictions,"GMM HOG")

compute_dummy(X_train, X_test, y_train, y_test)
compute_SVM(X_train, X_test, y_train, y_test)