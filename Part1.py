''''
COMP3710 D2 Eigenfaces Lachlan Bunt
Not all original code
'''
import torch
from time import perf_counter
# Speed up sklearn algorithms
if True:
    from sklearnex import patch_sklearn
    patch_sklearn()

from sklearn.datasets import fetch_lfw_people, get_data_home
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

'''Check Cuda'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning couldn't find cuda using cpu.")


'''PCA'''
start_time = perf_counter()
# Download the data, if not already on disk and load it as numpy arrays
#print(get_data_home())
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
print(f"Shape of the images {lfw_people.images.shape}")
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
print(f"Shape of the data {X.shape}")
n_features = X.shape[1]
# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# Split into a training set and a test set using a stratified k fold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

X_train.to(device)
X_test.to(device)
y_train.to(device)
y_test.to(device)

n_components = 150

# Center data

mean = torch.mean(X_train, dim=0)
X_train -= X_train.subtract(mean)
X_test -= X_test.subtract(mean)
#Eigen-decomposition
U, S, V = torch.svd(X_train, some=False)
components = V[:n_components]
eigenfaces = components.reshape((n_components, h, w))
#project into PCA subspace
X_transformed = torch.mm(X_train, components.T)
print(X_transformed.shape)
X_test_transformed = torch.mm(X_test, components.T)
print(X_test_transformed.shape)

'''Plot PCA'''

# Qualitative evaluation of the predictions using matplotlib
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.show()

#eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
#plot_gallery(eigenfaces, eigenface_titles, h, w)

# Results
def show_PCA_results():
    explained_variance = (torch.matrix_power(S, 2)).div(n_samples - 1)
    total_var = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_var
    ratio_cumsum = torch.cumsum(explained_variance_ratio, dim=0).sum()
    print(ratio_cumsum.shape)
    eigenvalueCount = torch.arange(n_components)
    plt.plot(eigenvalueCount, ratio_cumsum[:n_components])
    plt.title('Compactness')
    plt.show()

''''Model'''
#build random forest
estimator = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=150)
estimator.fit(X_transformed, y_train) #expects X as [n_samples, n_features]
predictions = torch.tensor(estimator.predict(X_test_transformed))
correct = predictions==y_test
total_test = len(X_test_transformed)
#print("Gnd Truth:", y_test)
print("Total time,", perf_counter() - start_time)
print("Total Testing", total_test)
print("Predictions", predictions)
print("Which Correct:",correct)
print("Total Correct:",torch.sum(correct))
print("Accuracy:",torch.sum(correct)/total_test)
print(classification_report(y_test, predictions, target_names=target_names))
