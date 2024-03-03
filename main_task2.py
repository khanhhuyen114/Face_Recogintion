import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import IncrementalPCA

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import cluster, decomposition

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class ModularPCA:
    def __init__(self, n_components=None, svd_solver='randomized', whiten=False, random_state=42, num_regions=4):
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.whiten = whiten
        self.random_state = random_state
        self.pca_models = []
        self.num_regions = num_regions

    def is_power_of_two(self,n):
        """Check if a number is a power of 2."""
        return n != 0 and (n & (n - 1)) == 0

    def divide_pics(self,images, n_regions, shape):
        '''images with original size'''
        all_parts = [[] for _ in range(n_regions)]

        if self.is_power_of_two(n_regions):
            rows_per_part = shape[0] // (n_regions // 2)
            cols_per_part = shape[1] // (n_regions // 2)
            for image in images:
                for i in range(n_regions):
                    row_start = (i // (n_regions // 2)) * rows_per_part
                    col_start = (i % (n_regions // 2)) * cols_per_part
                    row_end = row_start + rows_per_part
                    col_end = col_start + cols_per_part
                    img = image[row_start:row_end, col_start:col_end]
                    all_parts[i].append(img.flatten())
        else:
            raise ValueError("Can't divide pic into equal regions both in rows and columns")

        return all_parts

    def fit(self, X, y=None):
        '''X is Flatten Image'''
        X = [x.reshape(112,92) for x in X]
        # Split each face image into regions
        X_regions = self.divide_pics(images = X, n_regions=self.num_regions, shape=X[0].shape)

        # Apply PCA separately to each region
        self.pca_models = []
        for region in X_regions:
            pca = PCA(
                n_components=self.n_components,
                svd_solver=self.svd_solver,
                whiten=self.whiten,
                random_state=self.random_state
            )
            pca.fit(region)
            self.pca_models.append(pca)
        return self



    def transform(self, X):
        '''X is Flatten Image'''


        X = [x.reshape(112,92) for x in X]
        # Transform each region using its corresponding PCA model
        X_regions = self.divide_pics(X, self.num_regions, X[0].shape)

        X_transformed_parts = []
        X_transformed = []
        for pca, region in zip(self.pca_models, X_regions):
            transformed_region = pca.transform(region)
            X_transformed_parts.append(transformed_region)

        for parts in zip(*X_transformed_parts):
            X_transformed.append( np.concatenate([part for part in parts]))
        return X_transformed


    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def __repr__(self):
        return f'ModularPCA(n_components={self.n_components})'

    def visualize_eigenfaces(self, X):
        """  Best with 5 components"""
        self.fit(X)
        eigenfaces = []
        concatenated_image = []
        for pca_model in self.pca_models:
            eigenfaces.append(pca_model.components_)

        new_eigenfaces = [[] for _ in range(self.num_regions)]
        for i in range(self.num_regions):
            for j in range(self.n_components):
                new_eigenfaces[i].append(eigenfaces[i][j].reshape(112//(self.num_regions //2), 92//(self.num_regions //2)))

        for parts in zip(*new_eigenfaces):
            a = np.concatenate(parts[:(self.num_regions //2)],axis = 1)
            b = np.concatenate(parts[(self.num_regions //2):],axis = 1)
            concatenated_image.append(np.concatenate((a, b), axis=0))
        plt.figure(figsize=(10,5))
        for i in range(self.n_components):
            plt.subplot(1, 5, i + 1)
            plt.imshow(concatenated_image[i], cmap='gray')
            plt.xticks(())
            plt.yticks(())
            plt.title(i)
        plt.show()

    def inverse_transform(self, X):
        # Check if the PCA models are fitted
        if not self.pca_models:
            print("This ModularPCA1 instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        # Initialize the list to hold the inversely transformed regions
        inversely_transformed_parts = []

        # Determine the number of samples and reshape the input to separate regions
        num_samples = len(X) // self.num_regions
        split_X = [X[i * num_samples:(i + 1) * num_samples] for i in range(self.num_regions)]
        # for i in split_X:
        #   print(i)
        #   print(len(i))

        # Inverse transform each region
        for pca, region in zip(self.pca_models, split_X):
            inversely_transformed_region = pca.inverse_transform(region)
            part = inversely_transformed_region.reshape(56,46)
            inversely_transformed_parts.append(part)


        # Reconstruct the original images from the inversely transformed regions

        recon_part1 = np.concatenate((inversely_transformed_parts[0],inversely_transformed_parts[1]), axis = 1)
        recon_part2 = np.concatenate((inversely_transformed_parts[2],inversely_transformed_parts[3]), axis = 1)
        reconstructed_images = np.concatenate((recon_part1,recon_part2), axis = 0)
        img = reconstructed_images.flatten()
        return img

class FaceRecognitionModel:
  def __init__(self, pca_method, classifier_method, tune = False, params = None):
    self.pca = pca_method
    self.clf = classifier_method
    self.original_xtrain, self.original_ytrain, self.original_xtest, self.original_ytest = None, None, None, None
    self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
    self.y_pred = None
    self.tunning = tune
    self.param_grid = params

  def fit(self, X, y):
    # Dimensional Reduction
    self.x_train = self.pca.fit_transform(X)
    self.clf.fit(self.x_train,y)

    self.original_xtrain = X
    self.y_train = y

  def predict(self, X):
    self.original_xtest = X
    self.x_test = self.pca.transform(self.original_xtest)
    self.y_pred = self.clf.predict(self.x_test)

  def evaluate(self, y):
    self.y_test = y
    accuracy = accuracy_score(y, self.y_pred)

    return accuracy

  def visualize_error(self,y):
    incorrect_indices = np.where(self.y_pred != y.flatten())[0]

    if len(incorrect_indices) > 0:
        print("\n\tVisualizing images with wrong labels:")
        plt.figure(figsize=(12, 20))
        for i, idx in enumerate(incorrect_indices[:min(25, len(incorrect_indices))]):
            print(idx)
            img1 = self.x_test[idx]

            img1 = self.pca.inverse_transform(img1)
            img1 = img1.reshape((112, 92))
            img2 = self.original_xtest[idx]
            img2 = img2.reshape((112, 92))

            pred_idx = (self.y_pred[idx] - 1) * 2
            img3 = self.original_xtest[pred_idx]
            img3 = img3.reshape((112, 92))

            img4 = self.x_test[pred_idx]
            img4 = self.pca.inverse_transform(img4)
            img4 = img4.reshape((112, 92))



            plt.subplot(10,4, i * 4 + 1)
            plt.imshow(img1, cmap='gray')
            plt.title('Test Eigenface Projection', size = 10)
            plt.xticks(())
            plt.yticks(())

            plt.subplot(10, 4, i * 4 + 2)
            plt.imshow(img4, cmap='gray')
            plt.title(f'Predict Eigenface Projection', size = 10)
            plt.xticks(())
            plt.yticks(())

            plt.subplot(10, 4, i * 4 + 3)
            plt.imshow(img2, cmap='gray')
            plt.title(f'Test Image-Label {int(self.y_test[idx])}', size = 10)
            plt.xticks(())
            plt.yticks(())

            plt.subplot(10, 4, i * 4 + 4)
            plt.imshow(img3, cmap='gray')
            plt.title(f'Predicted Image-Label {38}', size = 10)
            plt.xticks(())
            plt.yticks(())

        plt.tight_layout()
        plt.show()

  def cross_validation(self, X, y, verbose = False):
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 123)
    scores_list = []
    i = 1
    for train_idx, test_idx in skf.split(X, y):


      X_train, X_test = X[train_idx], X[test_idx]
      y_train, y_test = y[train_idx], y[test_idx]

      if self.tunning:
        self.hyperparameter_tuning(X_train, y_train)

      self.fit(X_train, y_train)
      self.predict(X_test)

      scores = self.evaluate(y_test)
      scores_list.append(scores)

      if verbose:
        print(f'Fold {i}:')
        self.visualize_error(y_test)

      i+=1

    return np.array(scores_list)

  def hyperparameter_tuning(self, X, y):
    X = self.pca.fit_transform(X)
    kf = StratifiedKFold(n_splits=5, shuffle = True, random_state= 64)
    grid_search = GridSearchCV(self.clf, param_grid=self.param_grid, cv=kf,scoring='accuracy')
    grid_search.fit(X, y)

    self.clf = grid_search.best_estimator_
    print("\tBest hyperparameters found: ", grid_search.best_params_)

  def classification_report(self, X, y, verbose = False):
    vb = verbose
    scores = self.cross_validation(X,y,vb)
    print(scores)
    average_scores = np.mean(scores)
    # print("\tClassifier:", self.clf)
    print("\tHyperparameters:", self.clf.get_params())
    # print("\nClassification Report:")
    print(f'Accuracy: {average_scores:.4f}\n')