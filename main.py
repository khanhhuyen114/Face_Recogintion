import numpy as np
from time import time
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import mahalanobis
from colorama import Fore, Style

class FaceRecognitionModel:
    def __init__(self, data_path='/kaggle/input/images/Images', n_components=50):
        self.data_path = data_path
        self.n_components = n_components
        self.images = []
        self.labels = []
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.pca = None
        self.clf = None
        self.cov_matrix = None
        self.inv_cov_matrix = None
        self.seed = None

    def load_data(self):
        for i in range(1, 41):
            for j in range(1, 11):
                file_path = os.path.join(self.data_path,'s'+str(i),str(j)+'.pgm')
                img = cv2.imread(file_path, 0)
                img = img.flatten()
                self.images.append(img)
                self.labels.append(i)

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def split_data(self, rand_state = 42):
        self.seed = rand_state
        x_train, x_test, y_train, y_test = [], [], [], []
        for i in range(1, 41):
            indices = np.where(self.labels == i)[0]
            train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state= rand_state)

            x_train.extend(self.images[train_indices])
            y_train.extend(self.labels[train_indices])

            x_test.extend(self.images[test_indices])
            y_test.extend(self.labels[test_indices])

        self.x_train, self.y_train = np.array(x_train), np.array(y_train)
        self.x_test, self.y_test = np.array(x_test), np.array(y_test)

    
    def plot_elbow(self):
        X_std = (self.x_train - np.mean(self.x_train, axis=0)) / np.std(self.x_train, axis=0)
        pca = PCA()
        pca.fit(X_std)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
        plt.title('Elbow Chart for PCA')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.show()

    def perform_pca(self):
        print(f"Extracting the top {self.n_components} eigenfaces from {self.x_train.shape[0]} faces")
        t0 = time()
        self.pca = PCA(n_components=self.n_components, svd_solver='randomized', whiten=True).fit(self.x_train)
        print("Done in %0.3fs" % (time() - t0))

    def project_on_eigenfaces(self):
        print("Projecting the input data on the eigenfaces orthonormal basis")
        t0 = time()
        self.x_train = self.pca.transform(self.x_train)
        self.x_test = self.pca.transform(self.x_test)
        print("Done in %0.3fs" % (time() - t0))
        
    def mahalabonis(self,weight = 'uniform'):
        print("-"*100)
        print(Fore.RED + f"K = {self.n_components} WITH CLASSIFIER = MAHALABONIS IN WEIGHT = {weight} AND RANDOM_SEED = {self.seed}\n" +Style.RESET_ALL)
        print("-"*100)
        print("Predicting the people names on the testing set")
        t0 = time()
        self.cov_matrix = np.cov(self.x_train, rowvar=False)
        self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        distances = np.array([[mahalanobis(test_img, train_img, self.inv_cov_matrix)\
                                      for train_img in self.x_train] for test_img in self.x_test])
        if weight.lower() == 'uniform':
            weight = 1.0
        elif weight.lower() == 'distance':
            weight = 1.0 / (distances + 1e-10)
        distances = weight * distances
        five_neighbors_dis = np.argsort(distances, axis=1)[:, :5]
        pred = np.array([np.argmax(np.bincount(self.y_train[idx])) for idx in five_neighbors_dis])
        print("Done in %0.3fs" % (time() - t0))
        # Evaluate accuracy
        accuracy = accuracy_score(self.y_test, pred)
        f_score = f1_score(self.y_test, pred, average='weighted')
        # Calculate AUC
        auc_per_class = []

        for i in range(1,41):
            # Convert the problem into a binary classification problem for each class
            binary_true = (self.y_test == i).astype(int)
            binary_predict = (pred == i).astype(int)

            # Calculate AUC for each class
            auc_class = roc_auc_score(binary_true, binary_predict)
            auc_per_class.append(auc_class)

        # Calculate the average AUC
        auc_score = np.mean(auc_per_class)
        #The overall score
        recognition_rate = (float(accuracy) +float(f_score) + float(auc_score))/3
        
        print("\nClassification Report:")
        print(f'Accuracy: {accuracy}\n')
        print(f'F1_Score: {f_score}\n')
        print(f'AUC_Score: {auc_score}\n')
        print(f'Recognition Rate: {recognition_rate}\n')
        print("-"*100)
        
    def uniform_weight(distance):
        return 1.0

    def distance_weight(distance):
        return 1.0 / (distance + 1e-10)  # Adding a small epsilon to avoid division by zero
  
    def train_classifier(self,classifier):
        print("Fitting the classifier to the training set")
        t0 = time()
        self.clf = classifier
        self.clf.fit(self.x_train, self.y_train)
        print("Done in %0.3fs" % (time() - t0))
        
    def print_classifier_info(self):
        print("Classifier:", self.clf)
        print("Hyperparameters:", self.clf.get_params())
        print("\n")
        
    def evaluate_model(self):
        print("-"*100)
        print(Fore.RED + f"K = {self.n_components} WITH CLASSIFIER = {self.clf} AND RANDOM_SEED = {self.seed}\n" +Style.RESET_ALL)
        print("-"*100)
        print("Predicting the people names on the testing set")
        t0 = time()
        y_pred = self.clf.predict(self.x_test)
        print("Done in %0.3fs" % (time() - t0))
        accuracy = accuracy_score(self.y_test, y_pred)
        f_score = f1_score(self.y_test, y_pred, average='weighted')
        # Calculate AUC
        auc_per_class = []

        for i in range(1,41):
            # Convert the problem into a binary classification problem for each class
            binary_true = (self.y_test == i).astype(int)
            binary_predict = (y_pred == i).astype(int)

            # Calculate AUC for each class
            auc_class = roc_auc_score(binary_true, binary_predict)
            auc_per_class.append(auc_class)

        # Calculate the average AUC
        auc_score = np.mean(auc_per_class)
        #The overall score
        recognition_rate = (float(accuracy) +float(f_score) + float(auc_score))/3
        
        print("\nClassification Report:")
        print(f'Accuracy: {accuracy}\n')
        print(f'F1_Score: {f_score}\n')
        print(f'AUC_Score: {auc_score}\n')
        print(f'Recognition Rate: {recognition_rate}\n')
        print("-"*100)

    def plot_eigenfaces(self):
        eigenfaces = self.pca.components_.reshape((self.n_components, 112, 92))

        plt.figure(figsize=(16, 9))
        for i in range(self.n_components):
            plt.subplot(10, 10, i + 1)
            plt.imshow(eigenfaces[i], cmap='gray')
            plt.xticks(())
            plt.yticks(())
        plt.show()