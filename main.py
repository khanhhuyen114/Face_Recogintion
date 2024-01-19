import numpy as np
from time import time
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score

class FaceRecognitionModel:
    def __init__(self, classifier, data_path='att_faces', n_components=50):
        self.data_path = data_path
        self.n_components = n_components
        self.images = []
        self.labels = []
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None
        self.pca = None
        self.clf = classifier

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

    def split_data(self):
        x_train, x_test, y_train, y_test = [], [], [], []
        for i in range(1, 41):
            indices = np.where(self.labels == i)[0]
            train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

            x_train.extend(self.images[train_indices])
            y_train.extend(self.labels[train_indices])

            x_test.extend(self.images[test_indices])
            y_test.extend(self.labels[test_indices])

        self.x_train, self.y_train = np.array(x_train), np.array(y_train)
        self.x_test, self.y_test = np.array(x_test), np.array(y_test)

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

    def train_classifier(self):
        print("Fitting the classifier to the training set")
        t0 = time()
        self.clf.fit(self.x_train, self.y_train)
        print("Done in %0.3fs" % (time() - t0))
        
    def print_classifier_info(self):
        print("Classifier:", self.clf)
        print("Hyperparameters:", self.clf.get_params())
        
    def evaluate_model(self):
        print("Predicting the people names on the testing set")
        t0 = time()
        y_pred = self.clf.predict(self.x_test)
        print("Done in %0.3fs" % (time() - t0))
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        print('Accuracy:', accuracy_score(self.y_test, y_pred))
        # Print only overall stats recall and f1 score
        # print('Recall: ', recall_score(self.y_test, y_pred, average='macro'))
        # print('F1 score: ', f1_score(self.y_test, y_pred, average='macro'))

    def plot_eigenfaces(self):
        eigenfaces = self.pca.components_.reshape((self.n_components, 112, 92))

        plt.figure(figsize=(16, 9))
        for i in range(self.n_components):
            plt.subplot(10, 10, i + 1)
            plt.imshow(eigenfaces[i], cmap='gray')
            plt.xticks(())
            plt.yticks(())
        plt.show()

if __name__ == '__main__':
    clf = SVC(kernel='rbf', class_weight='balanced')
    face_model = FaceRecognitionModel(classifier=clf)
    face_model.load_data()
    face_model.split_data()
    face_model.perform_pca()
    face_model.project_on_eigenfaces()
    # face_model.plot_eigenfaces()
    face_model.print_classifier_info()
    face_model.train_classifier()
    face_model.evaluate_model()