
from time import time
import numpy as np
import cv2


face_1 = np.load('face_1.npy').reshape((200,50*50*3))
face_2 = np.load('face_2.npy').reshape((200,50*50*3))
face_3 = np.load('face_3.npy').reshape((200,50*50*3))

# create numpy matrix of zeros of size 600
labels = np.zeros((600,1))
labels[:200, :] = 0.0  # first 200 for person 1
labels[200:400, :] = 1.0  # second 200 for person 2
labels[400:, :] = 2.0  # last 200 for person 3

# combine all information into 1 data array
features = np.concatenate([face_1,face_2,face_3])
print features.shape, labels.shape

# shuffle features with respect to labels
# there is no need for shuffle as train_test_split already shuffle feature set long with labels
idx = np.random.permutation(len(features))
features, labels = features[idx], labels[idx]
labels = labels.flatten()
features = features.astype(float)


###############################################################################
# apply feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features_rescaling = scaler.fit_transform(features)



###############################################################################
# Split into a training and testing set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)



###############################################################################
# Apply PCA
# till now there are total 7500 features, so now reducing them into 400
from sklearn.decomposition import PCA
pca = PCA(n_components = 400, svd_solver='randomized')
pca.fit(features_train)
features_train_pca = pca.transform(features_train)
features_test_pca = pca.transform(features_test)
print "\nweight of 400 features : \n", pca.explained_variance_ratio_
print "\nNo of original features : ", features_train.shape[1]
print "No of principal components : ", features_train_pca.shape[1]



###############################################################################
# apply feature selection
### feature selection, because input is super high dimensional and
### can be really computationally chewy as a result
from sklearn.feature_selection import SelectPercentile, f_classif
selector = SelectPercentile(f_classif, percentile=50)
selector.fit(features_train_pca, labels_train)
features_train_transformed = selector.transform(features_train_pca)
features_test_transformed  = selector.transform(features_test_pca)
print "No of features after selection :", features_train_transformed.shape[1]


###############################################################################
# Train a SVM classification model
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

print "\nFitting the classifier to the training set"
t0 = time()
param_grid = {
         'kernel' : ('linear', 'rbf'),
         'C': [1, 100, 1e3, 5e3, 1e4, 5e4, 1e5],
         'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
}

svr = SVC(class_weight='balanced')
clf = GridSearchCV(svr, param_grid)
clf.fit(features_train_transformed, labels_train)
print "done in : ", round(time() - t0,3), "s"
print "Best parameters found by grid search:"
print clf.best_params_


###############################################################################
# Quantitative evaluation of the model quality on the test set
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print "\nPredicting the people names on the testing set"
t0 = time()
pred = clf.predict(features_test_transformed)
print "done in : ", round(time() - t0,3), "s"

print "Accuracy of test set is : ", clf.score(features_test_transformed, labels_test)
print classification_report(labels_test, pred)
print confusion_matrix(labels_test, pred, labels=range(labels.shape[0]))
