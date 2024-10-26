import h5py
import numpy as np
import os
import glob
import cv2
import warnings
import matplotlib.pyplot as plt

from PIL import Image
from img2vec_pytorch import Img2Vec
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

num_trees = 100
test_size = 0.20
seed      = 42
train_path = "dataset/train"
test_path  = "dataset/test"
h5_data    = 'output/data.h5'
h5_labels  = 'output/labels.h5'
fixed_size       = tuple((500, 500))

# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()

if not os.path.exists(test_path):
    os.makedirs(test_path)

# create all the machine learning models
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
models.append(('SVM', SVC(random_state=seed)))

# variables to hold the results and names
results = []
names   = []

# import the feature vector and trained labels
h5f_data  = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string   = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels   = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started...")

# split the training and testing data
(x_train, x_test, y_train, y_test) = train_test_split(global_features, global_labels, test_size=test_size, shuffle=True, stratify=global_labels)

print("[STATUS] splitted train and test data...")
print("Train data  : {}".format(x_train.shape))
print("Test data   : {}".format(x_test.shape))
print("Train labels: {}".format(y_train.shape))
print("Test labels : {}".format(y_test.shape))

# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring= "accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#-----------------------------------
# TESTING OUR MODEL
#-----------------------------------

# create the model - Support vector classifier
clf  = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(clf, parameters)

# fit the training data to the model
grid_search.fit(x_train, y_train)

best_estimator = grid_search.best_estimator_

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
        
img2vec = Img2Vec()

# loop through the test images
for file in glob.glob(test_path + "/*.jpg"):
    
    img = pil_loader(file)
    if img.mode == 'L':
            img = img.convert('RGB')
            
    img_features = img2vec.get_vec(img)

    # scale features in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_feature = scaler.fit_transform(img_features.reshape(-1, 1))

    # predict label of test image
    prediction = best_estimator.predict(rescaled_feature.reshape(1,-1))[0]

    image = cv2.imread(file)
    image = cv2.resize(image, fixed_size)
    
    # show predicted label on image
    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    
kfold = KFold(n_splits=10, random_state=seed)
cv_results = cross_val_score(clf, x_train, y_train, cv=kfold, scoring= "accuracy")
print(cv_results.mean())

