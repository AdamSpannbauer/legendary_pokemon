from __future__ import print_function
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np
from util import get_poke_xy

x_train, x_test, y_train, y_test = get_poke_xy()

#get indexes for 90/10 train/val split (stratified by y)
splits = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.1, random_state=42)

# form split
for train_index, val_index in splits:
	x_train, x_val = x_train[train_index], x_train[val_index]
	y_train, y_val = y_train[train_index], y_train[val_index]

#ks to eval
k_vals = range(1, 30, 2)
accuracies = []
 
# loop through k_vals and find best performance on val
for k in xrange(1, 30, 2):
	# train the knn with current k
	model = KNeighborsClassifier(n_neighbors=k)
	model.fit(x_train, y_train)
 
	# eval the model and update the accuracies list
	score = model.score(x_val, y_val)
	# print("k=%d, accuracy=%.2f%%" % (k, score * 100))
	accuracies.append(score)
 
# find the value of k that has the largest accuracy
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (k_vals[i],
	accuracies[i] * 100))

# build model with best k from train
model = KNeighborsClassifier(n_neighbors=k_vals[i])
model.fit(x_train, y_train)
predictions = model.predict(x_test)
 
# deploy and eval results
print("\n[RESULTS] KNN w/ k={}".format(i))
print(classification_report(y_test, predictions))
