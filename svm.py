from __future__ import print_function
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from util import get_poke_xy

#read data and split into train test
x_train, x_test, y_train, y_test = get_poke_xy()

# train linear SVM model, eval, and report results
print("[RESULTS] SVM w/ Linear Kernel")
model = SVC(kernel="linear")
model.fit(x_train, y_train)
print(classification_report(y_test, model.predict(x_test)))

# train poly kernel SVM model, eval, and report results
print("\n[RESULTS] SVM w/ Polynomial Kernel")
model = SVC(kernel="poly", degree=2, coef0=1)
model.fit(x_train, y_train)
print(classification_report(y_test, model.predict(x_test)))

