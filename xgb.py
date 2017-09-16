from __future__ import print_function
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from util import get_poke_xy

x_train, x_test, y_train, y_test = get_poke_xy()

# grid search for selected params
cv_params = {'max_depth': [4, 6, 8, 10],
			 'subsample': [.5, .75, 1.0],
			 'colsample_bytree': [.4, .6, .8, 1.0]}
n_estimators = 300
ind_params = {'learning_rate': 2/100.0,
			  'n_estimators': n_estimators,
			  'seed':42,
              'objective': 'reg:linear'}
model = GridSearchCV(xgb.XGBClassifier(**ind_params), cv_params, 
	scoring = 'neg_mean_squared_error', cv = 5, n_jobs = -1, verbose=1) 

model.fit(x_train, y_train)

# make predictions for test data
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]

# deploy and eval results
print("\n[RESULTS] xgboost")
print(classification_report(y_test, predictions))
