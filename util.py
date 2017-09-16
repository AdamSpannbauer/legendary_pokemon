import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit

def get_poke_xy():
	#read in data
	data = pd.read_csv("data/Pokemon 2.csv")

	#split to x and y; convert to numpy array; convert legendary flag to int
	x = np.array(data.iloc[:,4:11])
	y = np.array(data.iloc[:,12]).astype(int)

	#get indexes for 25/75 test/train split (stratified by y)
	splits = StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=42)

	#perform split
	for train_index, test_index in splits:
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]

	return x_train, x_test, y_train, y_test
