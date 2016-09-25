"""
This contains all the data that could be used in synthetic_experiment_boost.py
"""

import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, "~/.lib/scikit-learn/sklearn")

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron

from sklearn.datasets import make_gaussian_quantiles
from sklearn.svm import SVC
from RankBoost_nondistributed import RankBoost
from martiboost_nondistributed import MartiBoost
from MIBoosting_Xu_nondistributed import MIBoosting_Xu
from RankBoost_m3_nondistributed import RankBoost_m3
from RankBoost_modiII_nondistributed import RankBoost_modiII
from rBoost_nondistributed import RBoost

from mi_svm import SVM
import data
import dill


def getDataset0():
 	"""
	Construct dataset v0
	
	"""
	X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=200, n_features=2,
                                 n_classes=2, random_state=1)

	X = X1
	y = y1
	#import pdb;pdb.set_trace()
	return X, y

def getDataset1():
	"""
	Construct dataset v1
	construct two gaussians
	"""
	X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=200, n_features=2,
                                 n_classes=2, random_state=1)
	X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                 n_samples=300, n_features=2,
                                 n_classes=2, random_state=1)
	X = np.concatenate((X1, X2))
	y = np.concatenate((y1, - y2 + 1))

	f0_max = np.max( abs(X)[:,0] ) #scale the data to be within the unit box
	f1_max = np.max( abs(X)[:,1] )
	#import pdb;pdb.set_trace()
	X = np.vstack((X[:,0]/f0_max, X[:,1]/f1_max )).transpose()
	return X, y

def getDataset2():

	# Construct dataset v2
	X=np.array([[0,0],[0,1],[0,-1], [1, 0], [1, 1], [1, -1], [-1, 0], [2, 0], [0, 3], [1,3], [0, -3], [1,-3]])
	y=np.array([-1,-1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1 ])
	return X, y


def getDataset3():

	# Construct dataset v3
	X=np.array([[1,0],[-2,0],[0, -2], [0, 2]])
	y=np.array([1,1, -1, -1])
	#import pdb;pdb.set_trace()
	return X, y

def getDataset4():
	# construct dataset v4 -- banana~goldmedal
	pkl_file = open('banana_goldmedal.pkl', 'rb')
	train_class = dill.load(pkl_file)
	test_class = dill.load(pkl_file)
	X = train_class.instances
	y= (train_class.instance_labels_SIL >0)+0 #the boolean values 
	return X, y

def getDataset5():
	# construct dataset v5 -- musk1
	pkl_file = open('musk1.pkl', 'rb')
	train_class = dill.load(pkl_file)
	test_class = dill.load(pkl_file)
	X = train_class.instances
	y= 2*train_class.instance_labels_SIL - 1 #convert the boolean values to +1/-1 values for the labels
	return X, y

def getDataset6():
	# construct dataset v6 -- musk2
	pkl_file = open('musk2.pkl', 'rb')
	train_class = dill.load(pkl_file)
	test_class = dill.load(pkl_file)
	X = train_class.instances
	y= 2*train_class.instance_labels_SIL - 1 #convert the boolean values to +1/-1 values for the labels
	return X, y

def getDataset7(noise_rate = None):
	"""
	Construct dataset v1
	construct two gaussians
	"""
	if noise_rate is None:
		noise_rate = 0.45

	X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=2000, n_features=2,
                                 n_classes=1, random_state=1, shuffle = True)
	X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                 n_samples=2000, n_features=2,
                                 n_classes=1, random_state=1)
	y1_noised = []
	for i in range(y1.shape[0]):
		if np.random.uniform() < noise_rate:
			y1_noised.append( - y1[i] + 1)
		else:
			y1_noised.append( y1[i] )

	y1_noised = np.array(y1_noised)

	X = np.concatenate((X1, X2))
	y = np.concatenate((y1_noised, - y2 + 1))

	y_denoised = np.concatenate((y1, - y2 + 1))


	f0_max = np.max( abs(X)[:,0] ) #scale the data to be within the unit box
	f1_max = np.max( abs(X)[:,1] )
	#import pdb;pdb.set_trace()
	X = np.vstack((X[:,0]/f0_max, X[:,1]/f1_max )).transpose()
	return X, y, y_denoised

def getDataset8(noise_rate = None):
	"""
	construct synthetic dataset which looks like multiple instance learning dataset
	We use two almost-nonoverlapping gaussians to represent positive and negative classes
	the instances from negative class will be flipped to positive label with probability noise_rate (Note that noise-rate will never exceed 50% in real MI datasets)
	Add neccesssary number of instances from  positive class to ensure equal number of pos-labeled instances and neg-labeled instances
	"""
	if noise_rate is None:
		noise_rate = 0.3

	num_inst_per_side_prior_flipping = 300
	X1, y1 = make_gaussian_quantiles(cov=5.,
                                 n_samples=num_inst_per_side_prior_flipping, n_features=2,
                                 n_classes=1, random_state=1, shuffle = True)
	
	y1_noised = []
	num_flipped = 0 # number of instances in y1 whose labeled is flipped
	for i in range(y1.shape[0]):
		if np.random.uniform() < noise_rate:
			num_flipped += 1
			y1_noised.append( - y1[i] + 1)
		else:
			y1_noised.append( y1[i] )

	y1_noised = np.array(y1_noised)

	
	X2, y2 = make_gaussian_quantiles(mean=(10, 10), cov=5,
                                 n_samples=num_inst_per_side_prior_flipping - num_flipped , n_features=2,
                                 n_classes=1, random_state=1)

	X = np.concatenate((X1, X2))
	y = np.concatenate((y1_noised, - y2 + 1))

	y_denoised = np.concatenate((y1, - y2 + 1))

	"""
	f0_max = np.max( abs(X)[:,0] ) #scale the data to be within the unit box
	f1_max = np.max( abs(X)[:,1] )
	#import pdb;pdb.set_trace()
	X = np.vstack((X[:,0]/f0_max, X[:,1]/f1_max )).transpose()
	"""
	return X, y, y_denoised


def getDataset9(noise_rate = None):
	"""
	Increase the dimension of the space

	construct synthetic dataset which looks like multiple instance learning dataset
	We use two almost-nonoverlapping gaussians to represent positive and negative classes
	the instances from negative class will be flipped to positive label with probability noise_rate (Note that noise-rate will never exceed 50% in real MI datasets)
	Add neccesssary number of instances from  positive class to ensure equal number of pos-labeled instances and neg-labeled instances
	"""
	if noise_rate is None:
		noise_rate = 0.3
	n_f = 10

	num_inst_per_side_prior_flipping = 4000
	X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=num_inst_per_side_prior_flipping, n_features=n_f,
                                 n_classes=1, random_state=1, shuffle = True)
	
	y1_noised = []
	num_flipped = 0 # number of instances in y1 whose labeled is flipped
	for i in range(y1.shape[0]):
		if np.random.uniform() < noise_rate:
			num_flipped += 1
			y1_noised.append( - y1[i] + 1)
		else:
			y1_noised.append( y1[i] )

	y1_noised = np.array(y1_noised)

	
	X2, y2 = make_gaussian_quantiles(mean=([5]*n_f), cov=1.5,
                                 n_samples=num_inst_per_side_prior_flipping - num_flipped , n_features=n_f,
                                 n_classes=1, random_state=1)

	X = np.concatenate((X1, X2))
	y = np.concatenate((y1_noised, - y2 + 1))

	y_denoised = np.concatenate((y1, - y2 + 1))

	#import pdb;pdb.set_trace()
	f_max = {}
	for i in range(n_f):
		f_max[i] = np.max( abs(X)[:,i] ) #scale the data to be within the unit box
	#f1_max = np.max( abs(X)[:,1] )
	#import pdb;pdb.set_trace()
	X = np.vstack(([ X[:,i]/f_max[i] for i in range(n_f) ])).transpose()
	return X, y, y_denoised


def getDataset10():
 	"""
	Construct dataset v0
	
	"""
	X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=50, n_features=2,
                                 n_classes=2, random_state=1)

	X = X1
	y = y1
	#import pdb;pdb.set_trace()
	return X, y

def getDataset11(noise_rate = None):
	"""
	two-sided noise dataset
	construct synthetic dataset which looks like multiple instance learning dataset
	We use two almost-nonoverlapping gaussians to represent positive and negative classes
	the instances from any class will be flipped to another label with probability noise_rate (Note that noise-rate will never exceed 50% in real MI datasets)

	"""
	if noise_rate is None:
		noise_rate = 0.3

	num_inst_per_side_prior_flipping = 300
	X1, y1 = make_gaussian_quantiles(cov=5.,
                                 n_samples=num_inst_per_side_prior_flipping, n_features=2,
                                 n_classes=1, random_state=1, shuffle = True)
	
	y1_noised = []
	num_flipped = 0 # number of instances in y1 whose labeled is flipped
	for i in range(y1.shape[0]):
		if np.random.uniform() < noise_rate:
			num_flipped += 1
			y1_noised.append( - y1[i] + 1)
		else:
			y1_noised.append( y1[i] )

	y1_noised = np.array(y1_noised)
	
	X2, y2 = make_gaussian_quantiles(mean=(10, 10), cov=5,
                                 n_samples=num_inst_per_side_prior_flipping , n_features=2,
                                 n_classes=1, random_state=1)

	y2_noised = []
	num_flipped = 0 # number of instances in y1 whose labeled is flipped
	for i in range(y2.shape[0]):
		if np.random.uniform() < noise_rate:
			num_flipped += 1
			y2_noised.append( y2[i] )
		else:
			y2_noised.append( - y2[i] + 1)

	y2_noised = np.array(y2_noised)

	X = np.concatenate((X1, X2))
	y = np.concatenate((y1_noised, y2_noised))

	y_denoised = np.concatenate((y1, - y2 + 1))

	"""
	f0_max = np.max( abs(X)[:,0] ) #scale the data to be within the unit box
	f1_max = np.max( abs(X)[:,1] )
	#import pdb;pdb.set_trace()
	X = np.vstack((X[:,0]/f0_max, X[:,1]/f1_max )).transpose()
	"""
	return X, y, y_denoised

def getDataset12(noise_rate = None):
	"""
	one-sided noise, noise rate 0.3, Gaussian Covariance 10*I (I is identity matrix), Gaussian Center (0, 0) and (20, 0) to make sure they are linearly separable
	the number of instances with observed positive labels is the same with the number of instances with observed negative labels (balanced w.r.t. observed labels)

	construct synthetic dataset which looks like multiple instance learning dataset
	We use two almost-nonoverlapping gaussians to represent positive and negative classes. make them linearly separable.
	the instances from negative class will be flipped to positive label with probability noise_rate (Note that noise-rate will never exceed 50% in real MI datasets)
	Add neccesssary number of instances from  positive class to ensure equal number of pos-labeled instances and neg-labeled instances
	"""
	if noise_rate is None:
		noise_rate = 0.3

	num_inst_per_side_prior_flipping = 300
	X1, y1 = make_gaussian_quantiles(cov=10.,
                                 n_samples=num_inst_per_side_prior_flipping, n_features=2,
                                 n_classes=1, random_state=1, shuffle = True)
	
	y1_noised = []
	num_flipped = 0 # number of instances in y1 whose labeled is flipped
	for i in range(y1.shape[0]):
		if np.random.uniform() < noise_rate:
			num_flipped += 1
			y1_noised.append( - y1[i] + 1)
		else:
			y1_noised.append( y1[i] )

	y1_noised = np.array(y1_noised)

	
	X2, y2 = make_gaussian_quantiles(mean=(20, 0), cov=10,
                                 n_samples=num_inst_per_side_prior_flipping - num_flipped , n_features=2,
                                 n_classes=1, random_state=1)

	X = np.concatenate((X1, X2))
	y = np.concatenate((y1_noised, - y2 + 1))

	y_denoised = np.concatenate((y1, - y2 + 1))

	"""
	f0_max = np.max( abs(X)[:,0] ) #scale the data to be within the unit box
	f1_max = np.max( abs(X)[:,1] )
	#import pdb;pdb.set_trace()
	X = np.vstack((X[:,0]/f0_max, X[:,1]/f1_max )).transpose()
	"""
	return X, y, y_denoised


def getDataset13(noise_rate = None):
	"""
	two-sided noise dataset
	construct synthetic dataset which looks like multiple instance learning dataset
	We use two almost-nonoverlapping gaussians to represent positive and negative classes
	the instances from any class will be flipped to another label with probability noise_rate (Note that noise-rate will never exceed 50% in real MI datasets)

	"""
	if noise_rate is None:
		noise_rate = 0.3

	num_inst_per_side_prior_flipping = 300
	X1, y1 = make_gaussian_quantiles(cov=10.,
                                 n_samples=num_inst_per_side_prior_flipping, n_features=2,
                                 n_classes=1, random_state=1, shuffle = True)
	
	y1_noised = []
	num_flipped = 0 # number of instances in y1 whose labeled is flipped
	for i in range(y1.shape[0]):
		if np.random.uniform() < noise_rate:
			num_flipped += 1
			y1_noised.append( - y1[i] + 1)
		else:
			y1_noised.append( y1[i] )

	y1_noised = np.array(y1_noised)
	
	X2, y2 = make_gaussian_quantiles(mean=(20, 0), cov=10,
                                 n_samples=num_inst_per_side_prior_flipping , n_features=2,
                                 n_classes=1, random_state=1)

	y2_noised = []
	num_flipped = 0 # number of instances in y1 whose labeled is flipped
	for i in range(y2.shape[0]):
		if np.random.uniform() < noise_rate:
			num_flipped += 1
			y2_noised.append( y2[i] )
		else:
			y2_noised.append( - y2[i] + 1)

	y2_noised = np.array(y2_noised)

	X = np.concatenate((X1, X2))
	y = np.concatenate((y1_noised, y2_noised))

	y_denoised = np.concatenate((y1, - y2 + 1))

	"""
	f0_max = np.max( abs(X)[:,0] ) #scale the data to be within the unit box
	f1_max = np.max( abs(X)[:,1] )
	#import pdb;pdb.set_trace()
	X = np.vstack((X[:,0]/f0_max, X[:,1]/f1_max )).transpose()
	"""
	return X, y, y_denoised

def getDataset14(noise_rate = None):
	"""
	one-sided noise, noise rate 0.3, Gaussian Covariance 10*I (I is identity matrix), Gaussian Center (0, 0) and (20, 0) to make sure they are linearly separable
	the number of instances with observed positive labels is not the same with the number of instances with observed negative labels (balanced w.r.t. observed labels)

	construct synthetic dataset which looks like multiple instance learning dataset
	We use two almost-nonoverlapping gaussians to represent positive and negative classes. make them linearly separable.
	the instances from negative class will be flipped to positive label with probability noise_rate (Note that noise-rate will never exceed 50% in real MI datasets)
	Add neccesssary number of instances from  positive class to ensure equal number of pos-labeled instances and neg-labeled instances
	"""
	if noise_rate is None:
		noise_rate = 0.3

	num_inst_per_side_prior_flipping = 300
	X1, y1 = make_gaussian_quantiles(cov=10.,
                                 n_samples=num_inst_per_side_prior_flipping, n_features=2,
                                 n_classes=1, random_state=1, shuffle = True)
	
	y1_noised = []
	num_flipped = 0 # number of instances in y1 whose labeled is flipped
	for i in range(y1.shape[0]):
		if np.random.uniform() < noise_rate:
			num_flipped += 1
			y1_noised.append( - y1[i] + 1)
		else:
			y1_noised.append( y1[i] )

	y1_noised = np.array(y1_noised)

	
	X2, y2 = make_gaussian_quantiles(mean=(20, 0), cov=10,
                                 n_samples=5000 , n_features=2,
                                 n_classes=1, random_state=1)

	X = np.concatenate((X1, X2))
	y = np.concatenate((y1_noised, - y2 + 1))

	y_denoised = np.concatenate((y1, - y2 + 1))

	"""
	f0_max = np.max( abs(X)[:,0] ) #scale the data to be within the unit box
	f1_max = np.max( abs(X)[:,1] )
	#import pdb;pdb.set_trace()
	X = np.vstack((X[:,0]/f0_max, X[:,1]/f1_max )).transpose()
	"""
	return X, y, y_denoised


