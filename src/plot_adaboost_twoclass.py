"""
==================
Two-class AdaBoost
==================

This example fits an AdaBoosted decision stump on a non-linearly separable
classification dataset composed of two "Gaussian quantiles" clusters
(see :func:`sklearn.datasets.make_gaussian_quantiles`) and plots the decision
boundary and decision scores. The distributions of decision scores are shown
separately for samples of class A and B. The predicted class label for each
sample is determined by the sign of the decision score. Samples with decision
scores greater than zero are classified as B, and are otherwise classified
as A. The magnitude of a decision score determines the degree of likeness with
the predicted class label. Additionally, a new dataset could be constructed
containing a desired purity of class B, for example, by only selecting samples
with a decision score above some value.

"""
print(__doc__)

# Author: Noel Dawe <noel.dawe@gmail.com>
#
# License: BSD 3 clause

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

def plot_fig(classifier):

	plot_colors = "br"
	plot_step = 0.02
	class_names = "AB"

	plt.figure(figsize=(10, 5))

	# Plot the decision boundaries
	print "Plot the decision boundaries"
	plt.subplot(121)
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

	Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
	plt.axis("tight")

	# Plot the training points
	print "Plot the training points"
	for i, n, c in zip(range(2), class_names, plot_colors):
    		idx = np.where(y == i)
    		plt.scatter(X[idx, 0], X[idx, 1],
                	c=c, cmap=plt.cm.Paired,
                	label="Class %s" % n)
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.legend(loc='upper right')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Decision Boundary')
	plt.savefig('Adaboost_twoclass.pdf')

def get_bag_label(instance_predictions, bags):
	num_bag = len(bags)
	p_index= 0
	bag_predictions = []
	for bag_index in range(num_bag):
		n_index =p_index+ bags[bag_index].shape[0]
		
		bag_predictions.append( np.average(instance_predictions[p_index: n_index]) )
		p_index = n_index
	return np.array(bag_predictions)


#import pdb;pdb.set_trace()



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



def getDataset(index):
	hashmap_dataset = {
		0: getDataset0,
		1: getDataset1,
		2: getDataset2,
		4: getDataset4, #banana~goldmedal
		7: getDataset7,
		8: getDataset8,
		9: getDataset9,
		11: getDataset11,
		12: getDataset12,
		13: getDataset13,
		14: getDataset14,
	}
	return hashmap_dataset[index]
		

def getMethod1():
	#self-made Adaboost + decision stump
	from Adaboost_nondistributed import AdaBoost
	param = {"max_iter_boosting": 200}
	bdt = AdaBoost(**param)
	return bdt

def getMethod2():
	#AdaBoosted decision tree
	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)
	return bdt

def getMethod3():
	#linear svm+adaboost
	params = {'C': 100000000, 'kernel': 'linear'}
	bdt = AdaBoostClassifier(SVC(**params),
                         algorithm="SAMME",
                         n_estimators=2)
	return bdt

def getMethod4():
	#rbf svm+adaboost
	params = {'C': 10000, 'kernel': 'rbf','gamma':1000}
	bdt = AdaBoostClassifier(SVC(**params),
                         algorithm="SAMME",
                         n_estimators=15)
	return bdt

def getMethod5():
	#rankboost
	params = {'C': 10, 'kernel': 'linear', 'max_iter_boosting':10}
	bdt = RankBoost(**params)
	return bdt

def getMethod6():
	#linear svm
	params = {'C': 10, 'kernel': 'linear'}
	bdt = SVC(**params)
	return bdt

def getMethod7():
	#rbf svm
	params = {'C': 10, 'kernel': 'rbf', 'gamma': 10000}
	bdt = SVC(**params)
	return bdt


def getMethod8():
	#martiboost + linear svm
	params = {'C': 10, 'kernel': 'linear'}
	bdt = MartiBoost(**params)
	return bdt

def getMethod9():
	#martiboost + balanced_decision_stump
	params = {'weak_classifier': 'dtree_stump_balanced', 'max_iter_boosting': 200}
	bdt = MartiBoost(**params)
	return bdt

def getMethod10():
	#MIBoosting + decision_stump
	params = {'weak_classifier': 'dtree_stump','max_depth': 1,'max_iter_boosting': 200}
	bdt1 = MIBoosting_Xu(**params)
	return bdt


def getMethod11():
	#rankboost_m3
	params = {'weak_classifier': 'dtree_stump','max_depth': 1,'max_iter_boosting': 2000}
	bdt = RankBoost_m3(**params)
	return bdt

def getMethod12():
	#rankboost + decision stump
	params = {'weak_classifier': 'dtree_stump','max_depth': 1,'max_iter_boosting': 20}
	bdt = RankBoost(**params)
	return bdt


def getMethod13():
	#rankboost_modiII + decision stump
	params = {'weak_classifier': 'dtree_stump','max_depth': 1,'max_iter_boosting': 20}
	bdt = RankBoost_modiII(**params)
	return bdt

def getMethod14():
	#rboost + decision stump
	params = {'weak_classifier': 'dtree_stump','max_depth': 1,'max_iter_boosting': 20}
	bdt = RBoost(**params)
	return bdt

def getMethod15():
	#AdaBoosted Perceptron
	bdt = AdaBoostClassifier(Perceptron(penalty = 'l1', n_iter = 50),
                         algorithm="SAMME",
                         n_estimators=200)
	return bdt

def getMethod(index):
	hashmap_method = {
		1: getMethod1, #self-implemented AdaBoost + decision stump
		2: getMethod2, #AdaBoost+ decision stump
		3: getMethod3, #AdaBoost+ linear SVM
		14: getMethod14,  #rBoost+decision stump
		7: getMethod7, #rbf svm
		15: getMethod15, #Adaboost+perceptron
	}
	return hashmap_method[index]

def run_experiment():
	


	#X, y = getDataset(8)()
	X = array([[-1.05664139,  2.39349225],
       		[-0.24385031, -1.24147928],
       		[ 1.60036563,  2.14934555],
       		[-0.31441947, -0.28391478],
       		[ 0.42450494, -0.49815651],
       		[-2.04228533, -0.71342247],
       		[-0.17379302, -1.32337783],
       		[ 3.09087038, -1.97494406],
       		[ 2.29717124, -0.86515422],
       		[-0.8768136 ,  0.98716637],
       		[ 0.59891146,  0.10937537],
       		[ 2.06773287, -2.91347893],
       		[-0.53073307, -0.9033012 ],
       		[-0.27129644, -1.25529692],
       		[ 0.22632659,  1.23908997],
       		[ 0.26383765,  0.5799006 ],
       		[ 0.45118942, -0.35266297],
       		[ 0.69086899, -0.10687454],
       		[-0.37885096,  0.75003589],
       		[ 1.27504182,  0.71063431],
       		[ 2.34731475,  1.04940892],
       		[ 0.07185302, -0.90084788],
       		[-1.0668798 ,  1.77182314],
       		[ 1.2740027 , -0.96693721],
       		[ 1.69552593,  0.26185072],
       		[-1.55651057,  1.6188838 ],
       		[-0.9484597 ,  0.53395583],
       		[ 0.40388147,  1.25177864],
       		[ 0.26999527,  2.9702093 ],
       		[-0.29542106,  0.82961047],
       		[ 0.28043815,  0.16830364],
       		[ 0.05969925,  0.82422518],
       		[-0.94928538, -0.01791045],
       		[ 0.16993042,  0.87285701],
       		[ 1.22387121, -3.25486724],
       		[ 1.60339212, -1.55548115],
       		[ 0.44637522, -2.85982439],
       		[ 1.18650172,  1.31677719],
       		[-0.45596678, -0.54313488],
       		[ 0.72539231, -0.42156693],
       		[ 0.17228129,  1.59733146],
       		[ 2.46753646, -1.07650912],
       		[-1.58011545,  0.33151386],
       		[ 0.3254031 ,  1.07764655],
       		[-0.97180895, -1.19530128],
       		[-1.61576473, -0.49404522],
       		[-0.97815602, -0.56109422],
       		[-0.43303787,  1.17093297],
       		[-0.48628253,  0.06165527],
       		[-0.74694766, -1.51740678]])

	y = array([1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
       		1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1,
       		0, 0, 0, 1])

	bdt = getMethod(14)()
	bdt.fit(X, y)
	print np.average((bdt.predict(X)>0)== (y>0))
	import pdb;pdb.set_trace()


def run_experiment_with_two_sided_noise():
	"""
	manually add different levels of noise

	This function is used to test the reaction of algorithms to different noise density
	"""
	print "this is the beginning"
	noise_rates =[x/10.0 for x in range(0, 5)]
	accuracy = []
	accuracy_false_label = []
	for val in noise_rates:

		X, y, y_true = getDataset(11)(val)
		bdt = getMethod(2)()


		#import pdb;pdb.set_trace()
		#print "fitting the training set"
		bdt.fit(X, y)

		if abs(val - 0.4)<0.0001:
			filename = 'decision_bounary_nr_4.pdf'
			#plotDecisionBoundary(bdt, X, y, filename, y_true)
		if abs(val - 0.3)<0.0001:
			#import pdb;pdb.set_trace()
			filename = 'decision_bounary_nr_3.pdf'
			plotDecisionBoundary(bdt, X, y, filename, y_true)
			#import pdb;pdb.set_trace()

		if abs(val - 0.1)<0.0001:
			filename = 'decision_bounary_nr_1.pdf'
			#plotDecisionBoundary(bdt, X, y, filename, y_true)
		accuracy.append( np.average(bdt.predict(X) == y_true) )
		accuracy_false_label.append(np.average(bdt.predict(X) == y)   )
	plt.figure()
	plt.plot(noise_rates, accuracy, 'r.-')
	plt.plot(noise_rates, accuracy_false_label, 'b.-')
	plt.legend(["w.r.t true label", "w.r.t noisy label"])
	plt.xlabel("noise_rate")
	plt.ylabel("accuracy")
	plt.savefig("noise_rate.pdf")
	#import pdb;pdb.set_trace()



def run_experiment_with_MIL_show_changing_boosting_round(run_ID):
	"""
	show figure of accuracy vs boosting round
	manually add a fixed level of noise

	This function is used to test the reaction of algorithms to different noise density
	"""
	print "this is the beginning"

	accuracy = []
	accuracy_false_label = []

	X, y= getDataset(4)() #banana~goldmedal

	bdt = getMethod(2)() #AdaBoost+ decision stump
	
	#import pdb;pdb.set_trace()
	#print "fitting the training set"
	bdt.fit(X, y)
	#import pdb;pdb.set_trace()
	actual_round = len(bdt.estimators_)
	for predictions in bdt.staged_predict(X):
		#predictions = (bdt.predict_train(iter = round, getInstPrediction = True)>0)+0

		accuracy_false_label.append(np.average(predictions == y)   )
	plt.figure()
	plt.plot(range(actual_round), accuracy_false_label, 'b.-')

	plt.xlabel("boosting round")
	plt.ylabel("accuracy")
	plt.savefig("MIL"+str(run_ID)+"_boosting_round.pdf")
	
	alpha = bdt.estimator_weights_
	

	plt.figure()
	plt.plot(range(actual_round), alpha, 'r.-')
	plt.xlabel('boosting round')
	plt.ylabel(r'$\alpha$')
	plt.savefig("MIL"+str(run_ID)+"_alpha.pdf")
	
	import pdb;pdb.set_trace()



def run_experiment_with_two_sided_noise_show_changing_boosting_round(run_ID):
	"""
	show figure of accuracy vs boosting round
	manually add a fixed level of noise

	This function is used to test the reaction of algorithms to different noise density
	"""
	print "this is the beginning"
	#noise_rates =[x/10.0 for x in range(0, 5)]
	noise_rate = 0.3
	accuracy = []
	accuracy_false_label = []

	#X, y, y_true = getDataset(11)(noise_rate)
	X, y, y_true = getDataset(13)(noise_rate)

	#bdt = getMethod(2)()
	bdt = getMethod(15)()
	#bdt = getMethod(7)() #rbf svm
	
	#import pdb;pdb.set_trace()
	#print "fitting the training set"
	bdt.fit(X, y)
	#import pdb;pdb.set_trace()
	actual_round = len(bdt.estimators_)
	for predictions in bdt.staged_predict(X):
		#predictions = (bdt.predict_train(iter = round, getInstPrediction = True)>0)+0
		accuracy.append( np.average(predictions == y_true) )
		accuracy_false_label.append(np.average(predictions == y)   )
	plt.figure()
	plt.plot(range(actual_round), accuracy, 'r.-')
	plt.plot(range(actual_round), accuracy_false_label, 'b.-')
	plt.legend(["w.r.t true label", "w.r.t noisy label"])
	plt.xlabel("boosting round")
	plt.ylabel("accuracy")
	plt.savefig("SYN"+str(run_ID)+"_boosting_round.pdf")

	filename = "SYN"+str(run_ID)+"_decision_boundary.pdf"
	plotDecisionBoundary(bdt, X, y, filename, y_true)
	
	alpha = bdt.estimator_weights_
	

	plt.figure()
	plt.plot(range(actual_round), alpha, 'r.-')
	plt.xlabel('boosting round')
	plt.ylabel(r'$\alpha$')
	plt.savefig("SYN"+str(run_ID)+"_alpha.pdf")
	#import pdb;pdb.set_trace()
	index_TN =  [x for x in range(len(y)) if y_true[x]==0 and y[x]==0]
	index_FP =  [x for x in range(len(y)) if y_true[x]==0 and y[x]==1]
	index_TP = [x for x in range(len(y)) if y_true[x]==1 and y[x] == 1]
	index_FN = [x for x in range(len(y)) if y_true[x]==1 and y[x] == 0]


	
	D_TN =  [np.average([x[i] for i in index_TN ]) for x in bdt.sample_weights_]
	D_FP =  [np.average([x[i] for i in index_FP ]) for x in bdt.sample_weights_]
	D_TP =  [np.average([x[i] for i in index_TP ]) for x in bdt.sample_weights_]
	D_FN =  [np.average([x[i] for i in index_FN ]) for x in bdt.sample_weights_]

	plt.figure()
	plt.plot(range(actual_round), D_TN, 'r.-')
	plt.plot(range(actual_round), D_FP, 'b.-')
	plt.plot(range(actual_round), D_TP, 'k.-')
	plt.plot(range(actual_round), D_FN, 'c.-')

	plt.legend(['$D^{--}$','$D^{-+}$','$D^{++}$', '$D^{+-}$'])
	plt.xlabel("boosting round")
	plt.ylabel("distribution weight")
	plt.savefig("SYN"+str(run_ID)+"_distribution_weight.pdf")
	#import pdb;pdb.set_trace()



def run_experiment_with_one_sided_noise_show_changing_boosting_round(run_ID):
	"""
	show figure of accuracy vs boosting round
	manually add a fixed level of noise

	This function is used to test the reaction of algorithms to different noise density
	"""
	print "this is the beginning"
	#noise_rates =[x/10.0 for x in range(0, 5)]
	noise_rate = 0.3
	accuracy = []
	accuracy_false_label = []

	#X, y, y_true = getDataset(8)(noise_rate)
	X, y, y_true = getDataset(12)(noise_rate)
	#X, y, y_true = getDataset(14)(noise_rate)

	bdt = getMethod(2)()
	#bdt = getMethod(15)()
	#bdt = getMethod(7)() #rbf svm
	
	#import pdb;pdb.set_trace()
	#print "fitting the training set"
	bdt.fit(X, y)
	actual_round = len(bdt.estimators_)

	#import pdb;pdb.set_trace()

	for predictions in bdt.staged_predict(X):
		#predictions = (bdt.predict_train(iter = round, getInstPrediction = True)>0)+0
		accuracy.append( np.average(predictions == y_true) )
		accuracy_false_label.append(np.average(predictions == y)   )
	plt.figure()
	plt.plot(range(actual_round), accuracy, 'r.-')
	plt.plot(range(actual_round), accuracy_false_label, 'b.-')
	plt.legend(["w.r.t true label", "w.r.t noisy label"])
	plt.xlabel("Boosting Round")
	plt.ylabel("Accuracy")
	plt.savefig("SYN"+str(run_ID)+"_boosting_round.pdf")

	filename = "SYN"+str(run_ID)+"_decision_boundary.pdf"
	plotDecisionBoundary(bdt, X, y, filename, y_true)
	
	alpha = bdt.estimator_weights_
	

	plt.figure()
	plt.plot(range(actual_round), alpha[0:actual_round], 'r.-')
	plt.xlabel('Boosting Round')
	plt.ylabel(r'$\alpha$')
	plt.savefig("SYN"+str(run_ID)+"_alpha.pdf")
	#import pdb;pdb.set_trace()
	index_TN =  [x for x in range(len(y)) if y_true[x]==0 and y[x]==0]
	index_FP =  [x for x in range(len(y)) if y_true[x]==0 and y[x]==1]
	index_TP = [x for x in range(len(y)) if y_true[x]==1]

	
	D_TN =  [np.average([x[i] for i in index_TN ]) for x in bdt.sample_weights_]
	D_FP =  [np.average([x[i] for i in index_FP ]) for x in bdt.sample_weights_]
	D_TP =  [np.average([x[i] for i in index_TP ]) for x in bdt.sample_weights_]

	plt.figure()
	plt.plot(range(actual_round), D_TN[0:actual_round], 'r.-')
	plt.plot(range(actual_round), D_FP[0:actual_round], 'b.-')
	plt.plot(range(actual_round), D_TP[0:actual_round], 'k.-')
	plt.legend(['$D^{--}$','$D^{-+}$','$D^{+}$'])
	plt.xlabel("Boosting Round")
	plt.ylabel("Distribution Weight")
	plt.savefig("SYN"+str(run_ID)+"_distribution_weight.pdf")
	#import pdb;pdb.set_trace()

	margin_true_positive =  [np.average([x[i] for i in index_TP ]) for x in bdt.staged_decision_function_unnormalized(X)]
	margin_true_negative =  [np.average([-x[i] for i in index_TN+index_FP ]) for x in bdt.staged_decision_function_unnormalized(X)]

	plt.figure()
	plt.plot(range(actual_round), margin_true_positive[0:actual_round],'r.-' )
	plt.plot(range(actual_round), margin_true_negative[0:actual_round],'b.-' )
	plt.legend(['True Positive', 'True Negative'], loc='upper left') #loc='upper right'
	plt.xlabel("Boosting Round")
	plt.ylabel("Margin")
	plt.savefig("SYN"+str(run_ID)+"_margin.pdf")
	#import pdb;pdb.set_trace()



def run_experiment_with_one_sided_noise():
	"""
	manually add different levels of noise

	This function is used to test the reaction of algorithms to different noise density
	"""
	print "this is the beginning"
	noise_rates =[x/10.0 for x in range(0, 5)]
	accuracy = []
	accuracy_false_label = []
	for val in noise_rates:

		X, y, y_true = getDataset(8)(val)
		bdt = getMethod(2)()
		#bdt = getMethod(7)() #rbf svm


		#import pdb;pdb.set_trace()
		#print "fitting the training set"
		bdt.fit(X, y)

		if abs(val - 0.4)<0.0001:
			filename = 'decision_bounary_nr_4.pdf'
			#plotDecisionBoundary(bdt, X, y, filename)
		if abs(val - 0.3)<0.0001:
			#import pdb;pdb.set_trace()
			filename = 'decision_bounary_nr_3.pdf'
			plotDecisionBoundary(bdt, X, y, filename)
			import pdb;pdb.set_trace()

		if abs(val - 0.1)<0.0001:
			filename = 'decision_bounary_nr_1.pdf'
			#plotDecisionBoundary(bdt, X, y, filename)
		accuracy.append( np.average(bdt.predict(X) == y_true) )
		accuracy_false_label.append(np.average(bdt.predict(X) == y)   )
	plt.figure()
	plt.plot(noise_rates, accuracy, 'r.-')
	plt.plot(noise_rates, accuracy_false_label, 'b.-')
	plt.legend(["w.r.t true label", "w.r.t noisy label"])  
	plt.xlabel("noise_rate")
	plt.ylabel("accuracy")
	plt.savefig("noise_rate.pdf")
	import pdb;pdb.set_trace()

def getResults(bdt):
	#bdt1.fit(train_class.bags, train_class.bag_labels)
	print "fitting completed"
	print "Ranking Eror Bound",
	print bdt.getRankingErrorBound()
	print "Ranking Error of Last Ranker",
	print bdt.getRankingErrorOneClassifier()

	import pdb;pdb.set_trace()
	predictions_test = bdt.predict(test_class.instances)
	bag_predictions_test = get_bag_label(predictions_test, test_class.bags)
	predictions_train = bdt.predict(train_class.instances)
	bag_predictions_train = get_bag_label(predictions_train, train_class.bags)

	print np.average((bag_predictions_test > 0) == test_class.bag_labels)
	print np.average((bag_predictions_train > 0) == train_class.bag_labels)
	print np.average( (predictions_test == 1 )== test_class.instance_labels  )
	print np.average( (predictions_train == 1 )== train_class.instance_labels  )
	print np.average( (predictions_test == 1 )== test_class.instance_labels_SIL  )
	print np.average( (predictions_train == 1 )== train_class.instance_labels_SIL  )

	print "for bdt2"
	print np.average((bdt1.predict(test_class.bags)>0 )==test_class.bag_labels)
	import pdb;pdb.set_trace()


def plotDecisionBoundary(bdt, X, y, filename, y_true = None):
	"""
	plot the decision boundary of the classifier bdt
	"""

	#preset some params
	plot_colors = "br"
	plot_step = 0.02
	class_names = [['TN', 'FP'], ['FN', 'TP']]
	markers = ['o','v']
	#plt.figure(figsize=(10, 5))
	plt.figure()

	# Plot the decision boundaries
	print "Plot the decision boundaries"
	#plt.subplot(121)
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

	Z = (bdt.predict(np.c_[xx.ravel(), yy.ravel()]) >0 )+0
	Z = Z.reshape(xx.shape)
	cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
	plt.axis("tight")

	# Plot the training points
	print "Plot the training points"
	for j in range(2): #true label
	   for i, n, c in zip(range(2), class_names, plot_colors): #observed label
    		#idx = np.where(y == i)
		idx = np.array( [ id for id in range(len(y)) if  y[id] == i and y_true[id] == j] )
		if len(idx) ==0:
			continue
  		#plt.scatter(X[idx, 0], X[idx, 1],s=10, marker = markers[j],
                #	c=c, cmap=plt.cm.Paired,
                #	label=class_names[j][i])

		plt.scatter(X[idx, 0], X[idx, 1],s=25, marker = markers[j],
                	c=c, cmap=plt.cm.Paired,
                	label=class_names[j][i])
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.legend(loc='upper right')
	plt.xlabel('x')
	plt.ylabel('y')
	#plt.title('Decision Boundary on Synthetic Dataset')
	plt.savefig(filename)

def plotTwoClassScores(bdt):
	# Plot the two-class decision scores
	
	print "Plot the two-class decision scores"
	twoclass_output = bdt.decision_function(X)
	plot_range = (twoclass_output.min(), twoclass_output.max())
	plt.subplot(122)
	for i, n, c in zip(range(2), class_names, plot_colors):
    		plt.hist(twoclass_output[y == i],
             		bins=10,
             		range=plot_range,
             		facecolor=c,
             		label='Class %s' % n,
             		alpha=.5)
	x1, x2, y1, y2 = plt.axis()
	plt.axis((x1, x2, y1, y2 * 1.2))
	plt.legend(loc='upper left')
	plt.ylabel('Samples')
	plt.xlabel('Score')
	plt.title('Decision Scores')

	plt.tight_layout()
	plt.subplots_adjust(wspace=0.35)

if __name__ == "__main__":
	#run_experiment_with_one_sided_noise()
	#run_experiment_with_two_sided_noise()
	#run_experiment()
	
	#run_experiment_with_MIL_show_changing_boosting_round(0)
	
	for i in range(50):
		print "i: ",i
		run_experiment_with_one_sided_noise_show_changing_boosting_round(i)
		#run_experiment_with_two_sided_noise_show_changing_boosting_round(i)
	
	
	
