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

sys.path.insert(0, "~/.lib/scikit-learn")

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.svm import SVC
from RankBoost_nondistributed import RankBoost
from martiboost_nondistributed import MartiBoost
from MIBoosting_Xu_nondistributed import MIBoosting_Xu

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

print "load dataset"
#import pdb;pdb.set_trace()
'''
# Construct dataset v0
X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=200, n_features=2,
                                 n_classes=2, random_state=1)

X = X1
y = y1
#import pdb;pdb.set_trace()

# Construct dataset v1

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
import pdb;pdb.set_trace()
X = np.vstack((X[:,0]/f0_max, X[:,1]/f1_max )).transpose()


# Construct dataset v2
X=np.array([[2,2],[-2,-2],[2, -2], [-2, 2]])
y=np.array([1,1, -1, -1])


# Construct dataset v3
X=np.array([[1,0],[-2,0],[0, -2], [0, 2]])
y=np.array([1,1, -1, -1])
#import pdb;pdb.set_trace()
'''

# construct dataset v4 -- banana~goldmedal
pkl_file = open('banana_goldmedal.pkl', 'rb')
train_class = dill.load(pkl_file)
test_class = dill.load(pkl_file)
X = train_class.instances
y= 2*train_class.instance_labels_SIL - 1 #convert the boolean values to +1/-1 values for the labels

'''
# construct dataset v5 -- musk1
pkl_file = open('musk1.pkl', 'rb')
train_class = dill.load(pkl_file)
test_class = dill.load(pkl_file)
X = train_class.instances
y= 2*train_class.instance_labels_SIL - 1 #convert the boolean values to +1/-1 values for the labels

# construct dataset v6 -- musk2
pkl_file = open('musk2.pkl', 'rb')
train_class = dill.load(pkl_file)
test_class = dill.load(pkl_file)
X = train_class.instances
y= 2*train_class.instance_labels_SIL - 1 #convert the boolean values to +1/-1 values for the labels
'''
'''
#Adaboost + perceptron
bdt = AdaBoostClassifier(MLPClassifier(hidden_layer_sizes = ()),
			algorithm="SAMME",
                         n_estimators=30)


#AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=30)

#linear svm+adaboost
params = {'C': 100000000, 'kernel': 'linear'}
bdt = AdaBoostClassifier(SVC(**params),
                         algorithm="SAMME",
                         n_estimators=2)



#rbf svm+adaboost
params = {'C': 10000, 'kernel': 'rbf','gamma':1000}
bdt = AdaBoostClassifier(SVC(**params),
                         algorithm="SAMME",
                         n_estimators=15)

#rankboost
params = {'C': 10, 'kernel': 'linear', 'max_iter_boosting':10}
bdt = RankBoost(**params)

#linear svm
params = {'C': 10, 'kernel': 'linear'}
bdt = SVC(**params)

#rbf svm
params = {'C': 10, 'kernel': 'rbf', 'gamma': 1}
bdt = SVC(**params)

#martiboost + linear svm
params = {'C': 10, 'kernel': 'linear'}
bdt = MartiBoost(**params)
'''
#martiboost + balanced_decision_stump
params = {'weak_classifier': 'dtree_stump_balanced', 'max_iter_boosting': 200}
bdt = MartiBoost(**params)

#MIBoosting + decision_stump
params = {'weak_classifier': 'dtree_stump','max_depth': 1,'max_iter_boosting': 200}
bdt1 = MIBoosting_Xu(**params)

print "fitting the training set"
bdt.fit(X, y)
bdt1.fit(train_class.bags, train_class.bag_labels)
print "fitting completed"

#import pdb;pdb.set_trace()
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

Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
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

# Plot the two-class decision scores
'''
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
plt.legend(loc='upper right')
plt.ylabel('Samples')
plt.xlabel('Score')
plt.title('Decision Scores')

plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
'''
plt.savefig('Adaboost_twoclass.pdf')
import pdb;pdb.set_trace()
