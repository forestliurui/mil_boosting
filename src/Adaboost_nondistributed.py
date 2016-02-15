"""
This is the nondistributed version of adaboost. 
This script is basically a wrapper which implement the interface of adaboost from sklearn so that it is consistent with my other boosters
"""

from math import sqrt, exp
from mi_svm import SVM
import string
import data
import numpy as np
import copy


from sklearn.tree import DecisionTreeClassifier
from Dtree_Stump_Balanced import Dtree_Stump_Balanced
from sklearn.ensemble import AdaBoostClassifier

WEAK_CLASSIFIERS = {
	'svm': SVM,
	'dtree_stump': DecisionTreeClassifier,
	'dtree_stump_balanced': Dtree_Stump_Balanced
}

class AdaBoost(object):
	def __init__(self, **parameters):

		self.max_iter_boosting = parameters.pop("max_iter_boosting", 10)
		self.weak_classifier_name = parameters.pop('weak_classifier', 'dtree_stump') 
		if self.weak_classifier_name == 'dtree_stump':
			parameters['max_depth'] = 1
		parameters.pop('normalization', 0)
		self.parameters = parameters
		self.ensemble_classifier = AdaBoostClassifier(WEAK_CLASSIFIERS[self.weak_classifier_name](**self.parameters), algorithm="SAMME", n_estimators = self.max_iter_boosting)


	def fit(self, X_bags, y_labels):
		'''
		X_bags is a list of arrays, each bag is an array in the list
		The row of array corresponds to instances in the bag, column corresponds to feature
		y_labels is the list which contains the labels of bags. Here, binary labels are assumed, i.e. +1/-1
		'''
		
		

		if type(y_labels)!=list:
			y_labels=y_labels.tolist()
		
		if type( y_labels[0] )==bool or 0 in y_labels:  #convert the boolean labels into +1/-1 labels
			y_labels = map(lambda x:2*x-1, y_labels )
		
		max_iter_boosting=self.max_iter_boosting

		if type(X_bags) == list:  # treat it as the SIL in MIL setting
			num_bags=len(X_bags)
				
			num_instance_each_bag=[x.shape[0] for x in X_bags]
		
			instances=np.vstack((X_bags))
			

			instance_labels_generated_from_bag_labels=[y_labels[bag_index]*np.ones((1, num_instance_each_bag[bag_index]))[0] for bag_index in range(num_bags)  ]
			instance_labels_generated_from_bag_labels=np.hstack((instance_labels_generated_from_bag_labels))		
		else:
			instances = X_bags
			instance_labels_generated_from_bag_labels = y_labels

		instance_labels_generated_from_bag_labels = np.array( instance_labels_generated_from_bag_labels )
		num_instances = instances.shape[0]
		num_instances_positive = np.sum(instance_labels_generated_from_bag_labels == 1)	
		num_instances_negative = np.sum(instance_labels_generated_from_bag_labels != 1)		

		self.c=[] #the list of weights for weak classifiers

		#import pdb;pdb.set_trace()

		self.ensemble_classifier.fit(instances, instance_labels_generated_from_bag_labels)
		self.weak_classifiers = self.ensemble_classifier.estimators_
		self.actual_rounds_of_boosting = len(self.weak_classifiers)
	
	def _predict(self, instances, iter = None):
		
		if iter == None or iter > self.actual_rounds_of_boosting:
			iter = self.actual_rounds_of_boosting

		staged_generator = self.ensemble_classifier.staged_decision_function(instances)

		staged_decision_function_output = [x for x in staged_generator]  #a list of one dimensional array
		return staged_decision_function_output[iter - 1] 

	def predict(self, X_bags, iter = None):
		#X_bags is a list of arrays, each bag is an array in the list
		#The row of array corresponds to instances in the bag, column corresponds to feature

		#predictions_bag is the returned array of predictions which are real values 
		
		if iter == None or iter > self.actual_rounds_of_boosting:
			iter = self.actual_rounds_of_boosting
	
		print "number of boosting: ",
		print self.actual_rounds_of_boosting
		if type(X_bags) != list:  # treat it as normal supervised learning setting

			return self._predict(X_instances, iter)
		else:

			X_instances = np.vstack(X_bags)
			predictions_accum = self._predict(X_instances, iter)
			predictions_bag = get_bag_label(predictions_accum, X_bags)
			return predictions_bag

def get_bag_label(instance_predictions, bags):
	num_bag = len(bags)
	p_index= 0
	bag_predictions = []
	for bag_index in range(num_bag):
		n_index =p_index+ bags[bag_index].shape[0]
		
		bag_predictions.append( np.average(instance_predictions[p_index: n_index]) )
		p_index = n_index
	return np.array(bag_predictions)
		

