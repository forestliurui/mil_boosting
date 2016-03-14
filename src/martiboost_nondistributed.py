"""
This is the implementation of Martingale Boosting in the paper "Philip Long, Rocco Servedio, Martingale Boosting"
I didn't balance the output of each weak classifier as described in Section 5 of above paper, which is required for Theorem 2, i,e. any weak learner can be boosted to arbitrarily stong learner
For the current implementation, Theorem 1 gurantees that any weak two-sided learner can be boosted to arbitrarily strong two-sided learner
"""

import copy
from math import sqrt, exp
from mi_svm import SVM
import string
import data
import numpy as np
import sets
from sklearn.tree import DecisionTreeClassifier
from Dtree_Stump_Balanced import Dtree_Stump_Balanced

WEAK_CLASSIFIERS = {
	'svm': SVM,
	'dtree_stump': DecisionTreeClassifier,
	'dtree_stump_balanced': Dtree_Stump_Balanced
}

class TreeNode(object):
	def __init__(self):
		self.classifier = None
		#self.weights = np.array(weights)
		self.instances = None  #two-dimensional np-array (each row corresponds to each instance, each column corresponds to each feature)
		self.labels = None   #one-dimensional np-array
		self.indices = None #one-dimensional np-array which contains the indices of self.instances in the whole training instances array

		self.instances_test = None
		self.indices_test = None

	def init_classifier(self, classifier):
		self.classifier = copy.deepcopy( classifier )

	def compute_balanced_weights(self):
		if self.instances is None:
			self.weights = None
		else:
			num_instances = self.instances.shape[0]

			num_instances_positive = np.sum(self.labels == 1)	
			num_instances_negative = np.sum(self.labels != 1)	
			self.weights= np.ones((num_instances))
			for inst_index in range(num_instances):
				if self.labels[inst_index]==1:
					self.weights[inst_index] = float(1)/num_instances_positive
				else:
					self.weights[inst_index] = float(1)/num_instances_negative

	def update_instances_labels(self, instances, indices, labels): #update the training instances, indices and labels for each node
		if instances.shape[0] != 0:
			if self.instances is None:
				self.instances = np.array(instances)
				self.labels = np.array(labels)
				self.indices = np.array(indices)
			else:
				self.instances = np.vstack((self.instances, instances))
				self.labels = np.hstack((self.labels, labels))
				self.indices = np.hstack((self.indices, indices))

	def update_instances_test(self, instances, indices):
		if instances.shape[0] != 0:
			if self.instances_test is None:
				self.instances_test = np.array(instances)
				
				self.indices_test = np.array(indices)
			else:
				self.instances_test = np.vstack((self.instances_test, instances))
				
				self.indices_test = np.hstack((self.indices_test, indices))		

class SingleSideClassifier(object):  #classifier which give positive or negative prediction for every instance
	def __init__(self, **parameters):
		pass
	def fit(self, X, y, weights =  None):
		if y[0] >0:
			self.label = 1
		else:
			self.label = -1
	def predict(self, X):
		num_instances = X.shape[0]
		return self.label*np.ones((num_instances))

class RandomClassifier(object): #random classifier 
	def __init__(self):
		pass
	def fit(self, X, y, weights =  None):
		pass
	def predict(self, X):
		num_instances = X.shape[0]
		return np.random.rand(num_instances)-0.5  

class MartiBoost(object):
	def __init__(self, **parameters):
		self.weak_classifiers = {}
		self.weak_classifier_name = parameters.pop('weak_classifier')
		
		if 'max_iter_boosting' not in parameters:
			self.max_iter_boosting = 100  #the max num of layers of weak learners. Note that the predictions are according to the instances positions at (max+1)-th layer
		else:
			self.max_iter_boosting =  parameters.pop('max_iter_boosting')
		self.actul_boosting_iter = self.max_iter_boosting
		self.actual_rounds_of_boosting = self.actul_boosting_iter

		if self.weak_classifier_name == 'dtree_stump':
			parameters['max_depth'] = 1
		parameters.pop('normalization', 0)

		self.parameters = parameters

		self.X_bags_test = None
		self.X_bags = None
		self.y_labels = None
		self.X_instances = None

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
			self.X_bags = X_bags
			self.y_labels = y_labels 
			self.X_instances = instances	
		else:
			instances = X_bags
			instance_labels_generated_from_bag_labels = y_labels
			self.X_instances = instances

		instance_labels_generated_from_bag_labels = np.array( instance_labels_generated_from_bag_labels )
		num_instances = instances.shape[0]
		num_instances_positive = np.sum(instance_labels_generated_from_bag_labels == 1)	
		num_instances_negative = np.sum(instance_labels_generated_from_bag_labels != 1)		

		self.c=[] #the list of weights for weak classifiers

		#import pdb;pdb.set_trace()
		'''
		#initial inst weights
		weights_inst= np.ones((num_instances))
		for inst_index in range(num_instances):
			if instance_labels_generated_from_bag_labels[inst_index]==1:
				weights_inst[inst_index] = float(1)/num_instances_positive
			else:
				weights_inst[inst_index] = float(1)/num_instances_negative
		'''

		#instance_classifier = WEAK_CLASSIFIERS[self.weak_classifier_name](**self.parameters)

		indices = np.array(range(num_instances))		

		self.weak_classifiers[0] = {}
		self.weak_classifiers[0][0] = TreeNode() 
		self.weak_classifiers[0][0].update_instances_labels(instances, indices, instance_labels_generated_from_bag_labels )
		
		#current_level_list = []
		#current_level_list.append(self.head)

		for index_Boosting in range(self.max_iter_boosting):

			current_level_dict = self.weak_classifiers[index_Boosting]
			
			for current_key in current_level_dict.keys():
		
				#import pdb;pdb.set_trace()
				current = current_level_dict[current_key]
				if current.instances is None: #No training instances in the current node, so skip it
					continue

				if len( sets.Set(current.labels) ) == 2:
					current.init_classifier( WEAK_CLASSIFIERS[self.weak_classifier_name](**self.parameters) ) #SVM only works for 2 classes
				else: 
					current.init_classifier( SingleSideClassifier(**self.parameters) )  #use self-defined classifier for single class case

				current.compute_balanced_weights()
				current.classifier.fit(current.instances, current.labels.tolist(), current.weights)
						
				
			
				current.training_predictions = current.classifier.predict(current.instances)
				current.errors_instance = {}
				#import pdb;pdb.set_trace()
				current.errors_instance["positive"] = np.average(current.training_predictions[current.labels == 1]>=0)
				current.errors_instance["negative"] = np.average(current.training_predictions[current.labels != 1]<0)
			

				if index_Boosting+1 not in self.weak_classifiers.keys():
					self.weak_classifiers[ index_Boosting+1 ] = {}					

				#instance_classifier = WEAK_CLASSIFIERS[self.weak_classifier_name](**self.parameters) #left child--baised to negative predictions

				if current_key not in self.weak_classifiers[ index_Boosting+1 ].keys():
					self.weak_classifiers[ index_Boosting+1 ][current_key] = TreeNode()	
				self.weak_classifiers[ index_Boosting+1 ][current_key].update_instances_labels(current.instances[current.training_predictions <0], current.indices[current.training_predictions <0] , current.labels[ current.training_predictions <0 ])
				
		
				instance_classifier = WEAK_CLASSIFIERS[self.weak_classifier_name](**self.parameters)

				if current_key+1 not in self.weak_classifiers[ index_Boosting+1 ].keys():
					self.weak_classifiers[ index_Boosting+1 ][current_key+1] = TreeNode()
				self.weak_classifiers[ index_Boosting+1 ][current_key+1].update_instances_labels(current.instances[current.training_predictions >= 0], current.indices[current.training_predictions >=0] ,current.labels[ current.training_predictions >=0 ])

				#next_level_list.append(current.right)
			
			#current_level_list = next_level_list
		#import pdb;pdb.set_trace()

	def predict_train(self, iter = None, getInstPrediction = False):
		if iter == None or iter > self.actul_boosting_iter:
			iter = self.actul_boosting_iter

		num_instances = self.X_instances.shape[0]
		results = 0*np.ones((num_instances))
		current_level_dict = self.weak_classifiers[iter]
		for current_key in current_level_dict.keys():
			current = current_level_dict[current_key]
			if current.indices is None:
				continue
			
			if current_key < iter/float(2):
				results[current.indices] = -1
			else:
				results[current.indices] = 1

		#import pdb;pdb.set_trace()
		if getInstPrediction: #get instance level predictions
			return results   #the output is either 1 or -1 for each prediction	
		else: #get bag level predictions
			if self.X_bags is None:
				raise Exception("Can't get bag level prediction for training data due to the lack of training bag data")
			else:
				predictions_bag = get_bag_label(results, self.X_bags)
				return predictions_bag


		

	def _predict(self, X = None, iter = None):

		if iter == None or iter > self.actul_boosting_iter:
			iter = self.actul_boosting_iter
		if X is not None:
			num_instances_test = X.shape[0]	

			self.weakers = copy.deepcopy( self.weak_classifiers )		
		
			#import pdb;pdb.set_trace()
			#clear the data (instances_test and indices_test) for last test run
			'''
			for index_Boosting in range(iter):
				current_level_dict = self.weakers[index_Boosting]
				for current_key in current_level_dict.keys():
					current = current_level_dict[current_key]
					current.instances_test = None
					current.indices_test = None
			'''
			#clear the data for last test run

			self.weakers[0][0].instances_test = X
			self.weakers[0][0].indices_test = np.array(range(num_instances_test))

			for index_Boosting in range(self.actul_boosting_iter):
	
				current_level_dict = self.weakers[index_Boosting]
			
				for current_key in current_level_dict.keys():
					current = current_level_dict[current_key]
					if current.instances_test is None:
						continue
					if current.classifier is None:
						current.init_classifier(RandomClassifier())
				
					current.predictions_test = current.classifier.predict(current.instances_test)
				
				
					if index_Boosting+1 not in self.weakers.keys():
						self.weakers[ index_Boosting+1 ] = {}					

					#instance_classifier = WEAK_CLASSIFIERS[self.weak_classifier_name](**self.parameters) #left child--baised to negative predictions

					if current_key not in self.weakers[ index_Boosting+1 ].keys():
						self.weakers[ index_Boosting+1 ][current_key] = TreeNode()	
					self.weakers[ index_Boosting+1 ][current_key].update_instances_test(current.instances_test[current.predictions_test <0], current.indices_test[current.predictions_test <0])
				
		
					instance_classifier = WEAK_CLASSIFIERS[self.weak_classifier_name](**self.parameters)

					if current_key+1 not in self.weakers[ index_Boosting+1 ].keys():
						self.weakers[ index_Boosting+1 ][current_key+1] = TreeNode()
					self.weakers[ index_Boosting+1 ][current_key+1].update_instances_test(current.instances_test[current.predictions_test >= 0], current.indices_test[current.predictions_test >=0])
			#import pdb;pdb.set_trace()
		else:
			if type(self.X_bags_test) != list:  # treat it as normal supervised learning setting

				num_instances_test = self.X_bags_test.shape[0]
			else:
				X_instances_test = np.vstack(self.X_bags_test)
				num_instances_test = X_instances_test.shape[0]

		results = 0*np.ones((num_instances_test))
		current_level_dict = self.weakers[iter] #To see the results of one layer, we need to look at layer #1
		for current_key in current_level_dict.keys():
			current = current_level_dict[current_key]
			if current.indices_test is None:
				continue
			
			if current_key < iter/float(2):
				results[current.indices_test] = -1
			else:
				results[current.indices_test] = 1

		#import pdb;pdb.set_trace()
		return results   #the output is either 1 or -1 for each prediction

	def predict(self, X_bags = None, iter = None, getInstPrediction = False):		

		#X_bags is a list of arrays, each bag is an array in the list
		#The row of array corresponds to instances in the bag, column corresponds to feature

		#predictions_bag is the returned array of predictions which are real values
		#output is either 1 or -1 
		
		if iter == None or iter > self.actul_boosting_iter:
			iter = self.actul_boosting_iter
	
		#import pdb;pdb.set_trace()

		#print self.c
		if X_bags is not None:
			self.X_bags_test = X_bags
			if type(X_bags) != list:  # treat it as normal supervised learning setting
				#X_bags = [X_bags[inst_index,:] for inst_index in range(X_bags.shape[0])]
				#import pdb;pdb.set_trace()
				predictions_accum = self._predict(X = X_bags, iter = iter)

				return np.array(predictions_accum)
			else:
			
				X_instances = np.vstack(X_bags)
				predictions_accum = self._predict(X = X_instances, iter = iter)

				if getInstPrediction:  #return the instance level predictions for the input bags
					return np.array(predictions_accum)
				else:
					predictions_bag = get_bag_label(predictions_accum, X_bags)
					return predictions_bag

				''' #slow way 
				num_bags=len(X_bags)
				predictions_bag=[]
				for index_bag in range(num_bags):
					#import pdb;pdb.set_trace()
					predictions_bag_temp= np.max( self._predict(X_bags[index_bag], iter = iter) ) 
					predictions_bag.append(predictions_bag_temp)
				#import pdb; pdb.set_trace()

				predictions_bag=np.array( predictions_bag )
				return predictions_bag
				'''
		elif X_bags is None and self.X_bags_test is not None:
			if type(self.X_bags_test) != list:  # treat it as normal supervised learning setting
				#X_bags = [X_bags[inst_index,:] for inst_index in range(X_bags.shape[0])]
				#import pdb;pdb.set_trace()
				predictions_accum = self._predict(iter = iter)

				return np.array(predictions_accum)
			else:
			
				#X_instances = np.vstack(X_bags)
				predictions_accum = self._predict(iter = iter)
				if getInstPrediction:  #return the instance level predictions for the input bags
					return np.array(predictions_accum)
				else:
					predictions_bag = get_bag_label(predictions_accum, self.X_bags_test)
					return predictions_bag
		else:
			raise Exception('As the first time to call predict(), please specify the test dataset')

def get_bag_label(instance_predictions, bags):
	num_bag = len(bags)
	p_index= 0
	bag_predictions = []
	for bag_index in range(num_bag):
		n_index =p_index+ bags[bag_index].shape[0]
		
		bag_predictions.append( np.average(instance_predictions[p_index: n_index]) )
		p_index = n_index
	return np.array(bag_predictions)

