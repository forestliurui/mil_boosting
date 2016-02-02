"""
This is the implementation of decision stump whose predictions are balanced according to Lemma 4 in Philip Long's paper entitled 'Boosting the Area Under the ROC Curve'
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import random

class Dtree_Stump_Balanced(DecisionTreeClassifier):
	"""
	The balanced decision tree we want to implement in this script
	"""
	def __init__(self, **kwargs):

		if 'max_depth' in kwargs:
			kwargs.pop('max_depth')

		kwargs.update({'max_depth': 1}) #make sure this is a decision stump

		random.seed(123456) #initialize the random number generator

		super(Dtree_Stump_Balanced, self).__init__(**kwargs)

	def fit(self, X_bags, y_labels, weights = None ):
		'''
		X_bags is a list of arrays, each bag is an array in the list
		The row of array corresponds to instances in the bag, column corresponds to feature
		y_labels is the list which contains the labels of bags. Here, binary labels are assumed, i.e. +1/-1
		'''
		
		

		if type(y_labels)!=list:
			y_labels=y_labels.tolist()
		
		if type( y_labels[0] )==bool or 0 in y_labels:  #convert the boolean labels into +1/-1 labels
			y_labels = map(lambda x:2*x-1, y_labels )
		
		
		if type(X_bags) == list:  # treat it as the SIL in MIL setting
			num_bags=len(X_bags)
				
			num_instance_each_bag=[x.shape[0] for x in X_bags]
		
			instances=np.vstack((X_bags))
			

			instance_labels_generated_from_bag_labels=[y_labels[bag_index]*np.ones((1, num_instance_each_bag[bag_index]))[0] for bag_index in range(num_bags)  ]
			instance_labels_generated_from_bag_labels=np.hstack((instance_labels_generated_from_bag_labels))
			#instance_weights = 		
		else:
			instances = X_bags
			instance_labels_generated_from_bag_labels = y_labels
			instance_weights = weights

		self.train_instances = instances
		self.train_instance_labels = np.array(instance_labels_generated_from_bag_labels)
		
		if instance_weights is None:
			num_instance = instances.shape[0]
			self.train_instance_weights = float(1)/num_instance*np.ones(num_instance )
		else:
			self.train_instance_weights = instance_weights

		instance_labels_generated_from_bag_labels = np.array( instance_labels_generated_from_bag_labels )
		num_instances = instances.shape[0]
		num_instances_positive = np.sum(instance_labels_generated_from_bag_labels == 1)	
		num_instances_negative = np.sum(instance_labels_generated_from_bag_labels != 1)	

		super(Dtree_Stump_Balanced, self).fit(instances, instance_labels_generated_from_bag_labels, instance_weights)

	def _predict(self, X):

		theta = 0

		self.train_predictions = super(Dtree_Stump_Balanced, self).predict(self.train_instances)
		self.D = {}
		self.D['positive'] = self.train_instance_weights[self.train_instance_labels == 1]
		self.train_predictions_positive = self.train_predictions[self.train_instance_labels == 1]

		self.D['negative'] = self.train_instance_weights[self.train_instance_labels == -1]
		self.train_predictions_negative = self.train_predictions[self.train_instance_labels == -1]

		self.p={}
		#import pdb;pdb.set_trace()
		self.p['positive'] = np.sum(self.D['positive'][self.train_predictions_positive<theta] )/float(np.sum(self.D['positive']))
		#import pdb;pdb.set_trace()
		self.p['negative'] = np.sum(self.D['negative'][self.train_predictions_negative>=theta] )/float(np.sum(self.D['negative']))
		
		test_predictions = super(Dtree_Stump_Balanced, self).predict(X)

		if self.p['positive'] >= self.p['negative']:
			self.zeta = ( self.p['positive'] - self.p['negative'] )/(float(1) + self.p['positive'] - self.p['negative'])
			
			for inst_index in range(X.shape[0]):
				if test_predictions[inst_index] < theta:
					if random.random()< self.zeta:
						test_predictions[inst_index] = 1
					else: 
						test_predictions[inst_index] = -1 
		else:
			self.zeta = ( self.p['negative'] - self.p['positive'] )/(float(1) + self.p['negative'] - self.p['positive'])
			for inst_index in range(X.shape[0]):
				if test_predictions[inst_index] > theta:
					if random.random()< self.zeta:
						test_predictions[inst_index] = -1
					else: 
						test_predictions[inst_index] = 1 

		return test_predictions

	def predict(self, X_bags):		

		#X_bags is a list of arrays, each bag is an array in the list
		#The row of array corresponds to instances in the bag, column corresponds to feature

		#predictions_bag is the returned array of predictions which are real values 
		
			
		
		#print self.c
		if type(X_bags) != list:  # treat it as normal supervised learning setting
			#X_bags = [X_bags[inst_index,:] for inst_index in range(X_bags.shape[0])]
			#import pdb;pdb.set_trace()
			predictions_accum = self._predict(X_bags)

			return np.array(predictions_accum)
		else:
			num_bags=len(X_bags)

			predictions_bag=[]
			print len(self.c)
			#print self.c
			#print len(self.weak_classifiers)
			for index_bag in range(num_bags):
				#import pdb;pdb.set_trace()
				predictions_bag_temp= np.max( self._predict(X_bags[index_bag]) ) 
				predictions_bag.append(predictions_bag_temp)
			#import pdb; pdb.set_trace()

			predictions_bag=np.array( predictions_bag )
			return predictions_bag

		
