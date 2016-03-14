#This is the nondistributed version of RankBoost for bipartite setting, described in Figure 9.2 at the book "Foundation of Machine Learning"

from math import sqrt, exp
from mi_svm import SVM
import string
import data
import numpy as np
import copy

from sklearn.tree import DecisionTreeClassifier
from Dtree_Stump_Balanced import Dtree_Stump_Balanced

WEAK_CLASSIFIERS = {
	'svm': SVM,
	'dtree_stump': DecisionTreeClassifier,
	'dtree_stump_balanced': Dtree_Stump_Balanced
}

class RankBoost_pos(object):
	def __init__(self, **parameters):

		self.max_iter_boosting = parameters.pop("max_iter_boosting", 10)
		self.weak_classifier_name = parameters.pop('weak_classifier', 'dtree_stump') 
		if self.weak_classifier_name == 'dtree_stump':
			parameters['max_depth'] = 1
		parameters.pop('normalization', 0)
		self.parameters = parameters
		self.weak_classifiers = []
		self.weak_classifiers_pos = []
		self.epsilon = {}
		self.epsilon["positive"] = []
		self.epsilon["negative"] = []
		self.alphas = []
		self.alphas_pos = []
		self.weights_instance=[]

		


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

		#initial inst weights
		weights_inst= np.ones((num_instances))
		for inst_index in range(num_instances):
			if instance_labels_generated_from_bag_labels[inst_index]==1:
				weights_inst[inst_index] = float(1)/num_instances_positive
			else:
				weights_inst[inst_index] = float(1)/num_instances_negative

		
		for index_Boosting in range(max_iter_boosting):

			self.weights_instance.append(np.array(weights_inst))
			instance_classifier=WEAK_CLASSIFIERS[self.weak_classifier_name](**self.parameters)
		
			#import pdb;pdb.set_trace()

			instance_classifier.fit(instances, instance_labels_generated_from_bag_labels.tolist(), weights_inst)
			self.weak_classifiers.append(copy.deepcopy(instance_classifier))
			predictions = (instance_classifier.predict(instances) >0 )+0
			self.epsilon["positive"].append( np.average( predictions[instance_labels_generated_from_bag_labels == 1], weights = weights_inst[instance_labels_generated_from_bag_labels == 1] ) )
			self.epsilon["negative"].append( np.average( predictions[instance_labels_generated_from_bag_labels != 1], weights = weights_inst[instance_labels_generated_from_bag_labels != 1] ) )

			if self.epsilon["negative"][-1] == 0:
				self.alphas.append(20)
				break
			
			self.alphas.append(0.5*np.log(self.epsilon["positive"][-1]/self.epsilon["negative"][-1]))
			if self.alphas[-1] > 0:  #add only weak classifiers with positive weight
				self.alphas_pos.append(self.alphas[-1])
				self.weak_classifiers_pos.append(copy.deepcopy(instance_classifier))

			Z={}
			Z["positive"]=1-self.epsilon["positive"][-1]+np.sqrt( self.epsilon["positive"][-1]*self.epsilon["negative"][-1] )
			Z["negative"]=1-self.epsilon["negative"][-1]+np.sqrt( self.epsilon["positive"][-1]*self.epsilon["negative"][-1] )
			for inst_index in range(num_instances):
				if instance_labels_generated_from_bag_labels[inst_index]==1:
					weights_inst[inst_index] = weights_inst[inst_index]*np.exp(-self.alphas[-1]*predictions[inst_index])/Z["positive"]
				else:
					weights_inst[inst_index] = weights_inst[inst_index]*np.exp(+self.alphas[-1]*predictions[inst_index])/Z["negative"]
		self.actual_rounds_of_boosting = len(self.alphas)
			

	def predict(self, X_bags, iter = None):
		#X_bags is a list of arrays, each bag is an array in the list
		#The row of array corresponds to instances in the bag, column corresponds to feature

		#predictions_bag is the returned array of predictions which are real values 
		threshold = 0.5
		self.c=self.alphas
		if iter == None or iter > len(self.c):
			iter = len(self.c)
	
		print "self.c: ",
		print len(self.c)
		if type(X_bags) != list:  # treat it as normal supervised learning setting
			#X_bags = [X_bags[inst_index,:] for inst_index in range(X_bags.shape[0])]
			
			self.c_pos = self.alphas_pos
			#self.c_pos[self.c_pos<0] = 0

			predictions_list = [( instance_classifier.predict(X_bags).reshape((1, -1))>0 )+ 0 for instance_classifier in self.weak_classifiers_pos[0:iter] ]
			#import pdb;pdb.set_trace()
			predictions_accum = np.matrix(self.c_pos[0:iter])*np.matrix( np.vstack((predictions_list)) )/np.sum(self.c_pos[0:iter])

			#import pdb;pdb.set_trace()
			return np.array(predictions_accum)[0] - threshold   #we need to deduct a threshold because (instance_classifier.predict > 0) + 0 is either 0 or 1
		else:

			X_instances = np.vstack(X_bags)
			predictions_accum = self.predict(X_instances, iter)
			predictions_bag = get_bag_label(predictions_accum, X_bags)
			return predictions_bag

			'''slow way
			num_bags=len(X_bags)
			predictions_bag=[]
			#print len(self.c)
			#print self.c
			#print len(self.weak_classifiers)
			for index_bag in range(num_bags):
				#import pdb;pdb.set_trace()
				predictions_bag_temp=np.average( [ np.max( instance_classifier.predict(X_bags[index_bag]) )  for instance_classifier in self.weak_classifiers  ][0:iter]  ,  weights=self.c[0:iter] )/np.sum(self.c[0:iter]) - threshold 

				predictions_bag.append(predictions_bag_temp)
			#import pdb; pdb.set_trace()

			predictions_bag=np.array( predictions_bag )
			return predictions_bag
			'''

	def predict_inst(self, X_bags):
		#X_bags is a list of arrays, each bag is an array in the list
		#The row of array corresponds to instances in the bag, column corresponds to feature

		#predictions_inst is the returned list of arrays which are instance-level real-valued predictions for all test bags. Note the array in the list here is row array (1-D)
		
		num_bags=len(X_bags)

		predictions_inst=[]
		for index_bag in range(num_bags):
			predictions_inst_temp=np.average(  np.vstack(( [ instance_classifier.predict(X_bags[index_bag])  for instance_classifier in self.weak_classifiers  ] )) , axis=0, weights=self.c )
			predictions_inst.append(predictions_inst_temp)
		import pdb; pdb.set_trace()

		return predictions_inst

def get_bag_label(instance_predictions, bags):
	num_bag = len(bags)
	p_index= 0
	bag_predictions = []
	for bag_index in range(num_bag):
		n_index =p_index+ bags[bag_index].shape[0]
		
		bag_predictions.append( np.average(instance_predictions[p_index: n_index]) )
		p_index = n_index
	return np.array(bag_predictions)