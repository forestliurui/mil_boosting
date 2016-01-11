#This is the nondistributed version of RankBoost for bipartite setting, described in Figure 9.2 at the book "Foundation of Machine Learning"

from math import sqrt, exp
from mi_svm import SVM
import string
import data
import numpy as np

class RankBoost(object):
	def __init__(self, **parameters):
		self.parameters = parameters
		self.weak_classifiers = []
		self.epsilon = {}
		self.epsilon["positive"] = []
		self.epsilon["negative"] = []
		self.alphas = []
		self.weights_instance=[]

	def fit(self, X_bags, y_labels):
		'''X_bags is a list of arrays, each bag is an array in the list
		The row of array corresponds to instances in the bag, column corresponds to feature
		y_labels is the list which contains the labels of bags. Here, binary labels are assumed, i.e. +1/-1
		'''

		if type(y_labels)!=list:
			y_labels=y_labels.tolist()
		
		if(type( y_labels[0] )==bool):  #convert the boolean labels into +1/-1 labels
			y_labels = 2*y_labels-1 
	
		max_iter_boosting=10
		num_bags=len(X_bags)
				
		num_instance_each_bag=[x.shape[0] for x in X_bags]
		


		instances=np.vstack((X_bags))
		num_instances = instances.shape[0]

		instance_labels_generated_from_bag_labels=[y_labels[bag_index]*np.ones((1, num_instance_each_bag[bag_index]))[0] for bag_index in range(num_bags)  ]
		instance_labels_generated_from_bag_labels=np.hstack((instance_labels_generated_from_bag_labels))		
		
		num_instances_positive = np.sum(instance_labels_generated_from_bag_labels == 1)	
		num_instances_negative = np.sum(instance_labels_generated_from_bag_labels == -1)		

		self.c=[] #the list of weights for weak classifiers

		#initial inst weights
		weights_inst= np.ones((num_instances))
		for inst_index in range(num_instances):
			if instance_labels_generated_from_bag_labels[inst_index]==1:
				weights_inst[inst_index] = float(1)/num_instances_positive
			else:
				weights_inst[inst_index] = float(1)/num_instances_negative

		
		for index_Boosting in range(max_iter_boosting):

			self.weights_instance.append(weights_inst)
			instance_classifier=SVM(**self.parameters)
			
			instance_classifier.fit(instances, instance_labels_generated_from_bag_labels, weights_instance)
			predictions = (instance_classifier.predict(instances) >0 )+0
			self.epsilon["positive"].append( np.average( predictions[instance_labels_generated_from_bag_labels == 1] ) )
			self.epsilon["negative"].append( np.average( predictions[instance_labels_generated_from_bag_labels == -1] ) )
			self.alphas.append(0.5*np.log(self.epsilon["positive"][-1]/self.epsilon["negative"][-1]))
			Z={}
			Z["positive"]=1-self.epsilon["positive"][-1]+np.sqrt( self.epsilon["positive"][-1]*self.epsilon["negative"][-1] )
			Z["negative"]=1-self.epsilon["negative"][-1]+np.sqrt( self.epsilon["positive"][-1]*self.epsilon["negative"][-1] )
			for inst_index in range(num_instances):
				if instance_labels_generated_from_bag_labels[inst_index]==1:
					weights_inst[inst_index] = weights_inst[inst_index]*np.exp(-self.alphas[-1]*predictions[inst_index])/Z["positive"]
				else:
					weights_inst[inst_index] = weights_inst[inst_index]*np.exp(+self.alphas[-1]*predictions[inst_index])/Z["negative"]

			self.weak_classifiers.append(instance_classifier)

	def predict(self, X_bags):
		#X_bags is a list of arrays, each bag is an array in the list
		#The row of array corresponds to instances in the bag, column corresponds to feature

		#predictions_bag is the returned list of predictions which are real values 
		
		self.c=self.alphas
		num_bags=len(X_bags)

		predictions_bag=[]
		print len(self.c)
		#print self.c
		#print len(self.weak_classifiers)
		for index_bag in range(num_bags):
			#import pdb;pdb.set_trace()
			predictions_bag_temp=np.average( [ np.max( instance_classifier.predict(X_bags[index_bag]) )  for instance_classifier in self.weak_classifiers  ]  ,  weights=self.c )
			predictions_bag.append(predictions_bag_temp)
		#import pdb; pdb.set_trace()

		predictions_bag=np.array( predictions_bag )
		return predictions_bag

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
