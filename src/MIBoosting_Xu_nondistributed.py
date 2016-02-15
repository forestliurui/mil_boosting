#this is the implementation of MIBoosting algorithm in Xin Xu's paper "Logistic regresssion and Boosting for Labeled Bags of Intances"
#which is the base of Zhou's nips paper MIMLBoosting.

from mi_svm import SVM
import numpy as np
from scipy.optimize import minimize
from sklearn.tree import DecisionTreeClassifier
from Dtree_Stump_Balanced import Dtree_Stump_Balanced
from mi_svm import SVM
import copy


WEAK_CLASSIFIERS = {
	'svm': SVM,
	'dtree_stump': DecisionTreeClassifier,
	'dtree_stump_balanced': Dtree_Stump_Balanced
}

def subproblem_MIBoosting(weights, errors): 
	#This is the subproblem that needs to be solved in order to get weight c for each weak classifier
	num_bags=len(errors)
	return lambda x: np.sum([weights[index_bag]*np.exp( (2*errors[index_bag]-1)*x )     for index_bag in range(num_bags)   ])	


class MIBoosting_Xu(object):
	def __init__(self, **parameters):
		self.weak_classifier_name = parameters.pop('weak_classifier', 'dtree_stump') 

		self.max_iter_boosting =  parameters.pop('max_iter_boosting', 10)
		if self.weak_classifier_name == 'dtree_stump':
			parameters['max_depth'] = 1
		parameters.pop('normalization', 0)
		self.parameters=parameters
		self.weak_classifiers=[]
	def fit(self, X_bags, y_labels):
		#X_bags is a list of arrays, each bag is an array in the list
		#The row of array corresponds to instances in the bag, column corresponds to feature
		#y_labels is the list which contains the labels of bags. Here, binary labels are assumed, i.e. +1/-1
		#import pdb;pdb.set_trace()
		if type(y_labels)!=list:
			y_labels=y_labels.tolist()
		
		if type( y_labels[0] )==bool or 0 in y_labels:  #convert the boolean labels into +1/-1 labels
			y_labels = map(lambda x:2*x-1, y_labels )
	
		max_iter_boosting=self.max_iter_boosting
		num_bags=len(X_bags)
				
		num_instance_each_bag=[x.shape[0] for x in X_bags]
		


		instances=np.vstack((X_bags))

		instance_labels_generated_from_bag_labels=[y_labels[bag_index]*np.ones((1, num_instance_each_bag[bag_index]))[0] for bag_index in range(num_bags)  ]
		instance_labels_generated_from_bag_labels=np.hstack((instance_labels_generated_from_bag_labels))		
		
		
		self.c=[] #the list of weights for weak classifiers

		#initial bag weights
		weights_bag=np.ones((num_bags))

		for index_Boosting in range(max_iter_boosting):
			
			weights_instance= [( weights_bag[bag_index]/float(num_instance_each_bag[bag_index]) )*np.ones((1, num_instance_each_bag[bag_index]))[0] for bag_index in range(num_bags)]
			weights_instance=np.hstack((weights_instance))
			
			instance_classifier=WEAK_CLASSIFIERS[self.weak_classifier_name](**self.parameters)

			instance_classifier.fit(instances, instance_labels_generated_from_bag_labels, weights_instance)

			error_bag=[ np.average( ( instance_classifier.predict(X_bags[index_bag]) >0  ) != (y_labels[index_bag]==1) )  for index_bag in range(num_bags)  ]
		
			if (np.average(error_bag<0.5)==1):
				break	
			#import pdb;pdb.set_trace()
			res=minimize(  subproblem_MIBoosting(weights_bag, error_bag), 0 )
			self.c.append(res['x'][0])
			
			#print error_bag
			if(self.c[-1]<=0):					
				self.c.pop()
				break

			#update the bag weights
			weights_bag=[weights_bag[index_bag]*np.exp( (2*error_bag[bag_index]-1)*self.c[-1]  ) for index_bag in range(num_bags) ]
			sum_weights=float(sum(weights_bag)) 
			weights_bag= map( lambda x: x/sum_weights , weights_bag )
			
			#save current weak classifier 
			self.weak_classifiers.append(instance_classifier)
		self.actual_rounds_of_boosting = len(self.c)
		#import pdb; pdb.set_trace()	

	def predict(self, X_bags, iter = None):
		#X_bags is a list of arrays, each bag is an array in the list
		#The row of array corresponds to instances in the bag, column corresponds to feature

		#predictions_bag is the returned list of predictions which are real values 
		
		threshold = 0.5
		
		if iter == None or iter > len(self.c):
			iter = len(self.c)

		num_bags=len(X_bags)

		predictions_bag=[]
		print "self.c: ",
		print len(self.c)
		#print self.c
		#print len(self.weak_classifiers)
		for index_bag in range(num_bags):
			#import pdb;pdb.set_trace()
			predictions_bag_temp=np.average( [ np.average( instance_classifier.predict(X_bags[index_bag]) )  for instance_classifier in self.weak_classifiers  ][0:iter]  ,  weights=self.c[0:iter]  )/np.sum(self.c[0:iter]) - threshold
			predictions_bag.append(predictions_bag_temp)
		import pdb; pdb.set_trace()

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

		
