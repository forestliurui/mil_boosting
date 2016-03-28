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
		self.error_bags = []
		self.weights_bags = []
		self.weights_instances = []

		self.X_bags_test = None
		self.X_bags = None
		self.y_labels = None
		self.X_instances = None

		self.instance_labels_generated_from_bag_labels = None


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
		
		self.X_bags = X_bags
		self.y_labels = y_labels 
		self.X_instances = instances	
		self.instance_labels_generated_from_bag_labels = instance_labels_generated_from_bag_labels		


		self.c=[] #the list of weights for weak classifiers

		#initial bag weights
		weights_bag=np.ones((num_bags))/float(num_bags)
		self.weights_bags.append(weights_bag)

		for index_Boosting in range(max_iter_boosting):
			
			weights_instance= [( weights_bag[bag_index]/float(num_instance_each_bag[bag_index]) )*np.ones((1, num_instance_each_bag[bag_index]))[0] for bag_index in range(num_bags)]
			weights_instance=np.hstack((weights_instance))
			self.weights_instances.append(weights_instance)
			instance_classifier=WEAK_CLASSIFIERS[self.weak_classifier_name](**self.parameters)

			instance_classifier.fit(instances, instance_labels_generated_from_bag_labels, weights_instance)

			error_bag=[ np.average( ( instance_classifier.predict(X_bags[index_bag]) >0  ) != (y_labels[index_bag]==1) )  for index_bag in range(num_bags)  ]
			self.error_bags.append(error_bag)

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
			weights_bag=[weights_bag[index_bag]*np.exp( (2*error_bag[index_bag]-1)*self.c[-1]  ) for index_bag in range(num_bags) ]
			sum_weights=float(sum(weights_bag)) 
			weights_bag= map( lambda x: x/sum_weights , weights_bag )
			self.weights_bags.append(weights_bag)


			
			#save current weak classifier 
			self.weak_classifiers.append(instance_classifier)
		self.actual_rounds_of_boosting = len(self.c)
		#import pdb; pdb.set_trace()	

	def predict_train(self, iter = None, getInstPrediction = False):
		
		

		if iter == None or iter > len(self.c):
			iter = len(self.c)
		
		predictions_list = [2*(instance_classifier.predict(self.X_instances).reshape((1, -1))>0) - 1  for instance_classifier in self.weak_classifiers ]
		predictions_accum = np.matrix(self.c[0:iter])*np.matrix( np.vstack((predictions_list[0:iter])) )/np.sum(self.c[0:iter])
		results = np.array(predictions_accum)[0] 
		
		if getInstPrediction:
			return results
		else: #get bag level predictions
			if self.X_bags is None:
				raise Exception("Can't get bag level prediction for training data due to the lack of training bag data")
			else:
				predictions_bag = get_bag_label(results, self.X_bags)
				return predictions_bag



	def _predict(self, X = None, iter = None):
		"""
		X is assumed to be two dimensional array, each row corresponding to an instance
		"""

		

		if iter == None or iter > self.actual_rounds_of_boosting:
			iter = self.actual_rounds_of_boosting

		if X is not None:	
			predictions_list = [2*(instance_classifier.predict(X).reshape((1, -1))>0) - 1  for instance_classifier in self.weak_classifiers ]
			self.predictions_list_test = predictions_list
			#import pdb;pdb.set_trace()
			predictions_accum = np.matrix(self.c[0:iter])*np.matrix( np.vstack((predictions_list[0:iter])) )/np.sum(self.c[0:iter])

			#import pdb;pdb.set_trace()
			return np.array(predictions_accum)[0]   #entries within range [-1, 1] since 2*(instance_classifier.predict >0) - 1 is either -1 or 1
		else:
			predictions_accum = np.matrix(self.c[0:iter])*np.matrix( np.vstack((self.predictions_list_test[0:iter])) )/np.sum(self.c[0:iter])

			#import pdb;pdb.set_trace()
			return np.array(predictions_accum)[0]   #entries within range [-1, 1] since 2*(instance_classifier.predict >0) - 1 is either -1 or 1

	def predict(self, X_bags = None, iter = None, getInstPrediction = False):
		#X_bags is a list of arrays, each bag is an array in the list
		#The row of array corresponds to instances in the bag, column corresponds to feature

		#predictions_bag is the returned array of predictions which are real values 
		
		
		if iter == None or iter > len(self.c):
			iter = len(self.c)
	
		#print "self.c: ",
		print len(self.c)
		if X_bags is not None:
			self.X_bags_test = X_bags

			if type(X_bags) != list:  # treat it as normal supervised learning setting
				#X_bags = [X_bags[inst_index,:] for inst_index in range(X_bags.shape[0])]
				return self._predict(X = X_bags, iter = iter)
				
			else:

				X_instances = np.vstack(X_bags)
				predictions_accum = self._predict(X = X_instances, iter =  iter)
				if getInstPrediction:  #return the instance level predictions for the input bags
					return predictions_accum
				else:
					predictions_bag = get_bag_label(predictions_accum, X_bags)
					return predictions_bag

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





	def predict_old(self, X_bags, iter = None):
		#X_bags is a list of arrays, each bag is an array in the list
		#The row of array corresponds to instances in the bag, column corresponds to feature

		
		if iter == None or iter > len(self.c):
			iter = len(self.c)
	
		print "self.c: ",
		print len(self.c)
		if type(X_bags) != list:  # treat it as normal supervised learning setting
			#X_bags = [X_bags[inst_index,:] for inst_index in range(X_bags.shape[0])]
			predictions_list = [2*(instance_classifier.predict(X_bags).reshape((1, -1))>0) - 1  for instance_classifier in self.weak_classifiers ]
			#import pdb;pdb.set_trace()
			predictions_accum = np.matrix(self.c[0:iter])*np.matrix( np.vstack((predictions_list[0:iter])) )/np.sum(self.c[0:iter])

			#import pdb;pdb.set_trace()
			return np.array(predictions_accum)[0]   #entries within range [-1, 1] since 2*(instance_classifier.predict >0) - 1 is either -1 or 1
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
