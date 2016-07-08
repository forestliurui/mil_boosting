"""
This is the implementation of rBoost from 

Bootkrajang, Jakramate, and Ata Kaban. "Boosting in the presence of label noise." Uncertainty in Artificial Intelligence. 2013.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.optimize import minimize
from numpy import sinh, exp
import unittest

WEAK_CLASSIFIERS = {
	'dtree_stump': DecisionTreeClassifier,
}

class RBoost(object):
	def __init__(self, **parameters):

		self.max_iter_boosting = parameters.pop("max_iter_boosting", 10)
		self.weak_classifier_name = parameters.pop('weak_classifier', 'dtree_stump') 
		if self.weak_classifier_name == 'dtree_stump':
			parameters['max_depth'] = 1
		parameters.pop('normalization', 0)

		self.parameters = parameters

		self.X_bags_test = None
		self.X_bags = None
		self.y_labels = None
		self.X_instances = None

		self.instance_labels_generated_from_bag_labels = None
		self.weak_classifiers = []
		self.alphas = []
		self.predictions_list_train = []
		self.instance_weights = []

	def fit(self, X_bags, y_labels):
		'''
		X_bags is either (1) a list of arrays, each bag is an array in the list -- multiple instance learning  or (2) a single array -- supervised learning
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
			self.instance_labels_generated_from_bag_labels = instance_labels_generated_from_bag_labels		
		else:
			instances = X_bags
			instance_labels_generated_from_bag_labels = y_labels

			self.instance_labels_generated_from_bag_labels = instance_labels_generated_from_bag_labels

			self.X_instances = instances


		instance_labels_generated_from_bag_labels = np.array( instance_labels_generated_from_bag_labels )
		num_instances = instances.shape[0]
		num_instances_positive = np.sum(instance_labels_generated_from_bag_labels == 1)	
		num_instances_negative = np.sum(instance_labels_generated_from_bag_labels != 1)		

		# gamma[(j,k)] = p(\tilde{y} = k | y = j) i.e. the probablity of flipping true label y = j into observed label \tilde{y} = k
		#initialization of gamma_jk
		self.gamma = {}

		self.gamma[(0,0)] = [0.7]
		self.gamma[(0,1)] = [0.3]
		
		#self.gamma[(1,0)] = [0] #true positive instance will always remain as true observed instance
		#self.gamma[(1,1)] = [1]
		
		self.gamma[(1,0)] = [0.2] #true positive instance will always remain as true observed instance
		self.gamma[(1,1)] = [0.8]

		self.w = {}
		self.P  ={}
		self.g = {}
		for index_inst in range(num_instances):
			self.w[index_inst] = {}
			for i in [0,1]:
				for j in [0,1]:
					self.w[index_inst][(i,j)] = self.gamma[(i,j)][-1]

		for index_Boosting in range(self.max_iter_boosting):
			print "boosting iteration: ", index_Boosting
			weights_instance = self.getInstWeight(instance_labels_generated_from_bag_labels, self.w)
			self.instance_weights.append(weights_instance)

			#import pdb;pdb.set_trace()
			instance_classifier=WEAK_CLASSIFIERS[self.weak_classifier_name](**self.parameters)
			instance_classifier.fit(instances, instance_labels_generated_from_bag_labels, np.array(weights_instance) )
			
			self.predictions_list_train.append( instance_classifier.predict(instances) )
			
			weighted_error = np.average( instance_classifier.predict(instances) !=  self.predictions_list_train[-1], weights = weights_instance)
			res = minimize(self.subproblem_to_update_alpha(instance_labels_generated_from_bag_labels, self.w,  weighted_error ), 0)	
			alpha  = res['x'][0]

			self.alphas.append(alpha)
			self.weak_classifiers.append(instance_classifier)

			#update self.w
			for index_inst in range(num_instances):

				self.w[index_inst][(0,0)] = self.gamma[(0,0)][-1]*exp(-self.predict(instances[index_inst,:].reshape((1,-1))))[0]
				self.w[index_inst][(0,1)] = self.gamma[(0,1)][-1]*exp(self.predict(instances[index_inst,:].reshape((1,-1))))[0]
				self.w[index_inst][(1,0)] = self.gamma[(1,0)][-1]*exp(-self.predict(instances[index_inst,:].reshape((1,-1))))[0]
				self.w[index_inst][(1,1)] = self.gamma[(1,1)][-1]*exp(self.predict(instances[index_inst,:].reshape((1,-1))))[0]

			#update P Note: there are several ways to compute P. For simplicity, I use logistic calibratioin here
			for index_inst in range(num_instances):
				self.P[index_inst] = float(1)/(  1+exp(-self.predict(instances[index_inst,:].reshape((1,-1))))[0]  )
			self.g[(1, 1)] = self.gamma[(1,1)][-1]*sum([  (instance_labels_generated_from_bag_labels[index_inst]==1)*self.P[index_inst]/(self.gamma[(1,1)][-1]*self.P[index_inst]+self.gamma[(0,1)][-1]*(1-self.P[index_inst]) )   for index_inst in range(num_instances)  ])
			self.g[(1, 0)] = self.gamma[(1,0)][-1]*sum([  (instance_labels_generated_from_bag_labels[index_inst]!=1)*self.P[index_inst]/(self.gamma[(1,0)][-1]*self.P[index_inst]+self.gamma[(0,0)][-1]*(1-self.P[index_inst]) )   for index_inst in range(num_instances)  ])
			self.g[(0, 1)] = self.gamma[(0,1)][-1]*sum([  (instance_labels_generated_from_bag_labels[index_inst]==1)*(1-self.P[index_inst])/(self.gamma[(1,1)][-1]*self.P[index_inst]+self.gamma[(0,1)][-1]*(1-self.P[index_inst]) )   for index_inst in range(num_instances)  ])
			self.g[(0, 0)] = self.gamma[(0,0)][-1]*sum([  (instance_labels_generated_from_bag_labels[index_inst]!=1)*(1-self.P[index_inst])/(self.gamma[(1,0)][-1]*self.P[index_inst]+self.gamma[(0,0)][-1]*(1-self.P[index_inst]) )   for index_inst in range(num_instances)  ])

			#update gamma
			self.gamma[(0,0)].append( self.g[(0,0)]/(self.g[(0,0)]+self.g[(0,1)]) )
			self.gamma[(0,1)].append( self.g[(0,1)]/(self.g[(0,0)]+self.g[(0,1)]) )		
			self.gamma[(1,0)].append( self.g[(1,0)]/(self.g[(1,0)]+self.g[(1,1)]) )
			self.gamma[(1,1)].append( self.g[(1,1)]/(self.g[(1,0)]+self.g[(1,1)]) )
		#import pdb;pdb.set_trace()
		self.actual_rounds_of_boosting = len(self.alphas)
					
	def predict_train(self, iter = None, getInstPrediction = False):


		self.c = self.alphas
		threshold = 0.5
		if iter == None or iter > len(self.c):
			iter = len(self.c)

		predictions_accum = np.matrix(self.c[0:iter])*np.matrix( np.vstack((self.predictions_list_train[0:iter])) )/np.sum(self.c[0:iter])
		results = np.array(predictions_accum)[0] - threshold

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
		
		self.c = self.alphas
		threshold = 0.5

		if iter == None or iter > len(self.c):
			iter = len(self.c)
		if X is not None:	

			predictions_list = [( instance_classifier.predict(X).reshape((1, -1))>0 )+ 0 for instance_classifier in self.weak_classifiers ]
			self.predictions_list_test = predictions_list
			#import pdb;pdb.set_trace()
			predictions_accum = np.matrix(self.c[0:iter])*np.matrix( np.vstack((predictions_list[0:iter])) )/np.sum(self.c[0:iter])

			#import pdb;pdb.set_trace()
			return np.array(predictions_accum)[0] - threshold   #we need to deduct a threshold because (instance_classifier.predict > 0) + 0 is either 0 or 1
		else:
			predictions_accum = np.matrix(self.c[0:iter])*np.matrix( np.vstack((self.predictions_list_test[0:iter])) )/np.sum(self.c[0:iter])

			#import pdb;pdb.set_trace()
			return np.array(predictions_accum)[0] - threshold   #we need to deduct a threshold because (instance_classifier.predict > 0) + 0 is either 0 or 1

	def predict(self, X_bags = None, iter = None, getInstPrediction = False):
		#X_bags is a list of arrays, each bag is an array in the list
		#The row of array corresponds to instances in the bag, column corresponds to feature

		#predictions_bag is the returned array of predictions which are real values 
		
		self.c = self.alphas
		if iter == None or iter > len(self.c):
			iter = len(self.c)
	
		#print "self.c: ",
		#print len(self.c)
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


	def subproblem_to_update_alpha(self, labels, w, weighted_error):
		"""
		labels is one-D array of labels
		w is a dictionary containing w[(0,0)], w[(0,1)], w[(1,0)], w[(1,1)]

		weighted_error is a real number, which indicates the traning error of current weak classifier weighted by weights_instance
		"""

		term1 = 0
		term2 = 0
		for i in range(labels.shape[0]):
			if labels[i] == 1:
				term1 += w[i][(0,0)]
				term2 += w[i][(0,1)]
			else:
				term1 += w[i][(1,1)]
				term2 += w[i][(1,0)]


		return lambda x: 2*sinh(x)*weighted_error + exp(-x)*term1+exp(x)*term2
		


	def getInstWeight(self, labels, w):
		"""
		labels is one-D array of labels. Here, binary labels are assumed, i.e. +1/-1
		w is dictionary, which maps keys to real values. Key is taken value from [(0,0), (0,1), (1, 0), (1,1)]


		return the instance-level weights as a list
		"""

		result = []
		
		for i in range(labels.shape[0]):
			
			if labels[i] == 1:
				result.append(w[i][(0,0)] - w[i][(0, 1)])
			else:
				result.append(w[i][(1,1)] - w[i][(1, 0)])
		return result

def get_bag_label(instance_predictions, bags):
	num_bag = len(bags)
	p_index= 0
	bag_predictions = []
	for bag_index in range(num_bag):
		n_index =p_index+ bags[bag_index].shape[0]
		
		bag_predictions.append( np.max(instance_predictions[p_index: n_index]) )
		p_index = n_index
	return np.array(bag_predictions)

class TestRBoostFitMethod(unittest.TestCase):
	def test_fit1(self):
		X_bags = [np.array([[-1, 0]]), np.array([[1,0]]), np.array([[0, 1]]), np.array([[0, -1]])]
		y_labels = [-1, -1, 1, 1]
		param = {"max_iter_boosting": 50}
		rbooster = RBoost(**param)

		rbooster.fit(X_bags, y_labels)
		print rbooster.predict(X_bags)
		import pdb;pdb.set_trace()
	def no_test_fit1(self):
		X_bags = [np.array([[-1, 1],[1, 1]]), np.array([[-2,1],[-2, -1]]), np.array([[2, 1], [2, -1]]), np.array([[-1, -1], [1, -1]])]
		y_labels = [-1, 1, 1, 1]
		param = {"max_iter_boosting": 100}
		rbooster = RBoost(**param)

		rbooster.fit(X_bags, y_labels)
		print rbooster.predict(X_bags)
		

		X_bags_test= [np.array([[0, 1]]), np.array([[0, 2]]), np.array([[0, -1]]), np.array([[0, -2]]), np.array([[2, 0]]), np.array([[3, 0]]), np.array([[-2, 0]]), np.array([[-3, 0]])]
		print rbooster.predict(X_bags_test)
		import pdb;pdb.set_trace()

if __name__ == "__main__":
	unittest.main()







