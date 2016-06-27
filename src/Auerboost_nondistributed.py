"""
This is implementation of boosting for multiple instance learning from 

Auer, Peter, and Ronald Ortner. "A boosting approach to multiple instance learning." European Conference on Machine Learning. Springer Berlin Heidelberg, 2004.
"""

import sys
import numpy as np
import unittest

class Ball(object):
	"""
	closed ball
	"""
	def __init__(self, center, radius):
		self.center = center
		self.radius = radius

	def getDistributionAccurary(self, X_bags, y_labels, weight_hashmap = None):
		"""
		return the distribution accuracy of Ball with respect to X_bags	

		Here I assume y_labels is +/- 1
		If weigh_hashmap is None, then uniform distribution is assumed
		"""
		num_bags = len(X_bags)

		weight_temp = {}
		
		if weight_hashmap is None:
			for i in weight_hashmap.keys():
				weight_temp[i] = float(1)/num_bags
		else:
			for i in weight_hashmap.keys():
				weight_temp[i] = weight_hashmap[i].weight		
			
		accuracy = 0
		for i in range(num_bags):
			bag_prediction = -1
			for j in range(X_bags[i].shape[0]):
				dist = np.linalg.norm( self.center - X_bags[i][j,:]  )
				if dist <= self.radius:
					bag_prediction = 1
					break
			
			if y_labels[i] == bag_prediction:
				accuracy += weight_temp[i]
		return accuracy
	def predict_inst(self, instances):
		"""
		2D array -> 1D array

		like np.array([1,2,3],[4,5,6]) -> np.array([1, -1])

		Each row in instances(2D array) corresponds to an instance
		"""
		num_instance = instances.shape[0]
		predictions = []

		for i in range(num_instance):
			dist = np.linalg.norm(self.center - instances[i,:])
			if dist <= self.radius:
				predictions.append(1)
			else:
				predictions.append(-1)
		return np.array(predictions)

	def predict(self, bag):
		"""
		return the prediction of one single bag: +/- 1

		bag is a 2D array, where the row corresponds to instance		
		Here I assume y_labels is +/- 1
		"""

		bag_prediction = -1
		for j in range(bag.shape[0]):
			dist = np.linalg.norm(self.center - bag[j,:])
			if dist <= self.radius:
				bag_prediction = 1
				break
		return bag_prediction



class WeightBag(object):
	def __init__(self, bag_index):
		self.bag_index = bag_index
		self.weight  = None
		self.distances = {} #self.distances[(i,ii)] is the shortest distance from ii-th instance in i-th bag to this bag
	
	def update_dist(self, bag_index, inst_index, dist):
		if (bag_index, inst_index) not in self.distances:
			self.distances[(bag_index, inst_index)] = sys.maxint
		if dist < self.distances[(bag_index, inst_index)]:
			self.distances[(bag_index, inst_index)] = dist

class WeightBagSet(object):
	def __init__(self, hashmap, X_bags, y_labels):
		self.X_bags = X_bags
		self.hashmap =  hashmap
		self.y_labels = y_labels
		list_from_hashmap_positive = [(index, hashmap[index]) for index in hashmap.keys() if self.y_labels[index] == 1]
		list_from_hashmap_negative = [(index, hashmap[index]) for index in hashmap.keys() if self.y_labels[index] != 1]

		self.hash_temp_positive = {}
		self.hash_temp_negative = {}	
		for bag_index, inst_index in hashmap[0].distances.keys():
			self.hash_temp_positive[(bag_index, inst_index)]=sorted(list_from_hashmap_positive, key = lambda x: x[1].distances[(bag_index, inst_index)])
			self.hash_temp_negative[(bag_index, inst_index)]=sorted(list_from_hashmap_negative, key = lambda x: x[1].distances[(bag_index, inst_index)])
	
	def getOptimalBall(self):
		#import pdb;pdb.set_trace()
		center_optimal = None
		radius_optimal = None
		sum_optimal = None

		for bag_index, inst_index in self.hashmap[0].distances.keys():

			center = (bag_index, inst_index)
			running_sum = sum([x[1].weight for x in self.hash_temp_negative[(bag_index, inst_index)] ])
			radius = 0
			
			if sum_optimal is None or sum_optimal < running_sum:
				center_optimal = center
				radius_optimal = radius

				sum_optimal = running_sum 

			i_p = 0
			i_n = 0
		
			#if bag_index == 5:
			#	print "need to debug"
			#	import pdb;pdb.set_trace()

			while i_p < len(self.hash_temp_positive[(bag_index, inst_index)]) and i_n < len(self.hash_temp_negative[(bag_index, inst_index)]):
				#print "i_p: ", i_p, " ;i_n: ", i_n
				dist_p = self.hash_temp_positive[(bag_index, inst_index)][i_p][1].distances[(bag_index, inst_index)]
				dist_n = self.hash_temp_negative[(bag_index, inst_index)][i_n][1].distances[(bag_index, inst_index)]

				if dist_p == dist_n:
					radius = dist_p
					dist_temp = dist_p
					while i_p < len(self.hash_temp_positive[(bag_index, inst_index)]) and dist_temp == dist_p:
						
						running_sum += self.hash_temp_positive[(bag_index, inst_index)][i_p][1].weight
						i_p += 1
						if i_p < len(self.hash_temp_positive[(bag_index, inst_index)]):
							dist_p = self.hash_temp_positive[(bag_index, inst_index)][i_p][1].distances[(bag_index, inst_index)]
						else:
							break
						
					while i_n < len(self.hash_temp_negative[(bag_index, inst_index)]) and dist_temp == dist_n:
						
						running_sum -= self.hash_temp_negative[(bag_index, inst_index)][i_n][1].weight
						i_n += 1

						if i_n < len(self.hash_temp_negative[(bag_index, inst_index)]):
							dist_n = self.hash_temp_positive[(bag_index, inst_index)][i_n][1].distances[(bag_index, inst_index)]
						else:
							break
						
				elif  dist_p < dist_n:
					
					running_sum += self.hash_temp_positive[(bag_index, inst_index)][i_p][1].weight
					radius = dist_p
					i_p += 1 
				else:	
					running_sum -= self.hash_temp_positive[(bag_index, inst_index)][i_p][1].weight					
					radius = dist_n
					i_n += 1
					
				if running_sum > sum_optimal:
					center_optimal = center
					radius_optimal = radius
					sum_optimal = running_sum 
			
			while i_p < len(self.hash_temp_positive):		
				running_sum += self.hash_temp_positive[(bag_index, inst_index)][i_p][1].weight
				radius = dist_p
				i_p += 1 

			if running_sum > sum_optimal:
				center_optimal = center
				radius_optimal = radius
				sum_optimal = running_sum 
			#import pdb;pdb.set_trace()

		center_vector = self.X_bags[center_optimal[0]][ center_optimal[1] ,:]
		return Ball(center_vector, radius_optimal)
				
class Auerboost(object):
	def __init__(self, **parameters):

		self.max_iter_boosting = parameters.pop("max_iter_boosting", 10)
		self.weak_classifier_name = parameters.pop('weak_classifier', 'ball') 
		if self.weak_classifier_name == 'dtree_stump':
			parameters['max_depth'] = 1
		parameters.pop('normalization', 0)
		self.parameters = parameters

		self.weak_classifiers = []

		self.X_bags_test = None
		self.X_bags = None
		self.y_labels = None
		self.X_instances = None

		self.errors = []
		self.alphas = []

		self.instance_labels_generated_from_bag_labels = None

	def fit(self, X_bags, y_labels):
		'''
		X_bags is a list of arrays, each bag is an array in the list
		The row of array corresponds to instances in the bag, column corresponds to feature
		y_labels is the list which contains the labels of bags. Here, binary labels are assumed, i.e. +1/-1
		'''
		self.X_bags = X_bags
		self.y_labels = y_labels

		self.X_instances = np.vstack((X_bags))

		num_bag = len(X_bags)		

		hashmap = {} #hashmap is a dictionary which map the bag index to class WeightBag, which maintains the latest weight for each bag
		for i in range(len(X_bags)):
			
			if y_labels[i] != 1:
				continue
			for ii in range(X_bags[i].shape[0]):
				
				for j in range(len(X_bags)):
					for jj in range(X_bags[j].shape[0]):
						if j not in hashmap:
							hashmap[j] = WeightBag(j)
							hashmap[j].weight = float(1)/num_bag

						dist =np.linalg.norm( X_bags[i][ii,:] - X_bags[j][jj,:] )
						hashmap[j].update_dist(i,ii, dist)
	

		
		weight_bag_set = WeightBagSet(hashmap, X_bags, y_labels)
		#print "have constructed weight_bag_set"

		

		for t in range(self.max_iter_boosting):
			ball = weight_bag_set.getOptimalBall()
			error = 1 - ball.getDistributionAccurary(X_bags, y_labels, hashmap)
			
			if error >= 0.5:
				break	

			self.weak_classifiers.append(ball)
			self.errors.append( error )
			alpha = 0.5*np.log((1-error)/(error))
			self.alphas.append(alpha)
			
			

			#update weights		
			Z = 0 #normalization factor
			for i in range(num_bag):
				hashmap[i].weight = np.exp(-alpha*ball.predict(X_bags[i])*y_labels[i])*hashmap[i].weight				
				Z += hashmap[i].weight
			
			for i in range(num_bag):
				hashmap[i].weight /= Z
			#import pdb;pdb.set_trace()
		self.actual_rounds_of_boosting = len(self.alphas)

	def predict_train(self, iter = None, getInstPrediction = False):
		if iter == None or iter > self.actual_rounds_of_boosting:
			iter = self.actual_rounds_of_boosting


		predictions_list = [instance_classifier.predict_inst(self.X_instances).reshape((1, -1))  for instance_classifier in self.weak_classifiers ]
		predictions_accum = np.matrix(self.alphas[0:iter])*np.matrix( np.vstack((predictions_list[0:iter])) )/np.sum(self.alphas[0:iter])
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

		self.c = self.alphas

		if iter == None or iter > self.actual_rounds_of_boosting:
			iter = self.actual_rounds_of_boosting

		if X is not None:	
			predictions_list = [instance_classifier.predict_inst(X).reshape((1, -1))  for instance_classifier in self.weak_classifiers ]
			self.predictions_list_test = predictions_list
			#import pdb;pdb.set_trace()
			predictions_accum = np.matrix(self.c[0:iter])*np.matrix( np.vstack((predictions_list[0:iter])) )/np.sum(self.c[0:iter])

			#import pdb;pdb.set_trace()
			return np.array(predictions_accum)[0]   #entries within range [-1, 1] since instance_classifier.predict  is either -1 or 1
		else:
			predictions_accum = np.matrix(self.c[0:iter])*np.matrix( np.vstack((self.predictions_list_test[0:iter])) )/np.sum(self.c[0:iter])

			#import pdb;pdb.set_trace()
			return np.array(predictions_accum)[0]   #entries within range [-1, 1] since 2*(instance_classifier.predict >0) - 1 is either -1 or 1

	def predict(self, X_bags = None, iter = None, getInstPrediction = False):
		#X_bags is a list of arrays, each bag is an array in the list
		#The row of array corresponds to instances in the bag, column corresponds to feature

		#predictions_bag is the returned array of predictions which are real values 
		
		self.c = self.alphas
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





def get_bag_label(instance_predictions, bags):
	num_bag = len(bags)
	p_index= 0
	bag_predictions = []
	for bag_index in range(num_bag):
		n_index =p_index+ bags[bag_index].shape[0]
		
		bag_predictions.append( np.average(instance_predictions[p_index: n_index]) )
		p_index = n_index
	return np.array(bag_predictions)


class TestAuerboostFitMethod(unittest.TestCase):
	def test_WeightBagSet(self):
		X_bags =[np.array([[1,0]]),np.array([[-1,0]]),np.array([[0,1]]),np.array([[0,2]]),np.array([[0,-1]]), np.array([[0,0]]) ]
		y_labels = [-1,-1,1,1,1, 1]
		booster = Auerboost()
		
		booster.fit(X_bags, y_labels)

		print booster.predict_train()

		X_bags_test = [np.array([[2,0]]), np.array([[-2, 0]]), np.array([[0,5]]),np.array([[0, 1.5]])]
		print booster.predict(X_bags = X_bags_test)
		import pdb;pdb.set_trace()

if __name__ == "__main__":
	unittest.main()



							