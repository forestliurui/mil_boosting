"""
This is the implementation of decision stump as ranker

The basic idea is the root node correponding to a feature. Assume the feature takes on {v1, v2, ..., v5}. Then all examples will be paritioned into 5 parts, each corresponding to
one of root's children. For examples, in node n1, all the examples contained in it has this feature to be v1. 

Each children node will be assigned a predicted score, either 0 or 1. The way to do it is by the following:

We receive a list of critical pairs as training data, like y = [p1 = (x1, x2), p2 =(x3, x4), ...]. We assume the former should be ranker higher than the latter, i.e. x1(x3) higher than x2(x4)
We also have  a list of weights for each pair, like weight_pair = [w1, w2, ...]. 

For each example in a specific node, we check if it appears as the former point in some critical pair p_i. If so, we get a score of w_i. If as latter point, we
get a score of -w_i. Add scores for examples in all critical pairs, we get a final score s. If s>=0, we assign +1 to this node as predicted socre, otherwise 0.
"""

import numpy as np
import math
import unittest

class StumpRanker_derived(object):
	def __init__(self, feature_index, children_nodes_prediction):
		self.feature_index = feature_index
		self.children_nodes_prediction = children_nodes_prediction

class StumpRanker(object):
	def __init__(self):
		self.feature_index = None
		self.children_nodes_prediction = {}
	def fit(self, X, y, weight_pair = None):
		"""
		X is a hashtable with key being instance ID, value being the one-dimensional array containing its features
		y is the critial pairs, pair[0] should be ranked higher than pair[1]

		Assume the feature values are discrete.
		"""
		
		if weight_pair is None:
			weight_pair = {}
			for pair in y:
				weight_pair[pair] = float(1)/len(y)
		
		num_feature = len(X.values()[0]) 

		weight_dict = {}
		
		for pair in weight_pair:
			if pair[0] not in weight_dict:
				weight_dict[pair[0]] = 0
			weight_dict[pair[0]] += weight_pair[pair]

			if pair[1] not in weight_dict:
				weight_dict[pair[1]] = 0
			weight_dict[pair[1]] -= weight_pair[pair]
			
		score_optimal = None
		nodes_prediction_optimal = None
		feature_index_optimal = None
		for index in range(num_feature):
			score, nodes_prediction = self.getScore(X, y, weight_dict,weight_pair, index)
			
			if score_optimal is None or score_optimal < score:
				score_optimal = score
				nodes_prediction_optimal = nodes_prediction
				feature_index_optimal = index
		self.feature_index = feature_index_optimal
		self.children_nodes_prediction = nodes_prediction_optimal

		

	def predict(self, X):
		"""
		X is a hashtable
		"""
		predictions = {}
		for inst_index in X.keys():
			if X[inst_index][self.feature_index] in self.children_nodes_prediction:
				predictions.update({inst_index: self.children_nodes_prediction [X[inst_index][self.feature_index] ] } )
			else:
				predictions.update({inst_index: 0} )

		return predictions


	def getScore(self, X, y, weight_dict, weight_pair, feature_index):
		partition = {}
		missing_value_set = []		

		for i in X.keys():
			if X[i][feature_index] == -1:  #feature of value -1 indicates it's a missing value
				missing_value_set.append(i)
			else:
				if X[i][feature_index] not in partition:
					partition[X[i][feature_index]] = []

				partition[X[i][feature_index]].append(i)

		size_partition = {}
		radom_num_thresh_partition = {}
		temp_size = 0
		for val in partition.keys():
			size_partition[val] = len(partition[val])
			temp_size += size_partition[val] 
			radom_num_thresh_partition[temp_size] = val
		cand_thresh = sorted(radom_num_thresh_partition.keys(), reverse = True)
		#distribute instances with missing values
		for i in missing_value_set:
			r_num = np.random.uniform(low = 0, high= temp_size) 
			j = 0
			while j < len(cand_thresh) and cand_thresh[j] >= r_num:
				i_to_join =  radom_num_thresh_partition[cand_thresh[j]]
				j+=1
			partition[i_to_join].append(i)

		nodes_prediction = {}
		predictions = {}
		for val in partition.keys():
			score = sum([weight_dict[x]  for x in partition[val] ])
			if score >= 0:
				nodes_prediction[val] = 1
			else:
				nodes_prediction[val] = 0
			for i in partition[val]:
				predictions[i] = nodes_prediction[val]
			
		score = 0
		for pair in y:
			if predictions[pair[0]]> predictions[pair[1]]:
				score += weight_pair[pair]

		
		
		return score, nodes_prediction

class TestPredictMethod(unittest.TestCase):
	def test_predict_uniform_weight(self):
		X = {0:np.array([2,0]),1:np.array([3,1]), 2:np.array([4,2])}
		y = [(1, 0), (2, 1)]

		ranker = StumpRanker()
		ranker.fit(X, y)
		print ranker.predict(X)

		X_test = {4: np.array([2,4])}
		print ranker.predict(X_test)

		#import pdb;pdb.set_trace()

class TestFitMethod(unittest.TestCase):
	def test_fit_uniform_weight(self):
		X = {0:np.array([2,0]),1:np.array([3,1]), 2:np.array([4,2])}
		y = [(1, 0), (2, 1)]

		ranker = StumpRanker()
		ranker.fit(X, y)
		#import pdb;pdb.set_trace()

	def test_fit_uniform_weight_with_missing_value(self):
		X = {0:np.array([2,0]),1:np.array([3,1]), 2:np.array([-1,2]), 3:np.array([-1, 3])}
		y = [(1, 0), (2, 1), (3,2)]

		ranker = StumpRanker()
		ranker.fit(X, y)
		import pdb;pdb.set_trace()

	def test_fit_nonuniform_weight(self):
		X = {0:np.array([2,0]),1:np.array([3,1]), 2:np.array([4,2])}
		y = [(1, 0), (2, 1)]
		weight_pair = {(1,0):0.3, (2,1): 0.7}

		ranker = StumpRanker()
		ranker.fit(X, y, weight_pair)
		#import pdb;pdb.set_trace()

	def test_fit_nonuniform_weight1(self):
		X = {0:np.array([2,0]),1:np.array([2,1]), 2:np.array([4,2])}
		y = [(0, 1), (2, 1)]
		weight_pair = {(0,1):0.3, (2,1): 0.7}

		ranker = StumpRanker()
		ranker.fit(X, y, weight_pair)
		print " "
		print ranker.feature_index
		print ranker.children_nodes_prediction
		#import pdb;pdb.set_trace()

if __name__ == "__main__":
	unittest.main()

