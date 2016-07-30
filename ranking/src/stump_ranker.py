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
import random

class StumpRanker_derived(object):
	def __init__(self, feature_index, children_nodes_prediction):
		self.feature_index = feature_index
		self.children_nodes_prediction = children_nodes_prediction

class StumpRanker(object):
	def __init__(self):
		self.feature_index = None
		self.children_nodes_prediction = {}
		self.threshold = None
	@staticmethod
	def create(type):
		if type == "discrete":
			return StumpRanker_discrete()
		else:
			return StumpRanker_ContinuousFeature()			

	def fit(self, X, y, weight_pair = None):
		"""
		X is a hashtable with key being instance ID, value being the one-dimensional array containing its features
		y is the list of tuples -- each tuple is one critial pair, pair[0] should be ranked higher than pair[1]

		Assume the feature values are discrete.
		"""
		
		if weight_pair is None:
			weight_pair = {}
			for pair in y:
				weight_pair[pair] = float(1)/len(y)
		
		num_feature = len(X.values()[0]) 

		weight_dict = self.get_weight_dict(weight_pair)
			
		score_optimal = None
		nodes_prediction_optimal = None
		feature_index_optimal = None
		threshold_optimal = None
		for index in range(num_feature):
			score, nodes_prediction, threshold_temp = self.getScore(X, y, weight_dict,weight_pair, index)
			
			if score_optimal is None or score_optimal < score:
				score_optimal = score
				nodes_prediction_optimal = nodes_prediction
				feature_index_optimal = index
				threshold_optimal = threshold_temp

		self.feature_index = feature_index_optimal
		self.children_nodes_prediction = nodes_prediction_optimal
		self.threshold = threshold_optimal
		

	def predict(self, X):
		"""
		X is a hashtable
		"""
		raise  NotImplementedError("Please Implement this method in derived class")

	def get_weight_dict(self, weight_pair):
		weight_dict = {}
		
		for pair in weight_pair:
			if pair[0] not in weight_dict:
				weight_dict[pair[0]] = 0
			weight_dict[pair[0]] += weight_pair[pair]

			if pair[1] not in weight_dict:
				weight_dict[pair[1]] = 0
			weight_dict[pair[1]] -= weight_pair[pair]
		return weight_dict


	def getScore(self, X, y, weight_dict, weight_pair, feature_index):
		raise NotImplementedError("Please Implement this method")





class StumpRanker_ContinuousFeature(StumpRanker):
	def __init__(self):
		super(StumpRanker_ContinuousFeature, self).__init__()

	def predict(self, X):
		"""
		X is a hashtable
		"""
		predictions = {}
		for inst_index in X.keys():
			if X[inst_index][self.feature_index] >= self.threshold:
				predictions.update({inst_index: self.children_nodes_prediction["+"] } )
			else:
				predictions.update({inst_index: self.children_nodes_prediction["-"] }  )

		return predictions

	def getScore(self, X, y, weight_dict,weight_pair, index):
		"""
		X is a hashtable with key being instance ID, value being the one-dimensional array containing its features
		y is the list of tuples -- each tuple is one critical pair, pair[0] should be ranked higher than pair[1]
		weight_pair is a dict: key is critical pair expressed in tuple, value is the distribution weight for this critical pair
		weight_dict is a dict: key is instance ID, value is the total weight received by this instance ID		

		Assume feature values are continuous
		"""

		feature = [X[i][index]  for i in X.keys()]
		sorted_feature = sorted(feature)

		#get the thresholds
		temp = [ sorted_feature[0]-1 ]+sorted_feature + [ sorted_feature[1]+1 ]
		raw_thresholds = [ (temp[i]+temp[i+1])/float(2)  for i in range(len(temp)-1)  ]

		thresholds_len_max  = 500 
		if len(raw_thresholds) > thresholds_len_max:
			thresholds = random.sample(raw_thresholds, thresholds_len_max)
		else:
			thresholds = raw_thresholds

		score_max = None
		nodes_predictions_max = None
		threshold_max = None
		for thred in thresholds:
			score, nodes_predictions = self.getScore_helper(X, y, weight_dict, weight_pair, index, thred)
			if score_max is None or score_max < score:
				score_max = score			
				nodes_predictions_max = nodes_predictions
				threshold_max = thred
		return score_max, nodes_predictions_max, threshold_max


	def getScore_helper(self, X, y, weight_dict, weight_pair, index, thred):
		"""
		get the score for a single partition with certain threshold (i.e. thred) for a certain feature indexed by index argument
		"""
		#according to whether feature value is greater than threshold, assign the index of an instance to either partition["+"] or partition["-"]
		partition = {"+":[], "-":[]}
		for inst_index in X.keys():
			if X[inst_index][index] >= thred:
				partition["+"].append(inst_index)
			else:
				partition["-"].append(inst_index)

		#nodes_predictions["+"] is the prediction value (1 or 0) for any instance that falls into this node, i.e. feature value >= threshold
		#nodes_predictions["-"] is the prediction value (1 or 0) for any instance that falls into this node, i.e. feature value < threshold
		nodes_predictions = {}
		
		#inst_predictions is a dict, with key being instance ID, value being the prediction value (1 or 0) for this instance ID
		inst_predictions = {}
		for key in partition.keys():
			temp = sum( [weight_dict[x] for x in  partition[key] ] )
			if temp >= 0:
				nodes_predictions[key] = 1
			else:
				nodes_predictions[key] = 0
			for inst_id in partition[key]:
				inst_predictions[inst_id] = nodes_predictions[key]
		
		#score is the weighted accuracy of this partition
		score = 0
		for pair in y:
			if inst_predictions[pair[0]] > inst_predictions[pair[1]]:
				score += weight_pair[pair]
		return score, nodes_predictions

	


class TestGetScoreHelper(unittest.TestCase):
	def test_getScore(self):
		X = {0:np.array([0.4]), 1:np.array([0.6]), 2:np.array([1.0]), 3:np.array([1.2])  }
		y = [(3,1),(2,0),(1,2)]
		weight_pair = { (3,1):0.4, (2,0):0.4, (1,2): 0.2 }
		
		ranker = StumpRanker_ContinuousFeature()	
		weight_dict = ranker.get_weight_dict(weight_pair)
		
		self.assertEqual(weight_dict, {0:-0.4, 1: -0.2, 2: 0.2, 3: 0.4})	

		index =0
		#thred = 0.8
		score, nodes_predictions, thred = ranker.getScore(X, y, weight_dict, weight_pair, index)

		self.assertEqual(thred, 0.8)
		self.assertEqual(nodes_predictions, {"+": 1, "-": 0})
		self.assertEqual(score, 0.8 )
		#import pdb;pdb.set_trace()
		
	def test_getScoreHelper(self):
		X = {0:np.array([0.4]), 1:np.array([0.6]), 2:np.array([1.0]), 3:np.array([1.2])  }
		y = [(3,1),(2,0),(1,2)]
		weight_pair = { (3,1):0.4, (2,0):0.4, (1,2): 0.2 }
		
		ranker = StumpRanker_ContinuousFeature()	
		weight_dict = ranker.get_weight_dict(weight_pair)
		
		self.assertEqual(weight_dict, {0:-0.4, 1: -0.2, 2: 0.2, 3: 0.4})	

		index =0
		thred = 0.8
		score, nodes_predictions = ranker.getScore_helper(X, y, weight_dict, weight_pair, index, thred)

		self.assertEqual(nodes_predictions, {"+": 1, "-": 0})
		self.assertEqual(score, 0.8 )
		#import pdb;pdb.set_trace()

	def test_class1(self):

		X = {0:np.array([0.4]), 1:np.array([0.6]), 2:np.array([1.0]), 3:np.array([1.2])  }
		y = [(3,1),(2,0),(1,2)]
		weight_pair = { (3,1):0.4, (2,0):0.4, (1,2): 0.2 }

		ranker = StumpRanker.create("continuous")
		ranker.fit(X, y, weight_pair)
	
		self.assertEqual( ranker.predict(X), {0:0, 1:0 , 2: 1, 3: 1} )
		#import pdb;pdb.set_trace()	

	def test_class(self):

		X = {0:np.array([1.2, 0.4]), 1:np.array([0.4, 0.6]), 2:np.array([0.8, 1.0]), 3:np.array([0, 1.2])  }
		y = [(3,1),(2,0),(1,2)]
		weight_pair = { (3,1):0.4, (2,0):0.4, (1,2): 0.2 }

		ranker = StumpRanker.create("continuous")
		ranker.fit(X, y, weight_pair)
	
		print ranker.predict(X)
		import pdb;pdb.set_trace()

class StumpRanker_discrete(StumpRanker):
	def __init__(self):
		super(StumpRanker_discrete, self).__init__()

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
		inst_predictions = {}
		for val in partition.keys():
			score = sum([weight_dict[x]  for x in partition[val] ])
			if score >= 0:
				nodes_prediction[val] = 1
			else:
				nodes_prediction[val] = 0
			for i in partition[val]:
				inst_predictions[i] = nodes_prediction[val]
			
		score = 0
		for pair in y:
			if inst_predictions[pair[0]]> inst_predictions[pair[1]]:
				score += weight_pair[pair]

		
		
		return score, nodes_prediction, None #the last None is to be consistent with the output of continuous version

"""
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
		#import pdb;pdb.set_trace()

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
"""
if __name__ == "__main__":
	unittest.main()

