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

class StumpRanker(object):
	def __init__(object):
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

		for index in range(num_feature):
			

