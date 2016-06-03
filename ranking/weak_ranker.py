"""
This is the implementation of weak ranker with second method described in original rankboost paper
"""

import math
import numpy as np

class WeakRanker(object):
	def __init__(self):
		self.theta = None
		self.feature_index = None

	def fit(self, X, y, weight_pair = None):
		"""
		X is a hashtable with key being instance ID, value being the one-dimensional array containing its features
		y is the critial pairs, pair[0] should be ranked higher than pair[1]
		"""
		
		if weight_pair is None:
			weight_pair = {}
			for pair in y:
				weight_pair[pair] = float(1)/len(y)

		feature_max = np.max(X)
		feature_min = np.min(X)

		num_step = 100
			
		num_feature = len(X.values()[0])
		
		#threshold_candidates = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

		min_Z = None

		for index in range(num_feature):
			print "check feature No. ", index
			threshold_candidates = self.getThresholdCand(X, index)
			for threshold in threshold_candidates:
				Z = self.compute_Z(index, threshold, X, y, weight_pair)
				#import pdb;pdb.set_trace()
				if min_Z is None or min_Z > Z:
					self.theta = threshold
					self.feature_index = index
					min_Z = Z

	
	def getThresholdCand(self, X, index):
		featureSet = set([item[index] for item in X.values()  ])
		featureSet = sorted(featureSet)
		thresholdCan = []
		max_length = 100
		for i in range(1, len(featureSet)):		
			thresholdCan.append(np.average([featureSet[i-1], featureSet[i]]))
		
		return np.random.choice(thresholdCan, size = max_length, replace = False) if len(thresholdCan) > max_length else thresholdCan	

	def compute_Z(self, index, threshold, X, y, pair_weight):
		"""
		Here, Z is for modiII
		"""
		epsilon0= 0
		epsilon_pos = 0
		epsilon_neg = 0		

		for pair in y:
			prediction0 = (X[pair[0]][index] > threshold )+0
			prediction1 = (X[pair[1]][index] > threshold )+0
			if prediction0 > prediction1:
				epsilon_pos += pair_weight[pair]
			elif prediction0 == prediction1:
				epsilon0 += pair_weight[pair]
			else:
				epsilon_neg += pair_weight[pair]

		Z = epsilon_neg+epsilon0 - epsilon_pos
		#math.sqrt((epsilon_neg+0.5*epsilon0)/(epsilon_pos+0.5*epsilon0))
		return Z
		
	def predict(self, X):
		"""
		X is a hashtable
		"""
		predictions = {}
		for inst_index in X.keys():
			predictions.update({inst_index: (X[inst_index][self.feature_index] >self.theta) +0 } )

		return predictions

		



		