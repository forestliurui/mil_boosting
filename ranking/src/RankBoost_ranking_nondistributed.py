"""
This is the nondistributed version of RankBoost for bipartite setting, described in Figure 9.2 at the book "Foundation of Machine Learning"
With modification to computation of alpha and z, suggested in the Modification II in my draft

This is for general ranking problem
"""

from math import sqrt, exp

import string
#import data
import numpy as np
import copy
import unittest

from weak_ranker import WeakRanker
from stump_ranker import StumpRanker
from RankBoost_base_ranking_nondistributed import RankBoost_base_ranking

WEAK_CLASSIFIERS = {
        'weak_ranker': WeakRanker,
	'stump_ranker': StumpRanker,
}

class RankBoost_ranking(RankBoost_base_ranking):
	def __init__(self, **parameters):
            
            self.Z = []
            super(RankBoost_ranking, self).__init__(**parameters)

	def fit(self, X, y):
		'''
		X is a hashtable with key being instance ID, value being the one-dimensional array containing its features
		y is the list which contains critical paris, represented as tuples.
		'''
		self.y_train = y
		self.X_train = X		

		max_iter_boosting=self.max_iter_boosting
		num_instances = len(X)
		num_critical_pairs = len(y)

		self.c=[] #the list of weights for weak classifiers

		#import pdb;pdb.set_trace()

		#initial critical pair weights, which is a hashtable
		weights_pair= {}
	
		for pair in y:
			weights_pair[pair] = float(1)/num_critical_pairs

		for index_Boosting in range(max_iter_boosting):

			self.weights_pair.append(dict(weights_pair))
			if self.weak_classifier_name != 'stump_ranker':
				instance_classifier=WEAK_CLASSIFIERS[self.weak_classifier_name](**self.parameters)
			else:
				instance_classifier= StumpRanker.create("continuous") #get the continuous version by default 				

			#import pdb;pdb.set_trace()

			instance_classifier.fit(X, y, weights_pair)
			self.weak_classifiers.append(copy.deepcopy(instance_classifier))
			predictions = instance_classifier.predict(X) #predictions is a hashtable -- dictionary

			epsilon0, epsilon_pos, epsilon_neg = self.compute_epsilon( predictions, y, weights_pair)

			self.epsilon["positive"].append( epsilon_pos )
			self.epsilon["negative"].append( epsilon_neg )
			self.epsilon["zero"].append( epsilon0 )

			self.predictions_list_train.append(predictions)

			#epsilon_pair_pos_temp, epsilon_pair_neg_temp = self.getEpsilonPair(predictions, instance_labels_generated_from_bag_labels, weights_inst)
			self.epsilon_pair["positive"].append(epsilon_pos)
			self.epsilon_pair["negative"].append(epsilon_neg)

			if self.epsilon["negative"][-1] == 0:
				self.alphas.append(20000)
				break
			else:
				self.alphas.append(0.5*np.log(  (self.epsilon["positive"][-1])/(self.epsilon["negative"][-1])  ))
			
			Z_cur = self.epsilon["zero"][-1]+2*sqrt(self.epsilon["positive"][-1]*self.epsilon["negative"][-1])
			self.Z.append(Z_cur)

			for pair in y:
			
				weights_pair[pair] = weights_pair[pair]*np.exp(-self.alphas[-1]*(predictions[pair[0]]-predictions[pair[1]]))/self.Z[-1]

		self.actual_rounds_of_boosting = len(self.alphas)
		#import pdb;pdb.set_trace()

	def compute_epsilon(self, predictions, y, weights_pair):
		epsilon0= 0
		epsilon_pos = 0
		epsilon_neg = 0	

		for pair in y:
			if predictions[pair[0]] > predictions[pair[1]]:
				epsilon_pos += weights_pair[pair]
			elif predictions[pair[0]] == predictions[pair[1]]:
				epsilon0 += weights_pair[pair]
			else:
				epsilon_neg += weights_pair[pair]
		return epsilon0, epsilon_pos, epsilon_neg

	def getRankingErrorOneClassifier(self, iter = None):
		"""
		get the training ranking error
		"""
		self.c = self.alphas
		threshold = 0.5

		if iter == None or iter >= len(self.c):
			iter = len(self.c)-1
		
		#import pdb;pdb.set_trace()
		#predictions_accum_matrix = np.matrix(self.c[0:iter])*np.matrix( np.vstack((self.predictions_list_train[0:iter])) )/np.sum(self.c[0:iter])
		predictions = self.predictions_list_train[iter]

		ranking_error = self.getRankingError(predictions, self.y_train)
		return ranking_error

	def getRankingError(self, predictions, y):
		"""
		get the training ranking error
		"""
		

		num_pair = len(y)

		ranking_error = 0

		for pair in y:
			if predictions[pair[0]]	<= predictions[pair[1]]:
				ranking_error += 1
		ranking_error = ranking_error/float(num_pair)
		
		return ranking_error

        def getHalfTiedRankingError(self, predictions, y):
                """
                get the training ranking error which treats tie as half
                """
               	num_pair = len(y)

		ranking_error = 0

		for pair in y:
                        if abs( predictions[pair[0]] - predictions[pair[1]] ) < 10**(-5):
                                ranking_error += 0.5
			elif predictions[pair[0]]  < predictions[pair[1]]:
				ranking_error += 1
		ranking_error = ranking_error/float(num_pair)
		
		return ranking_error

	def getRankingErrorBound(self, iter = None):
		self.c = self.alphas
		threshold = 0.5
		if iter == None or iter > len(self.c):
			iter = len(self.c)

		epsilon_pos = np.array( self.epsilon_pair["positive"][0:iter] )
		epsilon_neg = np.array( self.epsilon_pair["negative"][0:iter] )

		bound = np.exp( -2*( np.sum( ((epsilon_pos - epsilon_neg)/2 )**2 ) )  )
		return bound

		
	
	def getEpsilonPair(self, predictions, labels, weights):
		"""
		return pairwise epsilons defined in Eq. (9.11) from Foundations of Machine Learning
		"""
		num_positive = labels[labels == 1].shape[0]
		num_negative = labels[labels != 1].shape[0]		
			
		predictions_positive = predictions[labels == 1]
		predictions_negative = predictions[labels != 1]

		weights_positive = weights[labels == 1]
		weights_negative = weights[labels != 1]		

		epsilon_pair = {}
		epsilon_pair["positive"] = 0
		epsilon_pair["negative"] = 0

		matrix_predictions_pos = np.matrix(np.ones((num_negative, 1)))* np.matrix(predictions_positive)
		matrix_predictions_neg =  np.matrix(predictions_negative.reshape((-1, 1)))*np.matrix(np.ones((num_positive)))
		matrix_weights = np.matrix(weights_negative.reshape((-1, 1)))*np.matrix(weights_positive)

		epsilon_pair["negative"] = np.sum(matrix_weights[matrix_predictions_pos < matrix_predictions_neg])
		epsilon_pair["positive"] = np.sum(matrix_weights[matrix_predictions_pos > matrix_predictions_neg])

		"""
		for i in range(num_positive):
			for j in range(num_negative):
				weight_pair = weights_positive[i]*weights_negative[j]
				if predictions_positive[i] < predictions_negative[j]: 
					epsilon_pair["negative"] += weight_pair 	
				elif predictions_positive[i] > predictions_negative[j]: 
					epsilon_pair["positive"] += weight_pair 
		"""

		return epsilon_pair["positive"], epsilon_pair["negative"]
					
	def predict_train(self, iter = None):


		self.c = self.alphas
		threshold = 0.5
		if iter == None or iter > len(self.c):
			iter = len(self.c)
		results = {}
		for inst_ID in self.X_train.keys():
			results[inst_ID] = np.average( [self.predictions_list_train[index][inst_ID] for index in range(iter) ] , weights = self.c[0:iter]   )

		return results

	def predict(self, X = None, iter = None):
		"""
		X is assumed to be two dimensional array, each row corresponding to an instance
		"""
		
		self.c = self.alphas
		threshold = 0.5

		if iter == None or iter > len(self.c):
			iter = len(self.c)
		if X is not None:	
			self.X_test = X
			predictions_list = [ instance_classifier.predict(X) for instance_classifier in self.weak_classifiers ]
			self.predictions_list_test = predictions_list
			results = {}
			for inst_ID in X.keys():
				results[inst_ID] = np.average( [self.predictions_list_test[index][inst_ID] for index in range(iter) ] , weights = self.c[0:iter]   )
			return results
		else:
			results = {}
			for inst_ID in self.X_test.keys():
				results[inst_ID] = np.average( [self.predictions_list_test[index][inst_ID] for index in range(iter) ] , weights = self.c[0:iter]   )
			return results


def get_bag_label(instance_predictions, bags):
	num_bag = len(bags)
	p_index= 0
	bag_predictions = []
	for bag_index in range(num_bag):
		n_index =p_index+ bags[bag_index].shape[0]
		
		bag_predictions.append( np.max(instance_predictions[p_index: n_index]) )
		p_index = n_index
	return np.array(bag_predictions)


class TestRankboostRanking(unittest.TestCase):
    def no_test_rankboost(self):
		X = {0: np.array([1, 2]), 1: np.array([1, 1]), 2: np.array([1, 0]), 3: np.array([1, -1]), 4: np.array([1, -2]), 5:np.array([2, 2]), 6: np.array([2, 1]), 7: np.array([2,0]), 8: np.array([2, -1]), 9: np.array([2, -2]), 10: np.array([3, 0]), 11: np.array([0, 0])}
		y = []
		neg_set = [0, 1,2, 3,4,10]
		pos_set = [5, 6, 7, 8, 9,11]
		for i in pos_set:
			for j in neg_set:
				y.append((i,j))
		#param = {'weak_classifier': 'stump_ranker'}
		param = {'weak_classifier': 'weak_ranker'}
		booster = RankBoost_ranking(**param)	
		booster.fit(X, y)
		import pdb;pdb.set_trace()

    def no_test1(self):
        """
        use some random data to test the syntactic error
        """
        X = {0: np.array([ 1,0, 0 ]), 1: np.array([0, 1, 0]), 2: np.array([0,0,1]), 3: np.array([1,0,1])}
        y = [(0,1), (2,3),(3,1)]
  
        print(X)
        print(y)
       
        ranker = RankBoost_ranking()
        ranker.fit(X, y) 
        print(ranker.predict_train())
        print(ranker.predict(X))
        import pdb;pdb.set_trace()
    
    def no_test2(self):
        """
        use some random data to test the syntactic error
        """
        X = {0: np.array([ 1,0, 0 ]), 1: np.array([0, 1, 0]), 2: np.array([0,0,1]), 3: np.array([1,0,1])}
        y = [(0,1), (2, 1),(3,1)]
  
        print(X)
        print(y)
       
        ranker = RankBoost_ranking()
        ranker.fit(X, y) 
        print(ranker.predict_train())
        print(ranker.predict(X))
        import pdb;pdb.set_trace()

    def test3(self):
        """
        use some random data to test the syntactic error
        """
        X = {0: np.array([ 1,0, 0 ]), 1: np.array([0, 1, 0]), 2: np.array([0,0,2]), 3: np.array([1,0,1])}
        y = [(0,1), (2,3),(3,1)]
  
        print(X)
        print(y)
       
        ranker = RankBoost_ranking()
        ranker.fit(X, y) 
        print(ranker.predict_train())
        print(ranker.predict(X))
        import pdb;pdb.set_trace()


if __name__ == '__main__':
	unittest.main()	



