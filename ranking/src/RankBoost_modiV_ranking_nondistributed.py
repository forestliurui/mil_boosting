"""
This is the nondistributed version of RankBoost modiV class.

This is for general ranking problem

It basically follows the algorithm for crankboost, which use alpha = 0.5*log((e_p + 0.5*e_0)/(e_n + 0.5*e_0)) and also a unique update function for distribution weight.

The only change is that we will remove all the identical and opposite ranker on the training dataset. And we don't allow the algorithm to repick the same ranker in later boosting rounds, i.e. we remove a ranker from hypothesis space once it's being picked. 
And we use argmax |epsilon_pos - epsilon_neg| as the criteria for selecting weak ranker (but still need to check)
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

class RankBoost_modiV_ranking(RankBoost_base_ranking):
	def __init__(self, **parameters):

            self.Z = []
            super(RankBoost_modiV_ranking, self).__init__(**parameters)

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

                type = "continuous"
                StumpRanker.instantiateAll(type, X)
                StumpRanker.prune(X)

		for index_Boosting in range(max_iter_boosting):

			self.weights_pair.append(dict(weights_pair))
                        if self.weak_classifier_name != 'stump_ranker':

			    instance_classifier=WEAK_CLASSIFIERS[self.weak_classifier_name](**self.parameters)
                        else:
                            instance_classifier = StumpRanker.create('continuous')		

			#import pdb;pdb.set_trace()
                        try:
			   instance_classifier.fit(X, y, weights_pair, True)
			
                           StumpRanker.pruneSingleRanker(instance_classifier) #prune the selected weak ranker from the hypothesis space
                        except ValueError as e:
                           if e.message == 'StumpRanker.ValidWeakRankers contains NO weak ranker!':
                                break
                           else:
                                raise e

                        self.weak_classifiers.append(copy.deepcopy(instance_classifier))
			predictions = instance_classifier.predict(X) #predictions is a hashtable -- dictionary
                       
			epsilon0, epsilon_pos, epsilon_neg = self.compute_epsilon( predictions, y, weights_pair)

			self.epsilon["positive"].append( epsilon_pos )
			self.epsilon["negative"].append( epsilon_neg )
			self.epsilon["zero"].append( epsilon0 )

			self.predictions_list_train.append(predictions)

			#epsilon_pair_pos_temp, epsilon_pair_neg_temp = self.getEpsilonPair(predictions, instance_labels_generated_from_bag_labels, weights_inst)
			#self.epsilon_pair["positive"].append(epsilon_pair_pos_temp)
			#self.epsilon_pair["negative"].append(epsilon_pair_neg_temp)

                        if self.epsilon["negative"][-1] == 0 and self.epsilon["zero"][-1] == 0:
                                self.alphas.append(20000)
                                break
                        else:
                                self.alphas.append(0.5*np.log(  (self.epsilon["positive"][-1]+0.5*self.epsilon["zero"][-1])/(self.epsilon["negative"][-1]+0.5*self.epsilon["zero"][-1])  ))
                        

                        Z_cur = (self.epsilon["positive"][-1]+0.5*self.epsilon["zero"][-1])*sqrt((self.epsilon["negative"][-1]+0.5*self.epsilon["zero"][-1])/(self.epsilon["positive"][-1]+0.5*self.epsilon["zero"][-1]))+(self.epsilon["negative"][-1]+0.5*self.epsilon["zero"][-1])*sqrt((self.epsilon["positive"][-1]+0.5*self.epsilon["zero"][-1])/(self.epsilon["negative"][-1]+0.5*self.epsilon["zero"][-1]))
                        self.Z.append(Z_cur)

                        for pair in y:
                                m = predictions[pair[0]]-predictions[pair[1]]
                                weights_pair[pair] = weights_pair[pair]*( ( np.exp(self.alphas[-1]*(1-m^2))+np.exp(-self.alphas[-1]*(1-m^2)) )/2 )*np.exp(-self.alphas[-1]*m)/self.Z[-1]
                        #import pdb;pdb.set_trace()

		self.actual_rounds_of_boosting = len(self.alphas)

class TestRankBoost_ModiV(unittest.TestCase):
    def no_test1(self):
        """
        use some random data to test the syntactic error
        """
        X = {0: np.array([ 1,0, 0 ]), 1: np.array([0, 1, 0]), 2: np.array([0,0,1]), 3: np.array([1,0,1])}
        y = [(0,1), (2,3),(3,1)]
  
        print(X)
        print(y)
       
        ranker = RankBoost_modiV_ranking()
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
       
        ranker = RankBoost_modiV_ranking()
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
       
        ranker = RankBoost_modiV_ranking()
        ranker.fit(X, y) 
        print(ranker.predict_train())
        print(ranker.predict(X))
        import pdb;pdb.set_trace()


if __name__ == "__main__":
    unittest.main()

