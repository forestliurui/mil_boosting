"""
This is the nondistributed version of RankBoost modiVI class.

This is for general ranking problem.

This is the algorithm proposed by Prof. Harold, which is based on a hypothesis/ranker space without any identical or opposite rankers on the training dataset.
If, at any iteration, it picks a weak ranker, that has been used before, we should adjust the coeffficient alpha and weight distribution accordingly. 
"""

from math import sqrt, exp
import sys

import string
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

class RankBoost_modiVI_ranking(RankBoost_base_ranking):
      def __init__(self, **parameters):

            self.Z = []
            super(RankBoost_modiVI_ranking, self).__init__(**parameters)
      
      def fit(self, X, y):
            '''
	    X is a hashtable with key being instance ID, value being the one-dimensional array containing its features
	    y is the list which contains critical paris, represented as tuples.
	    '''
            
            self.X_train = X
            self.y_train = y

            max_iter_boosting = self.max_iter_boosting
            num_instances = len(X)
            num_critical_pairs = len(y)

            StumpRanker.instantiateAll("continuous", X)
            StumpRanker.prune(X)

            self.alphas_dict = {} #the dict of total weights for weak rankers so far

            #initial critical pair weights, which is a hashtable
            weights_pair = {}

            for pair in y:
                weights_pair[pair] = float(1)/num_critical_pairs

            for index_Boosting in range(max_iter_boosting):
                self.weights_pair.append(dict(weights_pair))
                if self.weak_classifier_name != 'stump_ranker':
                     instance_classifier = WEAK_CLASSIFIERS[self.weak_classifier_name](**self.parameters)
                else:
                     instance_classifier = StumpRanker.create("continuous")

                instance_classifier.fit(X, y, weights_pair, True, self.alphas_dict)
                self.weak_classifiers.append(copy.deepcopy(instance_classifier))
                predictions = instance_classifier.predict(X)
                
                epsilon0, epsilon_pos, epsilon_neg = self.compute_epsilon( predictions, y, weights_pair)

                self.epsilon["positive"].append( epsilon_pos )
                self.epsilon["negative"].append( epsilon_neg )
                self.epsilon["zero"].append( epsilon0 )
                self.predictions_list_train.append(predictions)
     
                if instance_classifier not in self.alphas_dict:
                     pre_alpha = 0
                else:
                     pre_alpha = self.alphas_dict[instance_classifier]

                numerator = epsilon_pos + epsilon0*( (np.exp(-pre_alpha))/(2*np.cosh(pre_alpha))  )
                denominator = epsilon_neg + epsilon0*( (np.exp(pre_alpha))/(2*np.cosh(pre_alpha))  )
                if numerator == 0:
                      new_alpha = -sys.maxint
                elif denominator == 0:
                      new_alpha = sys.maxint
                else:
                      new_alpha = 0.5*np.log( float(numerator)/denominator )
                self.alphas_dict[instance_classifier] = new_alpha #i.e. the total alpha for any instance so far            
   
                ad_alpha = new_alpha - pre_alpha
                self.alphas.append(ad_alpha)
             
                if numerator == 0 or denominator == 0:
                     break
                
                cur_Z = epsilon_pos*np.exp(-new_alpha) + epsilon_neg*np.exp(new_alpha) + epsilon0*(np.cosh(new_alpha+pre_alpha))/np.cosh(pre_alpha)

                self.Z.append(cur_Z)  

                for pair in y:
                     r0 = 2*predictions[pair[0]] - 1
                     r1 = 2*predictions[pair[1]] - 1

                     first_part = np.exp(-new_alpha*r0) + np.exp(new_alpha*r1)
                     second_part_numerator = (np.exp( -new_alpha*r0 ) - np.exp( new_alpha*r1 ))*(np.exp(-pre_alpha*r0) - np.exp(pre_alpha*r1) )
                     second_part_denominator = np.exp( -pre_alpha*r0 ) + np.exp( pre_alpha*r1 )
 
                     weights_pair[pair] = (weights_pair[pair]/cur_Z)*0.5*(first_part + float(second_part_numerator)/second_part_denominator)

                #import pdb;pdb.set_trace()
      
            self.actual_rounds_of_boosting = len(self.alphas)


class TestRankBoost_ModiVI(unittest.TestCase):
    def no_test1(self):
        """
        use some random data to test the syntactic error
        """
        X = {0: np.array([ 1,0, 0 ]), 1: np.array([0, 1, 0]), 2: np.array([0,0,1]), 3: np.array([1,0,1])}
        y = [(0,1), (2,3),(3,1)]
  
        print(X)
        print(y)
       
        ranker = RankBoost_modiVI_ranking()
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
       
        ranker = RankBoost_modiVI_ranking()
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
       
        ranker = RankBoost_modiVI_ranking()
        ranker.fit(X, y) 
        print(ranker.predict_train())
        print(ranker.predict(X))
        import pdb;pdb.set_trace()


if __name__ == "__main__":
    unittest.main()
