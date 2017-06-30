"""
This is the nondistributed version of RankBoost for bipartite setting, described in Figure 9.2 at the book "Foundation of Machine Learning"
With modification to computation of alpha and z, suggested in the Modification II in my draft. We call this RankBoost+ in our paper/thesis

This is for general ranking problem
"""

from math import sqrt, exp

import string
#import data
import numpy as np
import copy

from weak_ranker import WeakRanker
from stump_ranker import StumpRanker
from RankBoost_base_ranking_nondistributed import RankBoost_base_ranking

WEAK_CLASSIFIERS = {
        'weak_ranker': WeakRanker,
	'stump_ranker': StumpRanker,
}

class RankBoost_modiII_ranking(RankBoost_base_ranking):
	def __init__(self, **parameters):

                super(RankBoost_modiII_ranking, self).__init__(**parameters)
                self.Z = []

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
			#self.epsilon_pair["positive"].append(epsilon_pair_pos_temp)
			#self.epsilon_pair["negative"].append(epsilon_pair_neg_temp)

			if self.epsilon["negative"][-1] == 0 and self.epsilon["zero"][-1] == 0:
				self.alphas.append(20000)
				break
			else:
				self.alphas.append(0.5*np.log(  (self.epsilon["positive"][-1]+0.5*self.epsilon["zero"][-1])/(self.epsilon["negative"][-1]+0.5*self.epsilon["zero"][-1])  ))
			
			Z_cur = self.epsilon["positive"][-1]*sqrt((self.epsilon["negative"][-1]+0.5*self.epsilon["zero"][-1])/(self.epsilon["positive"][-1]+0.5*self.epsilon["zero"][-1]))+self.epsilon["negative"][-1]*sqrt((self.epsilon["positive"][-1]+0.5*self.epsilon["zero"][-1])/(self.epsilon["negative"][-1]+0.5*self.epsilon["zero"][-1]))+self.epsilon["zero"][-1]
			self.Z.append(Z_cur)

			for pair in y:
			
				weights_pair[pair] = weights_pair[pair]*np.exp(-self.alphas[-1]*(predictions[pair[0]]-predictions[pair[1]]))/self.Z[-1]

		self.actual_rounds_of_boosting = len(self.alphas)

def get_bag_label(instance_predictions, bags):
	num_bag = len(bags)
	p_index= 0
	bag_predictions = []
	for bag_index in range(num_bag):
		n_index =p_index+ bags[bag_index].shape[0]
		
		bag_predictions.append( np.max(instance_predictions[p_index: n_index]) )
		p_index = n_index
	return np.array(bag_predictions)
