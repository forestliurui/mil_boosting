"""
This is the nondistributed version of RankBoost base class.

This is for general ranking problem
"""

from math import sqrt, exp

import string
#import data
import numpy as np
import copy

from weak_ranker import WeakRanker
from stump_ranker import StumpRanker

WEAK_CLASSIFIERS = {
        'weak_ranker': WeakRanker,
	'stump_ranker': StumpRanker,
}

class RankBoost_base_ranking(object):
	def __init__(self, **parameters):

		self.max_iter_boosting = parameters.pop("max_iter_boosting", 10)
		self.weak_classifier_name = parameters.pop('weak_classifier', 'stump_ranker') 
		if self.weak_classifier_name == 'dtree_stump':
			parameters['max_depth'] = 1
		parameters.pop('normalization', 0)
		self.parameters = parameters
		self.weak_classifiers = []
		self.epsilon = {}
		self.epsilon["positive"] = []
		self.epsilon["negative"] = []
		self.epsilon["zero"] = []
		self.alphas = []
		self.weights_pair=[]

		self.predictions_list_train = []
		self.X_test = None

		self.X_train = None
		self.y_train = None
		
		self.instance_labels_generated_from_bag_labels = None

		self.epsilon_pair = {}
		self.epsilon_pair["positive"] = []
		self.epsilon_pair["negative"] = []

		self.epsilon_pair_fast = {}
		self.epsilon_pair_fast["positive"] = []
		self.epsilon_pair_fast["negative"] = []
		self.epsilon_pair_fast["zero"] = []

                self.E1_bound = E1()
                self.E2_bound = E2()

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

		self.c = [] #the list of weights for weak classifiers

		#import pdb;pdb.set_trace()

		#initial critical pair weights, which is a hashtable
		weights_pair = {}
	
		for pair in y:
			weights_pair[pair] = float(1)/num_critical_pairs

		for index_Boosting in range(max_iter_boosting):

			self.weights_pair.append(dict(weights_pair))
			instance_classifier = WEAK_CLASSIFIERS[self.weak_classifier_name](**self.parameters)
		
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
			elif abs(self.epsilon["positive"][-1] - (self.epsilon["negative"][-1]+self.epsilon["zero"][-1]) )<0.000001:
				self.alphas.append(0.00001)
				break
			else:
				self.alphas.append(0.5*np.log(  (self.epsilon["positive"][-1])/(self.epsilon["negative"][-1]+self.epsilon["zero"][-1])  ))
			self.Z = []
			
			Z_cur = 2*sqrt(self.epsilon["positive"][-1]*(self.epsilon["zero"][-1]+self.epsilon["negative"][-1]))
			self.Z.append(Z_cur)

			for pair in y:
				m = (predictions[pair[0]]-predictions[pair[1]] )**2 + predictions[pair[0]]-predictions[pair[1]] - 1
				weights_pair[pair] = weights_pair[pair]*np.exp(-self.alphas[-1]*m)/self.Z[-1]

		self.actual_rounds_of_boosting = len(self.alphas)

	def compute_epsilon(self, predictions, y, weights_pair):
		epsilon0 = 0
		epsilon_pos = 0
		epsilon_neg = 0	

                bound = 10**(-5)
		for pair in y:
			if abs(predictions[pair[0]] - predictions[pair[1]]) <= bound:
				epsilon0 += weights_pair[pair]
			elif predictions[pair[0]] - predictions[pair[1]] > bound:
				epsilon_pos += weights_pair[pair]
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
		predictions_accum = self.predictions_list_train[iter][0] - threshold

		predictions_accum_with_true_positive_label = predictions_accum[self.instance_labels_generated_from_bag_labels == 1]
		predictions_accum_with_true_negative_label = predictions_accum[self.instance_labels_generated_from_bag_labels != 1]

		num_positive = predictions_accum_with_true_positive_label.shape[0]
		num_negative = predictions_accum_with_true_negative_label.shape[0]

		
		
		matrix_predictions_pos = np.matrix(np.ones((num_negative, 1)))* np.matrix(predictions_accum_with_true_positive_label)
		matrix_predictions_neg =  np.matrix(predictions_accum_with_true_negative_label.reshape((-1, 1)))*np.matrix(np.ones((num_positive)))

		ranking_error = np.sum(matrix_predictions_pos <= matrix_predictions_neg)
		"""
		for i in range(num_positive):
			for j in range(num_negative):
				if predictions_accum_with_true_positive_label[i] <= predictions_accum_with_true_negative_label[j]:
					ranking_error += 1
		"""
		ranking_error = ranking_error/float(num_positive*num_negative)
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

        def getE_vanilla_exp(self, iter = None):
                """
                return the E for vanilla rankboost, which is defined as E = (1/m)*( \sum_{i=1}^{m} exp(-y_i (g(x'_i) - g(x_i))) ),
                   where m is the number of examples, and g(x) is the predicted value for the example x, and y_i is the label for i-th pair
                """

                predictions_train = self.predict_train(iter)
                res = []

                for pair in self.y_train:
                    res.append( np.exp( -( predictions_train[pair[0]] - predictions_train[pair[1]] ) ) )
                
                return np.mean(res)

        def getE_vanilla(self, iter = None):
                """
                return E, based on the formula, (1/m)*\sum_{i=1}^m \prod_{s=1}^t scale_s(i)
                where scale_s(i) is defined as:
                   (1) for crankboost:
                       scale_s(i) = e**(-alpha_s) if pair i is correctly ranked
                                  = e**(alpha_s)  if pair i is reverse ranked
                                  = cosh(alpha_s) if pair i is tied
                   (2) for rankboost vanilla:
                       scale_s(i) = e**(-alpha_s) if pair i is correctly ranked
                                  = e**(alpha_s)  if pair i is reverse ranked
                                  = 1             if pair i is tied
                Note: iter starts from 1
                """

                self.E1_bound.setAll(rankers = self.weak_classifiers, alphas = self.alphas, epsilons = self.epsilon)

                if iter is None or iter > len(self.alphas):
                      iter = len(self.alphas)
                res_pair = {}

                for iteration in range(iter):
                     predictions = self.predict_train_weak_ranker(iteration)
                     for pair in self.y_train:
                         if pair not in res_pair:
                             res_pair[pair] = 1
                         ordering = self.comparePairPredictions( predictions[pair[0]], predictions[pair[1]] )
                         res_pair[pair] *= self.E1_bound.getScale(iteration, ordering = ordering)
                return np.mean(res_pair.values())

        def getE_Z_vanilla(self, iter = None):
                """
                return E, which is the upper bound for 0-1 loss, based on the assumption that E( \sum_{s=1}^{t} alpha_s*h_s ) = \prod_{s=1}^{t} Z_s
                Note that for vanilla rankboost, E is actually exponential function. However, E might take another form for other variants of rankboost algorithms.
                Note that iter here means the number of boosting rounds/iterations that we checked.
                """
                self.E1_bound.setAll(rankers = self.weak_classifiers, alphas = self.alphas, epsilons = self.epsilon)
                if iter == None or iter > len(self.alphas):
                     iter = len(self.alphas)

                Z = self.E1_bound.getZ()
                E = reduce(lambda x, y: x*y, Z[:iter])
                return E

	def getE_Z(self, iter = None):
                """
                return E, which is the upper bound for 0-1 loss, based on the assumption that E( \sum_{s=1}^{t} alpha_s*h_s ) = \prod_{s=1}^{t} Z_s
                Note that for vanilla rankboost, E is actually exponential function. However, E might take another form for other variants of rankboost algorithms.
                Note that iter here means the number of boosting rounds/iterations that we checked.
                """
                self.E2_bound.setAll(rankers = self.weak_classifiers, alphas = self.alphas, epsilons = self.epsilon)
                if iter == None or iter > len(self.alphas):
                     iter = len(self.alphas)

                Z = self.E2_bound.getZ()
                E = reduce(lambda x, y: x*y, Z[:iter])
                return E

        def getE(self, iter = None):
                """
                return E, based on the formula, (1/m)*\sum_{i=1}^m \prod_{s=1}^t scale_s(i)
                where scale_s(i) is defined as:
                   (1) for crankboost:
                       scale_s(i) = e**(-alpha_s) if pair i is correctly ranked
                                  = e**(alpha_s)  if pair i is reverse ranked
                                  = cosh(alpha_s) if pair i is tied
                   (2) for rankboost vanilla:
                       scale_s(i) = e**(-alpha_s) if pair i is correctly ranked
                                  = e**(alpha_s)  if pair i is reverse ranked
                                  = 1             if pair i is tied
                Note: iter starts from 1
                """
                self.E2_bound.setAll(rankers = self.weak_classifiers, alphas = self.alphas, epsilons = self.epsilon)
                if iter is None or iter > len(self.alphas):
                      iter = len(self.alphas)
                res_pair = {}
                
                for iteration in range(iter):
                     predictions = self.predict_train_weak_ranker(iteration)
                     for pair in self.y_train:
                         if pair not in res_pair:
                             res_pair[pair] = 1
                         ordering = self.comparePairPredictions( predictions[pair[0]], predictions[pair[1]] )
                         res_pair[pair] *= self.E2_bound.getScale(iteration, ordering = ordering)
                return np.mean(res_pair.values())
        
        def comparePairPredictions(self, prediction1, prediction2  ):
                """
                return 1  if prediction1 is considered greater than prediction2
                return -1 if reverse
                return 0  if prediction1 is equal to prediction2
                """     

                bound = 10**(-8)

                if abs(prediction1 -  prediction2) < bound:
                    return 0
                elif prediction1 > prediction2:
                    return 1
                else:
                    return -1
	 
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
		
        def predict_train_weak_ranker(self, iter = None):
               """
               return a dictionary which is the predicted scores of all training instance for the weake ranker obtained at round iter
               """
               return self.predictions_list_train[iter]
			
	def predict_train(self, iter = None):


		self.c = self.alphas
		threshold = 0.5
		if iter == None or iter > len(self.c):
			iter = len(self.c)
		results = {}
		for inst_ID in self.X_train.keys():
			#results[inst_ID] = np.average( [self.predictions_list_train[index][inst_ID] for index in range(iter) ] , weights = self.c[0:iter]   )*np.sign(sum(self.c[0:iter]))
                        results[inst_ID] = np.average( [self.predictions_list_train[index][inst_ID] for index in range(iter) ] , weights = self.c[0:iter]   )*sum(self.c[0:iter])
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
				#results[inst_ID] = np.average( [self.predictions_list_test[index][inst_ID] for index in range(iter) ] , weights = self.c[0:iter]   )*np.sign(sum(self.c[0:iter]))
			        results[inst_ID] = np.average( [self.predictions_list_test[index][inst_ID] for index in range(iter) ] , weights = self.c[0:iter]   )*sum(self.c[0:iter])
                        return results
		else:
			results = {}
			for inst_ID in self.X_test.keys():
				#results[inst_ID] = np.average( [self.predictions_list_test[index][inst_ID] for index in range(iter) ] , weights = self.c[0:iter]   )*np.sign(sum(self.c[0:iter]))
			        results[inst_ID] = np.average( [self.predictions_list_test[index][inst_ID] for index in range(iter) ] , weights = self.c[0:iter]   )*sum(self.c[0:iter])
                        return results
class E_base(object):
     def __init__(self):
         self.rankers = []
         self.epsilons = []
         self.alphas = []
         self.Z = []
     def setAll(self, rankers = None, epsilons = None, alphas = None):
         if rankers is not None:
            self.rankers = rankers
         if epsilons is not None:
            self.epsilons = epsilons
         if alphas is not None:
            self.alphas = alphas        

     def setRankers(self, rankers):
         self.rankers = rankers

     def setEpsilons(self, epsilons):
         #epsilons is a list of dict, each element is a dict, with keys being 'positive', 'negative' and 'zero'
         self.epsilons = epsilons
     def setAlphas(self, alphas):
         self.alphas = alphas

     def getZ(self):
          self.update()
          #E = reduce(lambda x, y: x*y, self.Z[:iter])
          return self.Z
     def update(self):
         raise NotImplementedError("member function update is NOT implemented in the base class")

     def getScale(self, iteration, ordering = 0):
         raise NotImplementedError("member function getScale is NOT implemented in the base class")

class E1(E_base):
     def __init__(self):
         super(E1, self).__init__()
     def update(self):

         if len(self.rankers) != len(self.alphas):
                raise ValueError("the lengths of rankers, epsilons and alphas are NOT equal!")

         start_index = len(self.Z)
         end_index = len(self.alphas)

         for index in range(start_index, end_index):
             cur_alpha = self.alphas[index]
             z = self.epsilons["positive"][index]*np.exp(-cur_alpha) + self.epsilons["negative"][index]*np.exp(cur_alpha) + self.epsilons["zero"][index]
             self.Z.append(z)

     def getScale(self, iteration, ordering = 0):
            """
            iteration starts from 0
            """
            self.update()
            if ordering == 1:
               return np.exp(-self.alphas[iteration])
            elif ordering == -1:
               return np.exp(self.alphas[iteration])
            else:
               return 1.0

class E2(E_base):
     def __init__(self):
         super(E2, self).__init__()
         
         self.pre_alphas = [] #total alphas for previous identical rankers
         self.alpha_dict = {}

     def update(self):     
         #import pdb;pdb.set_trace() 
         if len(self.rankers) != len(self.alphas):
                raise ValueError("the lengths of rankers, epsilons and alphas are NOT equal!")  
         if len(self.Z) != len(self.pre_alphas):
                raise ValueError("the lengths of Z and pre_alphas are NOT equal!")         

         start_index = len(self.Z)
         end_index = len(self.alphas)

         for index in range(start_index, end_index):
             ranker = self.rankers[index]
             if ranker not in self.alpha_dict:
                pre_alpha = 0
             else:
                pre_alpha = self.alpha_dict[ranker]
             cur_alpha = self.alphas[index]
             new_pre_alpha = pre_alpha + cur_alpha
             self.alpha_dict[ranker] = new_pre_alpha
             self.pre_alphas.append(pre_alpha)
             z = self.epsilons["positive"][index]*np.exp(-cur_alpha) + self.epsilons["negative"][index]*np.exp(cur_alpha) + self.epsilons["zero"][index]*np.cosh(new_pre_alpha)/np.cosh(pre_alpha)
             self.Z.append(z)

     def getScale(self, iteration, ordering = 0):
            """
            iteration starts from 0
            """
            self.update()
            if ordering == 1:
               return np.exp(-self.alphas[iteration])
            elif ordering == -1:
               return np.exp(self.alphas[iteration])
            else:       
               return np.cosh(self.alphas[iteration]+self.pre_alphas[iteration])/np.cosh( self.pre_alphas[iteration] )
        
    

def get_bag_label(instance_predictions, bags):
	num_bag = len(bags)
	p_index= 0
	bag_predictions = []
	for bag_index in range(num_bag):
		n_index =p_index+ bags[bag_index].shape[0]
		
		bag_predictions.append( np.max(instance_predictions[p_index: n_index]) )
		p_index = n_index
	return np.array(bag_predictions)
