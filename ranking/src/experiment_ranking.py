"""
Implements the actual client function to run the experiment without inner loop Cross Validation
"""
import os
import numpy as np
import time
import string
import dill
from RankBoost_ranking_nondistributed import RankBoost_ranking
from RankBoost_modiII_ranking_nondistributed import RankBoost_modiII_ranking
from RankBoost_modiIII_ranking_nondistributed import RankBoost_modiIII_ranking
from RankBoost_modiV_ranking_nondistributed import RankBoost_modiV_ranking
from RankBoost_modiVI_ranking_nondistributed import RankBoost_modiVI_ranking

import unittest
import sys
if os.path.exists("/home/rui/MIL_Boost/MIL_Boosting/MIL_Boost/MIL_Boost/src/"):
	sys.path.append("/home/rui/MIL_Boost/MIL_Boosting/MIL_Boost/MIL_Boost/src/")
elif os.path.exists("/home/rui/MIL_Boosting/src/"):
	sys.path.append("/home/rui/MIL_Boosting/src/")
else:
	sys.path.append("/home/rui/MIL_boosting/src/")

from data import get_dataset

CLASSIFIERS = {
    'rankboost': RankBoost_ranking, #RankBoost in the paper
    'rankboost_modiII': RankBoost_modiII_ranking, #RankBoost+ in the paper
    'rankboost_modiIII': RankBoost_modiIII_ranking,  #CrankBoost in the paper
    'rankboost_modiV': RankBoost_modiV_ranking, #a slight variant of CrankBoost in the paper
    'rankboost_modiVI': RankBoost_modiVI_ranking, #a new rankboost algorithm proposed by Harold, which keep track of total weights for any repicked weak ranker
}

IDX_DIR = os.path.join('box_counting', 'converted_datasets')
PRECOMPUTED_DIR = os.path.join('box_counting', 'precomputed')
IDX_FMT = '%s.idx'
PRECOMPUTED_FMT = '%s_%s.db'

def get_base_dataset(train):
    parts = train.split('.')
    i = 1
    while not parts[i].startswith('fold_'):
        i += 1
    return '.'.join(parts[:i])

class Timer(object):

    def __init__(self):
        self.starts = {}
        self.stops = {}

    def start(self, event):
        self.starts[event] = time.time()

    def stop(self, event):
        self.stops[event] = time.time()

    def get(self, event):
        return self.stops[event] - self.starts[event]

    def get_all(self, suffix=''):
        times = {}
        for event in self.stops.keys():
            times[event + suffix] = self.get(event)
        return times
class RankingDataSet(object):
	def __init__(self, X, p):
		self.instances = X
		self.critical_pairs = p	

def getDataset(dataset_category, user_id, fold_index):
	
	if dataset_category == "MovieLen":
		data_dir = 'ranking/data/MovieLen'
		movieLen = dill.load(open( os.path.join(data_dir,  'movieLen_user'+str(user_id)+'.pkl')  ))
		X_train = movieLen.X_train[fold_index]
		p_train = movieLen.p_train[fold_index]

		X_test = movieLen.X_test[fold_index]
		p_test = movieLen.p_test[fold_index]

		return RankingDataSet(X_train, p_train), RankingDataSet(X_test, p_test)
	elif dataset_category == "LETOR":
		data_dir = 'ranking/data/LETOR'
		movieLen = dill.load(open( os.path.join(data_dir,  'LETOR_query_'+str(user_id)+'.pkl')  ))
		X_train = movieLen.X_train[fold_index]
		p_train = movieLen.p_train[fold_index]

		X_test = movieLen.X_test[fold_index]
		p_test = movieLen.p_test[fold_index]
		#import pdb;pdb.set_trace()
		return RankingDataSet(X_train, p_train), RankingDataSet(X_test, p_test)

	else: #for UCI dataset
		outer_folds = 10
		train_dataset_name=string.replace( '%s.fold_%4d_of_%4d.train' % (user_id, fold_index, outer_folds),' ','0'  )
    		test_dataset_name=string.replace( '%s.fold_%4d_of_%4d.test' % (user_id, fold_index, outer_folds),' ','0'   )		

		train = get_dataset(train_dataset_name)
    		test = get_dataset(test_dataset_name)
		return convert_from_MIDataset_to_RankingDataSet(train),convert_from_MIDataset_to_RankingDataSet(test)

def convert_from_MIDataset_to_RankingDataSet(dataset):
	X = {}
	for inst_index in range(dataset.instances.shape[0]):
		X[inst_index] = dataset.instances[inst_index,:]
	
	p = []
	for i in X.keys():
		for j in X.keys():
			if i != j and  dataset.instance_labels[i] == True and dataset.instance_labels[j] != True:
				p.append((i,j))

	return RankingDataSet(X, p)


def client_target(task, callback):
    (dataset_catetory_no_use, user_id, fold_index) = task['key']

    print 'Starting task ..'
    printCurrentDateTime()
    print 'Ranker name: ',  task['param']['ranker']
    print "Dataset Category: %s" % dataset_catetory_no_use
    print 'User id:     %s' % user_id
    print 'fold index:  %d' % fold_index
	    
    dataset_category = task['param'].pop('dataset_category')	
    train, test = getDataset(dataset_category, user_id, fold_index)
    
    timer = Timer()
   
    parameters = {"max_iter_boosting":100, 'weak_classifier': 'stump_ranker'}
    #parameters = {"max_iter_boosting":200, 'weak_classifier': 'stump_ranker'}

    classifier_name = task['param'].pop('ranker')

    if classifier_name in CLASSIFIERS:
        ranker = CLASSIFIERS[classifier_name](**parameters)
    else:
        print 'Technique "%s" not supported' % classifier_name
        callback.quit = True
        return
    #import pdb;pdb.set_trace()
    print 'Training...'
    timer.start('training')
    
    ranker.fit(train.instances, train.critical_pairs)
    timer.stop('training')

    submission_boosting = {}
    for boosting_round in range(1,  ranker.actual_rounds_of_boosting+1 ):  #boosting_round starts from 1
	submission_boosting[boosting_round] = construct_submissions(ranker, train, test, boosting_round, timer)
    print 'Finished task.'
    return submission_boosting

def construct_submissions(ranker, train, test, boosting_round, timer):
    print ""
    print "computing the submission for boosting round: # %d" % boosting_round
    submission = {
        'accum':{
        	'instance_predictions' : {
            		'train' : {},
            		'test'  : {},
        	},
        	'bag_predictions' : {
            		'train' : {},
            		'test'  : {},
        	},
	},
        'statistics_boosting' : {}
    }
	
    j = boosting_round
    predictions = ranker.predict_train(iter = j)
    error = ranker.getRankingError(predictions, train.critical_pairs)
    submission['statistics_boosting']["train_error"] = error

    error_tied = ranker.getHalfTiedRankingError(predictions, train.critical_pairs)
    submission['statistics_boosting']["train_error_tied"] = error_tied

    E_vanilla_exp = ranker.getE_vanilla_exp(iter = j)
    E_vanilla = ranker.getE_vanilla(iter = j)
    E_modi = ranker.getE(iter = j)
    E_Z = ranker.getE_Z(iter = j)
    E_Z_vanilla = ranker.getE_Z_vanilla(iter = j)
    epsilon_0, epsilon_pos, epsilon_neg = ranker.getEpsilons(iter = j)
    num_unique_rankers = ranker.getNumUniqueRankers(iter = j)

    submission['statistics_boosting']['train_E_vanilla_exp'] = E_vanilla_exp
    submission['statistics_boosting']['train_E_vanilla'] = E_vanilla
    submission['statistics_boosting']['train_E_modi'] = E_modi
    submission['statistics_boosting']['train_E_Z'] = E_Z   
    submission['statistics_boosting']['train_E_Z_vanilla'] = E_Z_vanilla
    submission['statistics_boosting']['train_epsilon_0'] = epsilon_0
    submission['statistics_boosting']['train_epsilon_pos'] = epsilon_pos
    submission['statistics_boosting']['train_epsilon_neg'] = epsilon_neg 
    submission['statistics_boosting']['train_num_unique_rankers'] = num_unique_rankers

    if j == 1:
	predictions = ranker.predict(test.instances, iter = j)
    else:
	predictions = ranker.predict( iter = j)
    error = ranker.getRankingError(predictions, test.critical_pairs)
    submission['statistics_boosting']["test_error"] = error 

    error_tied = ranker.getHalfTiedRankingError(predictions, test.critical_pairs)
    submission['statistics_boosting']["test_error_tied"] = error_tied 

    print 'training_error: %f' % submission['statistics_boosting']["train_error"] 
    print 'testing_error:  %f' % submission['statistics_boosting']["test_error"]
    print 'training_error_tied: %f' % submission['statistics_boosting']["train_error_tied"] 
    print 'testing_error_tied:  %f' % submission['statistics_boosting']["test_error_tied"]

    print 'training_E_vanilla_exp: %f' % submission['statistics_boosting']['train_E_vanilla_exp']
    print 'training_E_vanilla: %f' % submission['statistics_boosting']['train_E_vanilla']
    print 'training_E_Z_vanilla: %f' % submission['statistics_boosting']['train_E_Z_vanilla']
    print 'training_E_modi:    %f' % submission['statistics_boosting']['train_E_modi']
    print 'training_E_Z:       %f' % submission['statistics_boosting']['train_E_Z']

    print 'training_num_unique_rankers: %d' % submission['statistics_boosting']['train_num_unique_rankers']
    print 'training_epsilon_0:   %f' % submission['statistics_boosting']['train_epsilon_0']
    print 'training_epsilon_pos: %f' % submission['statistics_boosting']['train_epsilon_pos']
    print 'training_epsilon_neg: %f' % submission['statistics_boosting']['train_epsilon_neg']

    #import pdb;pdb.set_trace()

    return submission
    #import pdb;pdb.set_trace()

def printCurrentDateTime():
	from datetime import datetime
	currentDT = datetime.now()
	timeString = "%d/%d/%d %d:%d" % (currentDT.year, currentDT.month, currentDT.day, currentDT.hour, currentDT.minute)
 	print timeString

class TestExperiment(unittest.TestCase):
    def test_experiment(self):
         pass

if __name__ == "__main__":
    unittest.main()
