"""
Implements the actual client function to run the experiment without inner loop Cross Validation
"""
import os
import numpy as np
import time
import string
import dill
from RankBoost_ranking_nondistributed import RankBoost_ranking

BAG_PREDICTIONS = False
INSTANCE_PREDICTIONS = False
INSTANCE_PREDICTIONS_SIL_STAT = True
BEST_BALANCED_ACCURACY = True

ERROR_BOUND = False

CLASSIFIERS = {
    'rankboost': RankBoost_ranking,
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

def getDataset(user_id, fold_index):
	
	movieLen = dill.load(open('ranking/movieLen/movieLen_user'+str(user_id)+'.pkl'))
	X_train = movieLen.X_train[fold_index]
	p_train = movieLen.p_train[fold_index]

	X_test = movieLen.X_test[fold_index]
	p_test = movieLen.p_test[fold_index]

	return RankingDataSet(X_train, p_train), RankingDataSet(X_test, p_test)

def client_target_test(task, callback):
    """
    used to test client/server
    """
    (user_id, fold_index) = task['key']



    print 'Starting task ..'
    print 'User id:     %d' % user_id
    print 'fold index:  %d' % fold_index

    time.sleep(5)

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
    submission['statistics_boosting']['accuracy']= user_id+fold_index
    submission_boosting = {}
    submission_boosting[0] = submission
    submission_boosting[1] = submission	
    print 'Finished task.'
    return submission_boosting




def client_target(task, callback):
    (user_id, fold_index) = task['key']

    

    print 'Starting task ..'
    print 'User id:     %d' % user_id
    print 'fold index:  %d' % fold_index
    

    train, test = getDataset(user_id, fold_index)

    
    timer = Timer()
   
    parameters = {"max_iter_boosting":500, 'weak_classifier': 'stump_ranker'}
    classifier_name = 'rankboost'

    if classifier_name in CLASSIFIERS:
        ranker = CLASSIFIERS[classifier_name](**parameters)
    else:
        print 'Technique "%s" not supported' % classifier_name
        callback.quit = True
        return

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
    if j == 1:
	predictions = ranker.predict(test.instances, iter = j)
    else:
	predictions = ranker.predict( iter = j)
    error = ranker.getRankingError(predictions, test.critical_pairs)
    submission['statistics_boosting']["test_error"] = error 

    print 'training_error: %f' % submission['statistics_boosting']["train_error"] 
    print 'testing_error:  %f' % submission['statistics_boosting']["test_error"]


    return submission
    #import pdb;pdb.set_trace()
