"""
Implements the actual client function to run the experiment
"""
import os
import numpy as np
import time
import string

from data import get_dataset
from mi_svm import MIKernelSVM, MIKernelSVR
from vocabulary import EmbeddedSpaceSVM
from Iterative_SVM import Iterative_SVM
from Iterative_SVM_pn import Iterative_SVM_pn

from sil import SIL
from sil_stump import SIL_Stump
from RankBoost_nondistributed import RankBoost
from MIBoosting_Xu_nondistributed import MIBoosting_Xu
from martiboost_nondistributed import MartiBoost
from martiboost_max_nondistributed import MartiBoost_max
from martiboost_median_nondistributed import MartiBoost_median
from Adaboost_nondistributed import AdaBoost

INSTANCE_PREDICTIONS = True
INSTANCE_PREDICTIONS_SIL = True
BEST_BALANCED_ACCURACY = True

CLASSIFIERS = {
    'rankboost': RankBoost,	
    'miboosting_xu': MIBoosting_Xu,
    'martiboost': MartiBoost,
    'martiboost_max': MartiBoost_max,
    'martiboost_median': MartiBoost_median,
    'adaboost': AdaBoost,	
    'svm': MIKernelSVM,
    'svr': MIKernelSVR,
    'embedded_svm' : EmbeddedSpaceSVM,
    'Iterative_SVM': Iterative_SVM,
    'Iterative_SVM_pn': Iterative_SVM_pn,
    'MIBoosting_Xu': MIBoosting_Xu,
    'SIL': SIL,
    'SIL_Stump': SIL_Stump
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


def client_target(task, callback):
    (experiment_name, experiment_id,
     train_dataset, test_dataset, _, _) = task['key']
    parameters = task['parameters']

    print 'Starting task %s...' % str(experiment_id)
    print 'Training Set: %s' % train_dataset
    print 'Test Set:     %s' % test_dataset
    print 'Parameters:'
    for k, v in parameters.items():
        print '\t%s: %s' % (k, str(v))

    train = get_dataset(train_dataset)
    test = get_dataset(test_dataset)

    
    timer = Timer()


    classifier_name = parameters.pop('classifier')
    if classifier_name in CLASSIFIERS:
        classifier = CLASSIFIERS[classifier_name](**parameters)
    else:
        print 'Technique "%s" not supported' % classifier_name
        callback.quit = True
        return

    print 'Training...'
    timer.start('training')
    if train.regression:
        classifier.fit(train.bags, train.bag_labels)
    else:
        classifier.fit(train.bags, train.pm1_bag_labels)
    timer.stop('training')

    submission_boosting = {}
    for boosting_round in range(1,  classifier.actual_rounds_of_boosting+1 ):  #boosting_round starts from 1
	submission_boosting[boosting_round] = construct_submissions(classifier, train, test, boosting_round, timer)
    print 'Finished task %s.' % str(experiment_id)
    return submission_boosting

def construct_submissions(classifier, train, test, boosting_round, timer):
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

    print 'Computing test bag predictions...'
    timer.start('test_bag_predict')
    bag_predictions = classifier.predict(test.bags, boosting_round)
    timer.stop('test_bag_predict')

    if INSTANCE_PREDICTIONS:
        print 'Computing test instance predictions...'
        timer.start('test_instance_predict')
        instance_predictions = classifier.predict(test.instances_as_bags, boosting_round)
        timer.stop('test_instance_predict')

    print 'Computing train bag predictions...'
    timer.start('train_bag_predict')
    train_bag_labels = classifier.predict(train.bags, boosting_round) # Saves results from training set
    timer.stop('train_bag_predict')

    if INSTANCE_PREDICTIONS:
        print 'Computing train instance predictions...'
        timer.start('train_instance_predict')
        train_instance_labels = classifier.predict(train.instances_as_bags, boosting_round)
        timer.stop('train_instance_predict')

    print 'Constructing submission...'
    # Add statistics
    for attribute in ('linear_obj', 'quadratic_obj'):
        if hasattr(classifier, attribute):
            submission['statistics_boosting'][attribute] = getattr(classifier,
                                                          attribute)
    submission['statistics_boosting'].update(timer.get_all('_time'))
    
    #construct submission for predictions
    for i, y in zip(test.bag_ids, bag_predictions.flat):
        submission['accum']['bag_predictions']['test'][i] = float(y)
    for i, y in zip(train.bag_ids, train_bag_labels.flat):
        submission['accum']['bag_predictions']['train'][i] = float(y)
    if INSTANCE_PREDICTIONS:
        for i, y in zip(test.instance_ids, instance_predictions.flat):
            submission['accum']['instance_predictions']['test'][i] = float(y)
        for i, y in zip(train.instance_ids, train_instance_labels.flat):
            submission['accum']['instance_predictions']['train'][i] = float(y)
    
    # For backwards compatibility with older versions of scikit-learn
    if train.regression:
        from sklearn.metrics import r2_score as score
        scorename = 'R^2'
    else:
        try:
            from sklearn.metrics import roc_auc_score as score
        except:
            from sklearn.metrics import auc_score as score
        scorename = 'AUC'

    bag_weights = None
    instance_weights = None
    bag_weights_test = None
    instance_weights_test = None

    #import pdb;pdb.set_trace()
    try:
        if train.bag_labels.size > 1:
	    train_bag_accuracy = np.average( train.bag_labels== ( train_bag_labels > 0  ) , weights= bag_weights )
            #import pdb;pdb.set_trace()
	    if bag_weights is None:
		train_bag_balanced_accuracy= np.average( [ np.average( train_bag_labels[train.bag_labels]>0,  weights= bag_weights ) ,   np.average( train_bag_labels[train.bag_labels==False]<0 ,  weights= bag_weights) ] )
	    else:
	    	train_bag_balanced_accuracy= np.average( [ np.average( train_bag_labels[train.bag_labels]>0,  weights= bag_weights[train.bag_labels] ) ,   np.average( train_bag_labels[train.bag_labels==False]<0 ,  weights= bag_weights[train.bag_labels==False]) ] )
            print ('Training Bag %s score: %f, accuracy: %f, balanced accuracy: %f'
                   % (scorename, score(train.bag_labels, train_bag_labels,  sample_weight= bag_weights) ,train_bag_accuracy, train_bag_balanced_accuracy ))
	    submission['statistics_boosting']['train_bag_'+scorename] = score(train.bag_labels, train_bag_labels,  sample_weight= bag_weights)
	    submission['statistics_boosting']['train_bag_accuracy']=train_bag_accuracy
	    submission['statistics_boosting']['train_bag_balanced_accuracy']=train_bag_balanced_accuracy

	#import pdb;pdb.set_trace()

        if INSTANCE_PREDICTIONS and train.instance_labels.size > 1:
	    train_instance_accuracy = np.average( train.instance_labels== ( train_instance_labels > 0  ) , weights= instance_weights )
	    if instance_weights == None:
		 train_instance_balanced_accuracy= np.average( [ np.average( train_instance_labels[train.instance_labels]>0,  weights= instance_weights ) ,   np.average( train_instance_labels[train.instance_labels==False]<0 ,  weights= instance_weights) ] )
   	    else:
	    	train_instance_balanced_accuracy= np.average( [ np.average( train_instance_labels[train.instance_labels]>0,  weights= instance_weights[train.instance_labels] ) ,   np.average( train_instance_labels[train.instance_labels==False]<0 ,  weights= instance_weights[train.instance_labels==False]) ] )
            print ('Training Instance %s score: %f, accuracy: %f, balanced accuracy: %f'
                   % (scorename, score(train.instance_labels, train_instance_labels,  sample_weight= instance_weights) ,train_instance_accuracy, train_instance_balanced_accuracy ))
	    submission['statistics_boosting']['train_instance_'+scorename] = score(train.instance_labels, train_instance_labels,  sample_weight= instance_weights)
	    submission['statistics_boosting']['train_instance_accuracy']=train_instance_accuracy
	    submission['statistics_boosting']['train_instance_balanced_accuracy']=train_instance_balanced_accuracy

        if test.bag_labels.size > 1:
	    test_bag_accuracy = np.average( test.bag_labels== ( bag_predictions > 0  ) , weights= bag_weights_test )
	    if bag_weights_test != None:
	    	test_bag_balanced_accuracy= np.average( [ np.average( bag_predictions[test.bag_labels]>0 ,  weights= bag_weights_test[test.bag_labels]) ,   np.average( bag_predictions[test.bag_labels==False]<0 ,  weights= bag_weights_test[test.bag_labels==False]) ]  )
            else:
		test_bag_balanced_accuracy= np.average( [ np.average( bag_predictions[test.bag_labels]>0 ) ,   np.average( bag_predictions[test.bag_labels==False]<0 ) ]  )
   
	    print ('Test Bag %s Score: %f, accuracy: %f, balanced accuracy: %f'
                   % (scorename, score(test.bag_labels, bag_predictions, sample_weight= bag_weights_test), test_bag_accuracy, test_bag_balanced_accuracy ))

	    submission['statistics_boosting']['test_bag_'+scorename] = score(test.bag_labels, bag_predictions, sample_weight= bag_weights_test)
  	    submission['statistics_boosting']['test_bag_accuracy']=test_bag_accuracy
	    submission['statistics_boosting']['test_bag_balanced_accuracy']=test_bag_balanced_accuracy

        if INSTANCE_PREDICTIONS and test.instance_labels.size > 1:
   	    test_instance_accuracy = np.average( test.instance_labels== ( instance_predictions > 0  ) , weights= instance_weights_test )
	    if instance_weights_test != None:
	    	test_instance_balanced_accuracy= np.average( [ np.average( instance_predictions[test.instance_labels]>0 ,  weights= instance_weights_test[test.instance_labels]) ,   np.average( instance_predictions[test.instance_labels==False]<0 ,  weights= instance_weights_test[test.instance_labels==False]) ]  )
            else:
		test_instance_balanced_accuracy= np.average( [ np.average( instance_predictions[test.instance_labels]>0 ) ,   np.average( instance_predictions[test.instance_labels==False]<0 ) ]  )
   
	    print ('Test Instance %s Score: %f, accuracy: %f, balanced accuracy: %f'
                   % (scorename, score(test.instance_labels, instance_predictions, sample_weight= instance_weights_test), test_instance_accuracy, test_instance_balanced_accuracy ))

	    submission['statistics_boosting']['test_instance_'+scorename] = score(test.instance_labels, instance_predictions, sample_weight= instance_weights_test)
  	    submission['statistics_boosting']['test_instance_accuracy']=test_instance_accuracy
	    submission['statistics_boosting']['test_instance_balanced_accuracy']=test_instance_balanced_accuracy
        #import pdb;pdb.set_trace()

	if INSTANCE_PREDICTIONS_SIL and train.instance_labels_SIL.size > 1:
	    train_instance_accuracy = np.average( train.instance_labels_SIL== ( train_instance_labels> 0  )  )
	    #import pdb; pdb.set_trace()
	    train_instance_balanced_accuracy= np.average( [ np.average( train_instance_labels[train.instance_labels_SIL]>0 ) ,   np.average( train_instance_labels[train.instance_labels_SIL==False]<0 ) ]  )
            print ('SIL: Training Inst. %s Score: %f, accuracy: %f, balanced accuracy: %f'
                   	% (scorename, score(train.instance_labels_SIL, train_instance_labels) ,train_instance_accuracy, train_instance_balanced_accuracy ))
            submission['statistics_boosting']['SIL_train_instance_'+scorename] = score(train.instance_labels_SIL, train_instance_labels)
	    submission['statistics_boosting']['SIL_train_instance_accuracy']=train_instance_accuracy
	    submission['statistics_boosting']['SIL_train_instance_balanced_accuracy']=train_instance_balanced_accuracy

        #import pdb;pdb.set_trace()

        if INSTANCE_PREDICTIONS_SIL and test.instance_labels_SIL.size > 1:
	    #import pdb;pdb.set_trace()
   	    test_instance_accuracy = np.average( test.instance_labels_SIL== ( instance_predictions > 0  )  )
	    test_instance_balanced_accuracy= np.average( [ np.average( instance_predictions[test.instance_labels_SIL]>0 ) ,   np.average( instance_predictions[test.instance_labels_SIL==False]<0 ) ]  )
            #import pdb;pdb.set_trace()
            print ('SIL: Test Inst. %s Score: %f, accuracy: %f, balanced accuracy: %f'
                   	% (scorename, score(test.instance_labels_SIL, instance_predictions),test_instance_accuracy, test_instance_balanced_accuracy ))
	    submission['statistics_boosting']['SIL_test_instance_'+scorename] = score(test.instance_labels_SIL, instance_predictions)
	    submission['statistics_boosting']['SIL_test_instance_accuracy']=test_instance_accuracy
	    submission['statistics_boosting']['SIL_test_instance_balanced_accuracy']=test_instance_balanced_accuracy

        if BEST_BALANCED_ACCURACY:
		
		if train.bag_labels.size > 1:
			submission['statistics_boosting']['train_bag_best_threshold_for_balanced_accuracy'], submission['statistics_boosting']['train_bag_best_balanced_accuracy'] = getBestBalancedAccuracy(train_bag_labels, train.bag_labels)
			print ('Train (Best Threshold, Best balanced accuracy) -- bag %f , %f' % (submission['statistics_boosting']['train_bag_best_threshold_for_balanced_accuracy'], submission['statistics_boosting']['train_bag_best_balanced_accuracy']))
		if train.instance_labels.size > 1:
			submission['statistics_boosting']['train_instance_best_threshold_for_balanced_accuracy'], submission['statistics_boosting']['train_instance_best_balanced_accuracy'] = getBestBalancedAccuracy(train_instance_labels, train.instance_labels)
			print ('Train (Best Threshold, Best balanced accuracy) --inst %f , %f' % (submission['statistics_boosting']['train_instance_best_threshold_for_balanced_accuracy'], submission['statistics_boosting']['train_instance_best_balanced_accuracy']))
		if test.bag_labels.size > 1:
			submission['statistics_boosting']['test_bag_best_threshold_for_balanced_accuracy'], submission['statistics_boosting']['test_bag_best_balanced_accuracy'] = getBestBalancedAccuracy(bag_predictions, test.bag_labels)
			threshold_temp = submission['statistics_boosting']['train_bag_best_threshold_for_balanced_accuracy']
			submission['statistics_boosting']['test_bag_best_balanced_accuracy_with_threshold_from_train'] = np.average( [ np.average( bag_predictions[test.bag_labels]>threshold_temp ) ,   np.average( bag_predictions[test.bag_labels==False]<threshold_temp ) ]  )
			print ('Test (Best Threshold, Best balanced accuracy, Best balanced accuracy with thres from train) -- bag %f , %f, %f' % (submission['statistics_boosting']['test_bag_best_threshold_for_balanced_accuracy'], submission['statistics_boosting']['test_bag_best_balanced_accuracy'], submission['statistics_boosting']['test_bag_best_balanced_accuracy_with_threshold_from_train']))
	
		if test.instance_labels.size > 1:
			submission['statistics_boosting']['test_instance_best_threshold_for_balanced_accuracy'], submission['statistics_boosting']['test_instance_best_balanced_accuracy'] = getBestBalancedAccuracy(instance_predictions, test.instance_labels)
			threshold_temp = submission['statistics_boosting']['train_instance_best_threshold_for_balanced_accuracy']
			submission['statistics_boosting']['test_instance_best_balanced_accuracy_with_threshold_from_train'] = np.average( [ np.average( instance_predictions[test.instance_labels]>threshold_temp ) ,   np.average( instance_predictions[test.instance_labels==False]<threshold_temp ) ]  )
			print ('Test (Best Threshold, Best balanced accuracy, Best balanced accuracy with thres from train) --inst %f , %f, %f' % (submission['statistics_boosting']['test_instance_best_threshold_for_balanced_accuracy'], submission['statistics_boosting']['test_instance_best_balanced_accuracy'], submission['statistics_boosting']['test_instance_best_balanced_accuracy_with_threshold_from_train']))



    except Exception as e:
        print "Couldn't compute scores."
        print e
    return submission
    #import pdb;pdb.set_trace()

def getBestBalancedAccuracy(predictions, labels):
	#predictions and labels are one-dimensional array, with the same index corresponding to the same instance/bag
	min_val = min(predictions)
	max_val = max(predictions)
	num_threshold = 100
	delta = (max_val -min_val)/float(num_threshold)
	best_threshold = None
	best_BBA = None
	for threshold_index in range( num_threshold+1):
		threshold = min_val + threshold_index* delta
		temp = np.average( [ np.average( predictions[labels]>threshold ) ,   np.average( predictions[labels==False]<threshold ) ]  )
		if best_BBA is None or best_BBA < temp:
			best_BBA = temp
			best_threshold = threshold
	return best_threshold, best_BBA
