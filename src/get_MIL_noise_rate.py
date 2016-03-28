"""
compute the fraction of negative instances in positive bag, i.e. noise rate for MIL datasets in given configuration file
"""

import os
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import time
import random
import data
import folds #must import folds, so that view data can be searched in 'folds'

import numpy as np
from math import log, exp

import threading
import matplotlib.pyplot as plt
import string
import gc
from plot_from_csv import get_results

from data import get_dataset


def draw_plot(noise_rate, metrics_results, outputfile_name):
	


	boosting_round = 150

	datasets = set(noise_rate.keys()).intersection(metrics_results.keys())
	result_list = []
	for dataset in datasets:
		if len(metrics_results[dataset]["adaboost"]) > boosting_round:
			metric = metrics_results[dataset]["adaboost"][boosting_round]
		else:
			metric = metrics_results[dataset]["adaboost"][-1]
		result_list.append((noise_rate[dataset], metric))
	result_list = sorted(result_list, key = lambda x: x[0])
	
	plt.figure()
	plt.plot([x[0] for x in result_list], [x[1] for x in result_list], 'r-')
	plt.xlabel("noise_rate")
	plt.ylabel("test_AUC")
	plt.savefig(outputfile_name)

def compute_noise_rate(configuration_file):
	
	noise_rate= {}

    	with open(configuration_file, 'r') as f:
        	configuration = yaml.load(f)  
	outer_folds, inner_folds=configuration['folds']

    	num_dataset = len(configuration['experiments'])
    	for index_dataset in range(num_dataset):
		
    	     	dataset_name=configuration['experiments'][index_dataset]['dataset']

		train_dataset=string.replace( '%s.fold_%4d_of_%4d.train' % (dataset_name,0, outer_folds),' ','0'  )
    	     	test_dataset=string.replace( '%s.fold_%4d_of_%4d.test' % (dataset_name,0, outer_folds),' ','0'   ) 


		train = get_dataset(train_dataset)
    		test = get_dataset(test_dataset)

		train_pos = train.instance_labels[train.instance_labels_SIL == 1]
		test_pos = test.instance_labels[test.instance_labels_SIL == 1]
		train_test_pos = np.hstack((train_pos, test_pos))
		
		noise_rate[dataset_name] = 1- np.average(train_test_pos)
	#import pdb;pdb.set_trace()
	return noise_rate

if __name__ == "__main__":
	from optparse import OptionParser, OptionGroup
    	parser = OptionParser(usage="Usage: %prog configfile resultsdir train_or_test statistic outputfile")
    	options, args = parser.parse_args()
	options = dict(options.__dict__)
	#import pdb;pdb.set_trace()
	noise_rate = compute_noise_rate(args[0])
	statistic_name = "test_instance_AUC"
	metrics_results = get_results(args[1], statistic_name)
	draw_plot(noise_rate, metrics_results, args[2])