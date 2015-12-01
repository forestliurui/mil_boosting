#!/usr/bin/env python

#usage: ./src/statistics_boosting.py config/nsk.yaml results test instance auc auc_stats.csv
#the first argument is the config file you were using before. 
#second argument is the same results directory
#the third arugment can be train or test, depending on what you want the statsitics for
#the fourth argument is instance or bag
#the fifth argument is acuracy , auc or balanced_accuracy
#Then the final argument is where you save it

import sqlite3
import yaml

def compute_statistics(configuration_file, results_directory, train_or_test, bag_or_instance , statistic, outputfile):

	
	
	if train_or_test not in ('train', 'test'):
        	raise ValueError('Third argument must be "train" or "test"')
	else:
		statistic_name=train_or_test 

	if bag_or_instance.startswith('b'):	
		statistic_name=statistic_name+'_bag'
	elif bag_or_instance.startswith('i'):
		statistic_name=statistic_name+'_instance'
	else:
		raise ValueError('Fourth argument must be "bag" or "instance"')

	import pdb;pdb.set_trace()

	if statistic.startswith('b'):	
		statistic_name=statistic_name+'_balanced_accuracy'
	elif statistic.startswith('au') or statistic.startswith('AU'):
		statistic_name=statistic_name+'_AUC'
	elif statistic.startswith('acc'):
		statistic_name=statistic_name+'_accuracy'
	else:
		raise ValueError('Fifth argument must be "acuracy" , "auc" or "balanced_accuracy"')

	
    	with open(configuration_file, 'r') as f:
        	configuration = yaml.load(f)    

    	num_dataset = len(configuration['experiments'])
    	for index_dataset in range(num_dataset):
	
    		dataset_name=configuration['experiments'][index_dataset]['dataset']
		dataset_name='musk1'
		
		dataset_result_path=results_directory+'/mi_kernels/'+ dataset_name+'.db'
		conn=sqlite3.connect(dataset_result_path)
		
		c=conn.cursor()
		import pdb;pdb.set_trace()

		for row in c.execute('select * from statistics_boosting'):
			print row  #row is of type tuple
			import pdb;pdb.set_trace()
				

	with open(outputfile, 'a+') as f:
                f.write(line)

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog configfile resultsdir train_or_test statistic outputfile")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 6:
        parser.print_help()
        exit()
    compute_statistics(*args, **options)	
