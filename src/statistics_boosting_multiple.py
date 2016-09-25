#!/usr/bin/env python
"""
usage: ./src/statistics_boosting_multiple.py configuration_file results_root_directory outputfile_raw

configuration_file is the string to represent path-file for the configuration file which is used when running the experiment
results_root_directory is the string to represent the root directory where results database is stored. 
outputfile_raw the string to respresent the basic name of csv file to store the output. It will be expanded to be parameter_set_name+outputfile_raw

The format of outputfile is like, for example,

dataset1, stat1, stat2
1,0.698972,0.658136
2,0.560522,0.507537
3,0.452127,0.421634
4,0.428380,0.383730
5,0.415185,0.366476
...
...
38,0.434098,0.233784
39,0.429312,0.232903
40,0.434183,0.231904
dataset2, stat1, stat2
1,0.662792,0.634783
2,0.501789,0.468243
3,0.434978,0.396406
4,0.401508,0.357880
5,0.385053,0.335721
...
...

"""
import sqlite3
import yaml
import numpy as np

def compute_statistics(configuration_file, results_directory, outputfile_raw):
	
	statistics_name=['train_bag_AUC', 'train_bag_accuracy', 'train_bag_balanced_accuracy', 'train_instance_AUC', 'train_instance_accuracy', 'train_instance_balanced_accuracy', 'test_bag_AUC', 'test_bag_accuracy', 'test_bag_balanced_accuracy', 'test_instance_AUC', 'test_instance_accuracy', 'test_instance_balanced_accuracy']
	statistics_name_SIL = ['SIL_train_instance_AUC', 'SIL_train_instance_accuracy', 'SIL_train_instance_balanced_accuracy', 'SIL_test_instance_AUC', 'SIL_test_instance_accuracy', 'SIL_test_instance_balanced_accuracy']
	statistics_name_best = ['train_bag_best_balanced_accuracy', 'train_bag_best_threshold_for_balanced_accuracy', 'train_instance_best_balanced_accuracy', 'train_instance_best_threshold_for_balanced_accuracy', 'test_bag_best_balanced_accuracy', 'test_bag_best_balanced_accuracy_with_threshold_from_train', 'test_instance_best_balanced_accuracy', 'test_instance_best_balanced_accuracy_with_threshold_from_train']
	statistics_error = ['ranking_error', 'ranking_error_bound']
	#statistics_name = statistics_name + statistics_error
	statistics_name = statistics_name + statistics_name_best

    	with open(configuration_file, 'r') as f:
        	configuration = yaml.load(f)    

    	num_dataset = len(configuration['experiments'])
    	for index_dataset in range(num_dataset):
		   	    
    	     dataset_name=configuration['experiments'][index_dataset]['dataset']
	     
	     if	dataset_name == 'trx':
		continue
		
	     dataset_result_path=results_directory+'/mi_kernels/'+ dataset_name+'.db'
	     conn=sqlite3.connect(dataset_result_path)
		
	     c=conn.cursor()

	     parameter_set_id_names = []
	     string_to_be_exe = 'select * from parameter_sets'
	     for row in c.execute(string_to_be_exe):
		parameter_set_id_names.append(row)

             for (parameter_set_id, parameter_set_name) in parameter_set_id_names:
		outputfile = parameter_set_name+outputfile_raw
		
	     	line=dataset_name
	     	line+= ','
	     	line+= (','.join(statistics_name) )
	     	line+= '\n'
 	     	with open(outputfile, 'a+') as f:
                	f.write(line)

	     	#for row in c.execute('select * from statistic_names'):
	     	#print row  #row is of type tuple

	     	boosting_rounds_list=[]
	     	string_to_be_exe = 'select boosting_rounds from statistics_boosting where parameter_set_id = %d' % parameter_set_id
	     	for row in c.execute(string_to_be_exe):
			boosting_rounds_list.append(row[0])
		if len(boosting_rounds_list) == 0:
			continue
	     	iter_max_boosting=max(boosting_rounds_list)

	     	for boosting_round in range(1,iter_max_boosting+1):
			line=('%d' % boosting_round)
			for statistic_name in statistics_name:
				#import pdb;pdb.set_trace()
				string_to_be_exe = 'select statistic_name_id from statistic_names where statistic_name = "%s" ' % statistic_name

				c.execute(string_to_be_exe)
				stat_id=c.fetchone()[0]

				statistic_value_list=[]
				string_to_be_exe = 'select  statistic_value from statistics_boosting where statistic_name_id = %d and boosting_rounds = %d and parameter_set_id = %d' % (stat_id, boosting_round, parameter_set_id)

				for row in c.execute(string_to_be_exe):
					statistic_value_list.append(row[0])
			
				line += (',%f' % np.average(statistic_value_list)  )
			line +='\n'
				
			with open(outputfile, 'a+') as f:
                		f.write(line)

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog configfile resultsdir train_or_test statistic outputfile")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 3:
        parser.print_help()
        exit()
    compute_statistics(*args, **options)	
