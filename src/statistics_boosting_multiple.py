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
import numpy as np

def compute_statistics(configuration_file, results_directory, outputfile):

	
	statistics_name=['train_bag_AUC', 'train_bag_accuracy', 'train_bag_balanced_accuracy', 'train_instance_AUC', 'train_instance_accuracy', 'train_instance_balanced_accuracy', 'test_bag_AUC', 'test_bag_accuracy', 'test_bag_balanced_accuracy', 'test_instance_AUC', 'test_instance_accuracy', 'test_instance_balanced_accuracy']

		
    	with open(configuration_file, 'r') as f:
        	configuration = yaml.load(f)    

    	num_dataset = len(configuration['experiments'])
    	for index_dataset in range(num_dataset):
		   	    
    	     dataset_name=configuration['experiments'][index_dataset]['dataset']
	     #dataset_name='trx'
		
	     line=dataset_name
	     line+= ','
	     line+= (','.join(statistics_name) )
	     line+= '\n'
 	     with open(outputfile, 'a+') as f:
                		f.write(line)


             
		
	     dataset_result_path=results_directory+'/mi_kernels/'+ dataset_name+'.db'
	     conn=sqlite3.connect(dataset_result_path)
		
	     c=conn.cursor()
	     #for row in c.execute('select * from statistic_names'):
	     #print row  #row is of type tuple
	

		
		
	     boosting_rounds_list=[]
	     string_to_be_exe = 'select boosting_rounds from statistics_boosting '
	     for row in c.execute(string_to_be_exe):
		boosting_rounds_list.append(row[0])
	     iter_max_boosting=max(boosting_rounds_list)

	     for boosting_round in range(1,iter_max_boosting+1):
		line=('%d' % boosting_round)
		for statistic_name in statistics_name:


			#import pdb;pdb.set_trace()
			string_to_be_exe = 'select statistic_name_id from statistic_names where statistic_name = "%s" ' % statistic_name

			c.execute(string_to_be_exe)
			stat_id=c.fetchone()[0]


			statistic_value_list=[]
			string_to_be_exe = 'select  statistic_value from statistics_boosting where statistic_name_id = %d and boosting_rounds = %d' % (stat_id, boosting_round)

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
