#!/usr/bin/env python


import sqlite3
import yaml
import numpy as np

def compute_statistics(method_name):
	#method_name = "rankboost"
	
	outputfile_raw = 'ranking/'+method_name+'.csv'

	#statistics_name = statistics_name + statistics_error
	statistics_name = ['test_error', 'train_error']
  

    	#num_dataset = None
	dataset_map = {}
	
	dataset_result_path='ranking/movieLen/results/database/movieLen_'+ method_name+'.db'
	conn=sqlite3.connect(dataset_result_path)
   	c=conn.cursor()
	for row in c.execute('select * from datasets'):
		if row[1] != 'train':
			dataset_map[int(row[0])] = row[1]
	
	num_dataset = len(dataset_map) 
 
    	for index_dataset in range(2, num_dataset+2):
	     
	     dataset_name = dataset_map[index_dataset]
	     
	     parameter_set_id_names = []
	     string_to_be_exe = 'select * from parameter_sets'
	     for row in c.execute(string_to_be_exe):
		parameter_set_id_names.append(row)

             
	     outputfile = 'ranking/'+outputfile_raw
	     #import pdb;pdb.set_trace()
	     line='user_'+dataset_name
	     line+= ','
	     line+= (','.join(statistics_name) )
	     line+= '\n'
 	     with open(outputfile, 'a+') as f:
                		f.write(line)


             
		

	     #for row in c.execute('select * from statistic_names'):
	     #print row  #row is of type tuple
	

		
		
	     boosting_rounds_list=[]
	     string_to_be_exe = 'select boosting_rounds from statistics_boosting where test_set_id= %d' % (index_dataset) 
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
				string_to_be_exe = 'select  statistic_value from statistics_boosting where statistic_name_id = %d and boosting_rounds = %d and test_set_id = %d' % (stat_id, boosting_round, index_dataset)

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
    #if len(args) != 1:
    #    parser.print_help()
    #    exit()
    #compute_statistics(*args, **options)
    import pdb;pdb.set_trace()	
    compute_statistics(*args)