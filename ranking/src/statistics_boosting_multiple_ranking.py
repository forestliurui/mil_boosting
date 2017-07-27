#!/usr/bin/env python

"""
usage: (from the directory containing the 'ranking' directory)
python ranking/src/statistics_boosting_multiple_ranking.py method_name dataset_category database_path  outputfile_name

method_name is the string representing the method name, which appears as part of the name of the database file. E.g. method_name = 'rankboost_modiII'
dataset_category is the string representing the category of datasets that are used in experiments. Currently, it's 'LETOR' or 'MovieLen'
database_path is the string (no trailing '/') representing the directory holding the .db database file
outputfile_name is the string to respresent the basic name of csv file to store the output. It will be expanded to 'ranking/'+method_name+'_'+outputfile_name
	
The format of outputfile is like, for example,

user_0,test_error,train_error
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
user_1,test_error,train_error
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

def compute_statistics(method_name, dataset_category, database_path, outputfile_name):
	"""
	method_name is the string representing the method name, which appears as part of the name of the database file. E.g. method_name = 'rankboost_modiII'
	dataset_category is the string representing the category of datasets that are used in experiments. Currently, it's 'LETOR' or 'MovieLen'
	database_path is the string representing the directory holding the database file (.db) where experimental results are stored
        outputfile_name is the string to respresent the basic name of csv file to store the output. It will be expanded to 'ranking/'+method_name+'_'+outputfile_name
	
	"""
	#method_name = "rankboost"	
	#outputfile_raw = method_name+'.csv'
	outputfile_raw = method_name+'_'+outputfile_name

	#statistics_name = statistics_name + statistics_error
	statistics_name = ['test_error', 'train_error', 'test_error_tied', 'train_error_tied','train_E_vanilla_exp','train_E_vanilla','train_E_Z_vanilla', 'train_E_modi', 'train_E_Z']
       
        statistics_name += ['train_epsilon_pos', 'train_epsilon_neg', 'train_epsilon_0', 'train_num_unique_rankers' ]
 
    	#num_dataset = None
	dataset_map = {}
	
	#dataset_result_path='ranking/movieLen/results/movieLen_'+ method_name+'.db'
	#dataset_result_path = 'ranking/results/'+dataset_category+'/'+dataset_category+'_'+ method_name+ '.db'
	#dataset_result_path = 'ranking/results/'+dataset_category+'/'+'UCI'+'_'+ method_name+ '.db'
        dataset_result_path = database_path+'/'+dataset_category+'_'+method_name+'.db'

	conn=sqlite3.connect(dataset_result_path)
   	c=conn.cursor()
	for row in c.execute('select * from datasets'):
		index_dataset_str, index_fold_str, train_test_str = row[1].split('.')
		if train_test_str == 'test':
			if int(index_dataset_str) not in dataset_map:
				dataset_map[int(index_dataset_str)] = {}	
			dataset_map[int(index_dataset_str)][int(index_fold_str)] = int(row[0])
			
	#num_dataset = len(dataset_map) 
 	#num_dataset = 303
	#fold_index_set = range(5)

    	for index_dataset in dataset_map.keys():
	     
	     if index_dataset>95:
		continue

	     dataset_name = str(index_dataset)
	     
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
	     for index_fold in dataset_map[index_dataset].keys():
	     	string_to_be_exe = 'select boosting_rounds from statistics_boosting where test_set_id= %d' % (dataset_map[index_dataset][index_fold]) 
	     	for row in c.execute(string_to_be_exe):
			boosting_rounds_list.append(row[0])
	     if len(boosting_rounds_list) == 0:
		continue
	     iter_max_boosting=max(boosting_rounds_list)
             lines = ""
	     for boosting_round in range(1,iter_max_boosting+1):
		line=('%d' % boosting_round)
		for statistic_name in statistics_name:

				#import pdb;pdb.set_trace()
				string_to_be_exe = 'select statistic_name_id from statistic_names where statistic_name = "%s" ' % statistic_name

				c.execute(string_to_be_exe)
				stat_id=c.fetchone()[0]

				statistic_value_list=[]
				for index_fold in dataset_map[index_dataset].keys():

					string_to_be_exe = 'select  statistic_value from statistics_boosting where statistic_name_id = %d and boosting_rounds = %d and test_set_id = %d' % (stat_id, boosting_round, dataset_map[index_dataset][index_fold])

					for row in c.execute(string_to_be_exe):
						statistic_value_list.append(row[0])

				line += (',%f' % np.average(statistic_value_list)  )
		line +='\n'
		lines += line				
	     with open(outputfile, 'a+') as f:
                   f.write(lines)


def compute_statistics_for_fold(method_name, dataset_category, database_path, outputfile_name):
	"""
	method_name is the string representing the method name, which appears as part of the name of the database file. E.g. method_name = 'rankboost_modiII'
	dataset_category is the string representing the category of datasets that are used in experiments. Currently, it's 'LETOR' or 'MovieLen'
	database_path is the string representing the directory holding the database file (.db) where experimental results are stored
        outputfile_name is the string to respresent the basic name of csv file to store the output. It will be expanded to 'ranking/'+method_name+'_'+outputfile_name
	
	"""
	#method_name = "rankboost"
	
	#outputfile_raw = method_name+'.csv'
	outputfile_raw = method_name+'_'+outputfile_name


	#statistics_name = statistics_name + statistics_error
	statistics_name = ['test_error', 'train_error', 'test_error_tied', 'train_error_tied','train_E_vanilla_exp','train_E_vanilla','train_E_Z_vanilla', 'train_E_modi', 'train_E_Z']
       
        statistics_name += ['train_epsilon_pos', 'train_epsilon_neg', 'train_epsilon_0', 'train_num_unique_rankers' ]
 
    	#num_dataset = None
	dataset_map = {}
	
	#dataset_result_path='ranking/movieLen/results/movieLen_'+ method_name+'.db'
	#dataset_result_path = 'ranking/results/'+dataset_category+'/'+dataset_category+'_'+ method_name+ '.db'
	#dataset_result_path = 'ranking/results/'+dataset_category+'/'+'UCI'+'_'+ method_name+ '.db'
        dataset_result_path = database_path+'/'+dataset_category+'_'+method_name+'.db'

	conn=sqlite3.connect(dataset_result_path)
   	c=conn.cursor()
	for row in c.execute('select * from datasets'):
		index_dataset_str, index_fold_str, train_test_str = row[1].split('.')
		if train_test_str == 'test':
			if int(index_dataset_str) not in dataset_map:
				dataset_map[int(index_dataset_str)] = {}	
			dataset_map[int(index_dataset_str)][int(index_fold_str)] = int(row[0])
			
	#num_dataset = len(dataset_map) 
 	#num_dataset = 303

	#fold_index_set = range(5)

    	for index_dataset in dataset_map.keys():

           for index_fold in dataset_map[index_dataset].keys(): 
	     if index_dataset>45:
		continue
             boosting_rounds_list = []
             string_to_be_exe = 'select boosting_rounds from statistics_boosting where test_set_id= %d' % (dataset_map[index_dataset][index_fold])
             for row in c.execute(string_to_be_exe):
                        boosting_rounds_list.append(row[0])
             if len(boosting_rounds_list) == 0:
                continue
             iter_max_boosting=max(boosting_rounds_list)

	     dataset_name = str(index_dataset)
	     
	     parameter_set_id_names = []
	     string_to_be_exe = 'select * from parameter_sets'
	     for row in c.execute(string_to_be_exe):
		parameter_set_id_names.append(row)

	     outputfile = 'ranking/'+outputfile_raw
	     #import pdb;pdb.set_trace()
	     line='user_'+dataset_name + "_" + str(index_fold)
	     line+= ','
	     line+= (','.join(statistics_name) )
	     line+= '\n'
 	     with open(outputfile, 'a+') as f:
                		f.write(line)

	     #for row in c.execute('select * from statistic_names'):
	     #print row  #row is of type tuple
	
	     lines = "" 
	     for boosting_round in range(1,iter_max_boosting+1):
		line=('%d' % boosting_round)
		for statistic_name in statistics_name:

				#import pdb;pdb.set_trace()
				string_to_be_exe = 'select statistic_name_id from statistic_names where statistic_name = "%s" ' % statistic_name

				c.execute(string_to_be_exe)
				stat_id=c.fetchone()[0]

				statistic_value_list=[]
				#for index_fold in dataset_map[index_dataset].keys():

			        string_to_be_exe = 'select  statistic_value from statistics_boosting where statistic_name_id = %d and boosting_rounds = %d and test_set_id = %d' % (stat_id, boosting_round, dataset_map[index_dataset][index_fold])

				for row in c.execute(string_to_be_exe):
						statistic_value_list.append(float(row[0]))

				line += (',%f' % np.array(statistic_value_list)  )
		line +='\n'
		lines += line				
	     with open(outputfile, 'a+') as f:
                	f.write(lines)



if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog configfile resultsdir train_or_test statistic outputfile")
    options, args = parser.parse_args()
    #import pdb;pdb.set_trace()
    options = dict(options.__dict__)
   
    compute_statistics(*args, **options)
    #compute_statistics_for_fold(*args, **options) 
