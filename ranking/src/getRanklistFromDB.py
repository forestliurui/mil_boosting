"""
This is where I tried to reduce the time to extract ranking list of methods in comparative experiments.
Previously, I do it in two seperate steps by using two separate scripts: 
(1) use statistics_boosting_multiple_ranking to get a csv file for each method. This csv file contains statistsics for each and every round of boosting 
   results on all datasets
(2) use plot_from_csv_ranking to read results from csv files just got and compare different methods on a pre-specified round. Write to a csv file as
   output

This script will directly read the statistics on a pre-specified round from DataBase, based on which a csv file is generated which contains their 
   relative ranking.
"""
import unittest
import sqlite3
import glob
import numpy as np
from scipy.stats.mstats import rankdata

def get_results(dir_DB, boosting_round_preset):
	"""
	dir_DB is a string which represents the directory containing the '.db' files. This string should NOT end with '/' .
	boosting_round is a int which indicates the boosting round that we want to compare on

	return 
	a dictionary like results[statistic_name][dataset_name][method_name]
	"""
	statistic_names = ['test_error', 'train_error']
	
	results = {}
	for statistic_name in statistic_names:
		results[statistic_name] = {}
	DB_file_names = glob.glob(dir_DB+'/*.db')
	
	for DB_file_name in  DB_file_names:
		with sqlite3.connect(DB_file_name) as conn:
			c = conn.cursor()
			method_name = DB_file_name.split('.')[0]
			method_name = method_name.split('/')[-1]
			method_name = '_'.join(method_name.split('_')[1:]) #exclude the substring before the first '_', which represents dataset name
			
			dataset_map = {}
			for row in c.execute('select * from datasets'):
				index_dataset_str, index_fold_str, train_test_str = row[1].split('.')
				if train_test_str == 'test':
					if int(index_dataset_str) not in dataset_map:
						dataset_map[int(index_dataset_str)] = {}	
					dataset_map[int(index_dataset_str)][int(index_fold_str)] = int(row[0])
			statistic_id_map = {}
			for statistic_name in statistic_names:

				string_to_be_exe = 'select statistic_name_id from statistic_names where statistic_name = "%s" ' % statistic_name
				c.execute(string_to_be_exe)
				stat_id=c.fetchone()[0]
				statistic_id_map[statistic_name] = stat_id

			for index_dataset in dataset_map.keys():
				#if index_dataset > 5:
				#	break
				#if index_dataset != 96:
				#	continue
				print 'method_name: '+method_name+' ;dataset_index: '+str(index_dataset)
	     
	     			dataset_name = str(index_dataset)
	     			#if dataset_name not in results:
				#	results[dataset_name] = {}
				#results[dataset_name][method_name] = {}

				boosting_rounds_list=[]
	     			for index_fold in dataset_map[index_dataset].keys():
	     				string_to_be_exe = 'SELECT MAX(boosting_rounds) FROM statistics_boosting WHERE test_set_id= %d' % (dataset_map[index_dataset][index_fold]) 
	     				for row in c.execute(string_to_be_exe):
						if row[0] is not None:
							boosting_rounds_list.append(row[0])
	     			#if index_dataset == 5:
				#	import pdb;pdb.set_trace()
				if len(boosting_rounds_list) == 0:
					continue
	     			max_boosting_round=max(boosting_rounds_list)
					
				boosting_round =min(boosting_round_preset, max_boosting_round)
				for statistic_name in statistic_names:

						#string_to_be_exe = 'select statistic_name_id from statistic_names where statistic_name = "%s" ' % statistic_name
						#c.execute(string_to_be_exe)
						#stat_id=c.fetchone()[0]
						statistic_value_list=[]
						for fold_index in dataset_map[index_dataset].keys():
							string_to_be_exe = 'select  statistic_value from statistics_boosting where statistic_name_id = %d and boosting_rounds = %d and test_set_id = %d' % (statistic_id_map[statistic_name], boosting_round, dataset_map[index_dataset][index_fold])

							for row in c.execute(string_to_be_exe):
								statistic_value_list.append(row[0])
						if dataset_name not in results[statistic_name]:
							results[statistic_name][dataset_name] = {}
							
						results[statistic_name][dataset_name][method_name] = [np.average(statistic_value_list)]


	return results


def generateRank(directory, outputfile_name):
	
	boosting_round = 150
        '''
	statistics_name = ['test_instance_AUC', 'test_bag_AUC',  'test_instance_balanced_accuracy', 'test_bag_balanced_accuracy']
	statistics_name_best = ['test_instance_best_balanced_accuracy',  'test_bag_best_balanced_accuracy']
	
	statistics_name += statistics_name_best
	statistics_name += ['train_instance_AUC']
        '''

	statistics_name = ['test_error', 'train_error']

	results = {}
	dataset_names = []
	method_names = []

	ranks = {}
	ranks_average = {}
	results = get_results(directory, boosting_round)
	for statistic in statistics_name:
		#results[statistic] = get_results(directory, statistic)
		dataset_names += results[statistic].keys()
		ranks[statistic] = {}
		for dataset_name in results[statistic].keys():
			method_names+=results[statistic][dataset_name].keys()
			
	dataset_names = set(dataset_names)
	method_names = set(method_names)
	
	method_names_prechosen = set(["rankboost","adaboost","martiboost","miboosting_xu", "rankboost_modiII", "rankboost_modiOp", "rankboost_modiIII"])
	method_names = method_names.intersection(method_names_prechosen)
	
	for statistic in statistics_name:
		if statistic not in ranks:
			ranks[statistic] = {}
		if statistic not in ranks_average:
			ranks_average[statistic] = {}
		for dataset_name in results[statistic].keys():
			if not 	method_names.issubset(set(results[statistic][dataset_name].keys())):
				continue		

			raw_data_per_stat_dataset = []
			for method_name in method_names:
				
				if boosting_round < len(results[statistic][dataset_name][method_name]):
					raw_data_per_stat_dataset.append(results[statistic][dataset_name][method_name][boosting_round])
				else:
					raw_data_per_stat_dataset.append(results[statistic][dataset_name][method_name][-1])

			raw_rank = rankdata(map(lambda x: float(x), raw_data_per_stat_dataset))
			index = 0
			for method_name in method_names:
				if method_name not in ranks[statistic]:
					ranks[statistic][method_name] = []
				ranks[statistic][method_name].append(raw_rank[index])
				
				index += 1

		for method_name in method_names:
			if method_name not in ranks_average[statistic]:
				ranks_average[statistic][method_name] ={}
			ranks_average[statistic][method_name]["rank"] = np.average(ranks[statistic][method_name])
			ranks_average[statistic][method_name]["num_dataset"] = len(ranks[statistic][method_name])

		#import pdb;pdb.set_trace()
		output_file_name_extended = 'ranking/'+statistic+"_"+outputfile_name
		
		for method_name in method_names:
			line = method_name
			line += ","
			line += str(ranks_average[statistic][method_name]["num_dataset"])
			line += ","
			line += str(ranks_average[statistic][method_name]["rank"])
			line += "\n"
			#import pdb;pdb.set_trace()

			with open(output_file_name_extended, 'a+') as f:
				f.write(line)
			
class TestGetResults(unittest.TestCase):
	def test_profile_get_results(self):
		import cProfile
		#import re
		dir_DB = 'ranking/movieLen/results'

		cProfile.run('re.compile("foo|bar")')
			


if __name__ == "__main__":
	dir_DB = 'ranking/movieLen/results'
	outputfile_name = 'ranking_results.csv'
	#boosting_round_preset = 150
	#results = getRanking(dir_DB, boosting_round_preset)
	generateRank(dir_DB, outputfile_name)
	import pdb;pdb.set_trace()	