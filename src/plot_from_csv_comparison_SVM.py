"""
This script is used to generate rank file from csv files that contains that statistics for AdaBoost and some SVM-based algorithms.

The format of input is like
method    SVM    AdaBoost
dataset1  0.8     0.7
dataset2  0.9     0.87
 
"""

import csv
import glob
import matplotlib.pyplot as plt
import math
import matplotlib
from scipy.stats.mstats import rankdata
import numpy as np

def readFromCSV(inputfile_name):
	results = {}
	map_index2method = {}
	with open(inputfile_name) as csvfile:
		csvreader = csv.reader(csvfile)
		row_num = 0
		for row in csvreader:
				row_num += 1

				if row_num == 1:
					for col_index in range(1, len(row)):
						map_index2method[col_index] = row[col_index]
				else:					
					dataset_name = row[0]
					results[dataset_name]={}
					for col_index in range(1, len(row)):
						results[dataset_name][map_index2method[col_index]] = [float(row[col_index])]
	return results
				


def generateRank(inputfile_name, outputfile_name):
	statistics_name = ['accuracy']
	
	results = {}
	dataset_names = []
	method_names = []

	ranks = {}
	ranks_average = {}
	for statistic in statistics_name:
		results[statistic] = readFromCSV(inputfile_name)
		dataset_names += results[statistic].keys()
		ranks[statistic] = {}
		for dataset_name in results[statistic].keys():
			method_names+=results[statistic][dataset_name].keys()
			
	dataset_names = set(dataset_names)
	method_names = set(method_names)
	
	#method_names_prechosen = set(["rankboost","adaboost","martiboost","miboosting_xu", "rankboost_modiII", 'rboost', 'auerboost'])
	method_names_prechosen = set(["adaboost","SIL","MI_SVM", "mi_SVM"])

	method_names = method_names.intersection(method_names_prechosen)
	
	boosting_round = 150

	for statistic in statistics_name:
		if statistic not in ranks:
			ranks[statistic] = {}
		if statistic not in ranks_average:
			ranks_average[statistic] = {}
		for dataset_name in results[statistic].keys():
			 
			raw_data_per_stat_dataset = []
			for method_name in method_names:
				
				if boosting_round < len(results[statistic][dataset_name][method_name]):
					raw_data_per_stat_dataset.append(results[statistic][dataset_name][method_name][boosting_round])
				else:
					raw_data_per_stat_dataset.append(results[statistic][dataset_name][method_name][-1])

			raw_rank = rankdata(map(lambda x: -float(x), raw_data_per_stat_dataset))
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
		output_file_name_extended = statistic+"_"+outputfile_name
		
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
			

if __name__ == '__main__':
	from optparse import OptionParser, OptionGroup
    	parser = OptionParser(usage="Usage: %resultsdir  statistic outputfile")
    	options, args = parser.parse_args()
    	options = dict(options.__dict__)
    	if len(args) != 2:
        		parser.print_help()
        		exit()		

	inputfile_name = args[0]
	outputfile_name = args[1]

	generateRank(inputfile_name, outputfile_name)


			
