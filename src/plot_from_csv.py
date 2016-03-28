import csv
import glob
import matplotlib.pyplot as plt
import math
import matplotlib
from scipy.stats.mstats import rankdata
import numpy as np

def get_results(directory, statistic_name):
	
	statistics_names=['train_bag_AUC', 'train_bag_accuracy', 'train_bag_balanced_accuracy', 'train_instance_AUC', 'train_instance_accuracy', 'train_instance_balanced_accuracy', 'test_bag_AUC', 'test_bag_accuracy', 'test_bag_balanced_accuracy', 'test_instance_AUC', 'test_instance_accuracy', 'test_instance_balanced_accuracy']
	statistics_names_SIL = ['SIL_train_instance_AUC', 'SIL_train_instance_accuracy', 'SIL_train_instance_balanced_accuracy', 'SIL_test_instance_AUC', 'SIL_test_instance_accuracy', 'SIL_test_instance_balanced_accuracy']
	statistics_names_best = ['train_bag_best_balanced_accuracy', 'train_bag_best_threshold_for_balanced_accuracy', 'train_instance_best_balanced_accuracy', 'train_instance_best_threshold_for_balanced_accuracy', 'test_bag_best_balanced_accuracy', 'test_bag_best_balanced_accuracy_with_threshold_from_train', 'test_instance_best_balanced_accuracy', 'test_instance_best_balanced_accuracy_with_threshold_from_train']
	earlyStop_names = ['ranking_error', 'ranking_error_bound']
	
	statistics_names = statistics_names + statistics_names_best
	
	#for modified rankboost
		
	#statistics_names = statistics_names + earlyStop_names
	#for modified rankboost

	for index in range(len(statistics_names)):
		if statistics_names[index] == statistic_name:
			statistic_index = index

	results = {}
	csv_file_names = glob.glob(directory+'/*.csv')
	
	for csv_file_name in  csv_file_names:
		with open(csv_file_name) as csvfile:
			csvreader = csv.reader(csvfile)
			method_name = csv_file_name.split('.')[0]
			method_name = method_name.split('/')[1]
			for row in csvreader:
				try:
					boosting_round = int(row[0])
					
					if dataset_name not in results:
						results[dataset_name]={}
					
					if method_name not in results[dataset_name]:
						results[dataset_name][method_name]  = []

					results[dataset_name][method_name].append(row[statistic_index + 1])

				except:
					dataset_name = row[0]
	return results

def draw_plot(results, statistic_name,  outputfile_name):
	#colors=['r', 'b', 'k','c', 'y', 'm']
	colors={'rankboost':'r', 'miboosting_xu':'b','adaboost':'k', 'martiboost':'c', 'rankboost_m3':'m','martiboost_max':'y'}
	dataset_names =  results.keys()
	num_dataset = len(dataset_names)
	plt.figure(figsize=(6*num_dataset, 6*num_dataset))

	index_dataset = -1
	for dataset_name in dataset_names:
		index_dataset += 1

		plt.subplot(math.ceil( len(dataset_names )/2 + 1), 3, index_dataset)

		plt.xlabel('Boosting Iterations')
		plt.ylabel(statistic_name)
		color_index = -1
		plt.axis([0, 500, 0.49, 1.1])

		method_names = results[dataset_name].keys()
		
		for method_name in method_names:
			color_index +=1
			plt.plot(results[dataset_name][method_name], colors[method_name]+'.-')

		plt.legend(method_names)
	     	plt.title(dataset_name)
	plt.savefig(outputfile_name)

def generateRank(directory, outputfile_name):
	
	boosting_round = 150

	statistics_name = ['test_instance_AUC', 'test_bag_AUC',  'test_instance_balanced_accuracy', 'test_bag_balanced_accuracy']
	statistics_name_best = ['test_instance_best_balanced_accuracy',  'test_bag_best_balanced_accuracy']
	
	statistics_name += statistics_name_best
	


	results = {}
	dataset_names = []
	method_names = []

	ranks = {}
	ranks_average = {}
	for statistic in statistics_name:
		results[statistic] = get_results(directory, statistic)
		dataset_names += results[statistic].keys()
		ranks[statistic] = {}
		for dataset_name in results[statistic].keys():
			method_names+=results[statistic][dataset_name].keys()
			
	dataset_names = set(dataset_names)
	method_names = set(method_names)
	
	method_names_prechosen = set(["rankboost","adaboost","martiboost","miboosting_xu"])
	method_names = method_names.intersection(method_names_prechosen)
	
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
			

			
 
			

	
	

def draw_plot1(directory, outputfile_name):

	colors={'rankboost':'r', 'miboosting_xu':'b','adaboost':'k', 'martiboost':'c', 'rankboost_pos':'y','rankboost_m3':'m', 'rankboost_m3_pos':'g', 'rankboost_earlyStop': 'r'}


	#statistics_name = ['test_instance_AUC', 'train_instance_AUC', 'test_bag_AUC', 'train_bag_AUC', 'test_instance_balanced_accuracy', 'train_instance_balanced_accuracy', 'test_bag_balanced_accuracy', 'train_bag_balanced_accuracy']
	#statistics_name = ['test_instance_AUC', 'test_bag_AUC',  'test_instance_balanced_accuracy', 'test_bag_balanced_accuracy']
	statistics_name = ['test_instance_AUC', 'test_bag_AUC',  'test_instance_balanced_accuracy', 'test_bag_balanced_accuracy']
	statistics_name_best = ['test_instance_best_balanced_accuracy',  'test_bag_best_balanced_accuracy']
	


	statistics_name += statistics_name_best
	
	# for modified rankboost
	earlyStop_name = ['test_instance_AUC', 'test_bag_AUC', 'train_instance_AUC', 'train_bag_AUC', 'ranking_error', 'ranking_error_bound']	
	statistics_name = earlyStop_name
	# for modified rankboost


	results = {}
	dataset_names = []
	for statistic in statistics_name:
		results[statistic] = get_results(directory, statistic)
		dataset_names += results[statistic].keys()
	dataset_names = set(dataset_names)

	index_dataset = -1
	#matplotlib.rc('legend', fontsize=0.5, linewidth=2)
	#plt.tick_params(labelsize=50)
	for dataset_name in dataset_names:
		
		output_name = dataset_name + outputfile_name
		plt.figure(figsize=(14*len(statistics_name), 10*len(statistics_name)))
		index_dataset += 1
		subplot_handle = {}
		for stat_index in range(len(statistics_name)):
			stat_name = statistics_name[stat_index]
			#plt.subplot(4, math.ceil( len(statistics_name)/3), stat_index+1)
			#plt.subplot(2, 2, stat_index+1)
			plt.subplot(3, 2, stat_index+1)
			plt.yticks(fontsize = 50)
			plt.xticks(fontsize = 50)
			plt.xlabel('Boosting Iterations', fontsize = 40)
			plt.ylabel(stat_name, fontsize = 60)
			color_index = -1
			if stat_name != "ranking_error" and stat_name != "ranking_error_bound":
				plt.axis([0, 500, 0.49, 1.1], fontsize = 50)
			else:
				plt.axis([0, 500, 0, 1.1], fontsize = 50)

			method_names = results[stat_name][dataset_name].keys()
		
			for method_name in method_names:
				color_index +=1
				subplot_handle[method_name], = plt.plot(results[stat_name][dataset_name][method_name], colors[method_name]+'.-')

			#plt.legend(method_names, fontsize = 35)
	     		#plt.title(dataset_name, fontsize = 30)
		plt.suptitle(dataset_name, fontsize = 50)
		#plt.legend(method_names, fontsize = 35)
		plt.figlegend([subplot_handle[x] for x in method_names], method_names, loc = 'upper right',  fontsize = 50)
		plt.savefig(output_name, orientation = 'landscape')

		#break



if __name__ == '__main__':
	from optparse import OptionParser, OptionGroup
    	parser = OptionParser(usage="Usage: %resultsdir  statistic outputfile")
    	options, args = parser.parse_args()
    	options = dict(options.__dict__)
    	if len(args) != 3:
        		parser.print_help()
        		exit()		

	directory = args[0]
	statistic_name = args[1]
	outputfile_name = args[2]

	#draw_plot1(directory, outputfile_name)
	
	generateRank(directory, outputfile_name)

	#results = get_results(directory, statistic_name)
	#draw_plot(results, statistic_name,  outputfile_name)
