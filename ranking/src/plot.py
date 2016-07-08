import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import dill
import os

def draw_plot(results):


	ori_output_name = "modiII_results.pdf"
	num_dataset = len(results)
	dataset_names = results.keys()
	
	stat_names = ["test_error", "train_error"]
	#stat_index =0
	index_dataset = -1
	for dataset in dataset_names:
		output_name = dataset+ori_output_name
		index_dataset += 1
		plt.figure()
		for stat_index in range(len(stat_names)):

			plt.subplot(2, 1, stat_index+1)
			plt.plot([dataset][stat_names[stat_index]] )
			plt.ylabel(stat_names[stat_index])
			plt.xlabel("Boosting Iterations")

		plt.suptitle(dataset)
		plt.savefig("ranking/"+output_name)

def generateRank(results):
	stat_names = ["test_error", "train_error"]

def draw_plot_multiple(results):
	colors = {"rankboost":"r", "rankboost_modiIV":"b", "rankboost_modiIII": 'k'}

	num_method = len(results)
	method_names = results.keys()

	ori_output_name = "combined_results.pdf"
	num_dataset = len(results)
	
	dataset_names= []
	for method in method_names:
		if len(dataset_names) == 0:
			dataset_names = set(results[method].keys())
		else:
			dataset_names = dataset_names.intersection( results[method].keys() )
	#dataset_names = set(dataset_names)
	import pdb;pdb.set_trace()
	stat_names = ["test_error", "train_error"]
	#stat_index =0


	index_dataset = -1
	for dataset in dataset_names:
		output_name = dataset+ori_output_name
		index_dataset += 1
		plt.figure()
		subplot_handle = {}
		for stat_index in range(len(stat_names)):
			plt.subplot(2, 1, stat_index+1)
			plt.ylabel(stat_names[stat_index])
			plt.xlabel("Boosting Iterations")
			
			for method_index in range(num_method):
				method = method_names[method_index]
				
				if dataset in results[method]:				
					subplot_handle[method], =plt.plot(results[method][dataset][stat_names[stat_index]], colors[method]+'.-' )

		plt.suptitle(dataset)
		plt.figlegend([subplot_handle[x] for x in method_names if x in subplot_handle.keys()], method_names, loc = 'upper right')
		plt.savefig("ranking/"+output_name)


if __name__ == "__main__":
	results= {}
	#results_modiII = dill.load(open("ranking/results/results_modiII.pkl","r"))
	#results["rankboost_modiII"] = results_modiII

	pkl_file = open('ranking/movieLen.pkl', 'r')

	movieLen = dill.load(pkl_file)
	"""
	for index in range(len(movieLen.y_train.keys())):
		count = 3
		#user is 827
		user = movieLen.y_train.keys()[index]
		files_modiIII = "ranking/result/results_modiIII_user_"+user+".pkl"
		files_modiIV = "ranking/result/results_modiIV_user_"+user+".pkl"
		files_rankboost = "ranking/result/results_rankboost_user_"+user+".pkl"

		
		if not( os.path.isfile(files_modiIII) and os.path.isfile(files_modiIV) and os.path.isfile(files_rankboost)):	
			results_modiIII = dill.load(open(files_modiIII_pre,"r"))
			results["rankboost_modiIII"] = results_modiIII

			results_modiIV = dill.load(open(files_modiIV_pre,"r"))
			results["rankboost_modiIV"] = results_modiIV
	
			results_rankboost = dill.load(open(files_rankboost_pre,"r"))
			results["rankboost"] = results_rankboost
		else:

			files_modiIII_pre = files_modiIII
			files_modiIV_pre = files_modiIV
			files_rankboost_pre = files_rankboost
	#
	user = "827"
	files_modiIII = "ranking/result/results_modiIII_user_"+user+".pkl"
	files_modiIV = "ranking/result/results_modiIV_user_"+user+".pkl"
	files_rankboost = "ranking/result/results_rankboost_user_"+user+".pkl"

	results_modiIII = dill.load(open(files_modiIII,"r"))
	results["rankboost_modiIII"] = results_modiIII

	results_modiIV = dill.load(open(files_modiIV,"r"))
	results["rankboost_modiIV"] = results_modiIV
	
	results_rankboost = dill.load(open(files_rankboost,"r"))
	results["rankboost"] = results_rankboost
	"""

	results["rankboost_modiIII"] = {}
	results["rankboost"] = {}
	for index in range(len(movieLen.y_train.keys())):
		

		user = movieLen.y_train.keys()[index]

		print "index: ", index, " user: ", user
		files_modiIII = "ranking/result/results_modiIII_user_"+user+".pkl"
		#files_modiIV = "ranking/result/results_modiIV_user_"+user+".pkl"
		files_rankboost = "ranking/result/results_rankboost_user_"+user+".pkl"

		if os.path.isfile(files_modiIII) and os.path.isfile(files_rankboost):
			results_modiIII = dill.load(open(files_modiIII,"r"))
			results["rankboost_modiIII"][user] = results_modiIII[user]
	
			results_rankboost = dill.load(open(files_rankboost,"r"))
			results["rankboost"][user] = results_rankboost[user]


	draw_plot_multiple(results)