import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import dill

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

def draw_plot_multiple(results):
	colors = {"rankboost":"r", "rankboost_modiII":"b", "rankboost_modiIII": 'k'}

	num_method = len(results)
	method_names = results.keys()

	ori_output_name = "modiII_results.pdf"
	num_dataset = len(results)
	
	dataset_names= []
	for method in method_names:
		dataset_names += results[method].keys()
	dataset_names = set(dataset_names)
	
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
	results_modiII = dill.load(open("ranking/results_modiII.pkl","r"))
	results["rankboost_modiII"] = results_modiII
	
	results_rankboost = dill.load(open("ranking/results_rankboost.pkl","r"))
	results["rankboost"] = results_rankboost

	results_modiIII = dill.load(open("ranking/results_modiIII.pkl","r"))
	results["rankboost_modiIII"] = results_modiIII

	draw_plot_multiple(results)