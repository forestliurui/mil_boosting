#combine results for different methods into one single figure

from statistics_boosting_multiple_plot import compute_statistics
import numpy as np
import matplotlib.pyplot as plt
import math

args={}
#args['rankboost']=["./config/test_boosting_rankboost_m3_no_tune1.yaml", "results_test_boosting_rankboost_m3_no_tune1", "rankboost1.pdf"]
#args['MIBoosting_Xu']=["./config/test_boosting_Xu_no_tune1.yaml", "results_test_boosting_Xu_no_tune1", "MIBoosting_Xu1.pdf"]

args['rankboost']=["./config/test_boosting_rankboost_m3_no_tune.yaml", "results_test_boosting_rankboost_m3_no_tune", "rankboost.pdf"]
args['MIBoosting_Xu']=["./config/test_boosting_Xu_no_tune.yaml", "results_test_boosting_Xu_no_tune", "MIBoosting_Xu.pdf"]

results={}
results['rankboost']=compute_statistics(*args['rankboost'])
results['MIBoosting_Xu']=compute_statistics(*args['MIBoosting_Xu'])

num_dataset=max(( len(results['rankboost'].keys()), len(results['MIBoosting_Xu'].keys()) ))

colors=['r', 'b', 'k','c', 'y', 'm']
plt.figure(figsize=(6*num_dataset, 6*num_dataset))



for index_dataset in range(num_dataset):
	dataset_name=results['rankboost'].keys()[index_dataset]
	plt.subplot(math.ceil(num_dataset/2), 3, index_dataset+1)
	plt.xlabel('Boosting Iterations')
	plt.ylabel('Instance AUC')
	iter_max_boosting=10
	
	color_index = -1
	legend_string=[]
	for method in ['rankboost', 'MIBoosting_Xu']:
		for statistic_name in results[method][dataset_name].keys():
			iter_max_boosting=max((iter_max_boosting, len(results[method][dataset_name][statistic_name])))
			color_index+=1
			#import pdb;pdb.set_trace()
	     		plt.plot(results[method][dataset_name][statistic_name], colors[color_index]+'*-')
			legend_string.append( method+ '_'+statistic_name  )
	plt.legend(legend_string )

	plt.title(dataset_name)
	plt.axis([0, iter_max_boosting+1, 0.49, 1.1])
		
#plt.savefig('rankboost_MIBoosting_Xu1.pdf')

plt.savefig('rankboost_MIBoosting_Xu.pdf')



#import pdb;pdb.set_trace()


