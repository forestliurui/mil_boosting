#This is the implementation of RankBoost for bipartite setting, described in Figure 9.2 at the book "Foundations of Machine Learning"

from math import sqrt, exp
import string
import data
import numpy as np


INSTANCE_PREDICTIONS = True
INNER_CROSS_VALIDATION = False


class RankBoost(object):
	def __init__(self):
		self.raw_predictions = {}
		self.raw_predictions['bag']={}
		self.raw_predictions['bag']['train']=[]
		self.raw_predictions['bag']['test']=[]
		self.raw_predictions['instance']={}
		self.raw_predictions['instance']['train']=[]
		self.raw_predictions['instance']['test']=[]


		self.accum_predictions = {}
		self.accum_predictions['bag']={}
		self.accum_predictions['bag']['train']=[]
		self.accum_predictions['bag']['test']=[]
		self.accum_predictions['instance']={}
		self.accum_predictions['instance']['train']=[]
		self.accum_predictions['instance']['test']=[]

		self.alphas=[]
		self.epsilons_positive=[]
		self.epsilons_negative=[]
		self.Z_positive = []
		self.Z_negative = []

		self.results_manager=None
	
	def fit(self, train_dataset_name, auxiliary_struct):
		
		#get the name of train and test dataset 
		dataset_name=string.split(train_dataset_name,'.')[0]

		train_dataset_name_list=string.split(train_dataset_name,'.')

		test_dataset_name_list = train_dataset_name_list
		test_dataset_name_list[-1] = 'test'

		test_dataset_name = string.join(test_dataset_name_list,'.')
		#get the name of train and test dataset 

		train_dataset=data.get_dataset(train_dataset_name)
		test_dataset=data.get_dataset(test_dataset_name)
		self.train_dataset = train_dataset
		self.test_dataset = test_dataset

		self.train_dataset_name=train_dataset_name
		self.test_dataset_name=test_dataset_name

		instance_ids_train={}
		instance_ids_train['positive']=[]
		instance_ids_train['negative']=[]

		for inst_id in train_dataset.instance_ids: 
			#import pdb;pdb.set_trace()
			if train_dataset.bag_labels[ train_dataset.bag_ids.index(inst_id[0]) ]:
				instance_ids_train['positive'].append(inst_id)
			else:
				instance_ids_train['negative'].append(inst_id)

			
		max_iter_boosting=10

		key_statistic='test_instance_AUC'

		inst_weight_temp = {}
		#import pdb;pdb.set_trace()

		inst_weight_temp.update( dict.fromkeys(instance_ids_train['positive'] , float(1)/len(instance_ids_train['positive']) )  )
		inst_weight_temp.update( dict.fromkeys(instance_ids_train['negative'] , float(1)/len(instance_ids_train['negative']) ) )
		
		


		for iter_boosting in range(max_iter_boosting):
			print 'Boosting Iteration: %d' % iter_boosting
			auxiliary_struct['shared_variables']['inst_weights'][dataset_name] = inst_weight_temp

			task_key = run_tune_parameter(train_dataset_name, test_dataset_name , auxiliary_struct, key_statistic  ,label_index=None)
			task = auxiliary_struct['task_dict'][task_key]

			if self.results_manager==None:
				self.results_manager=task.results_manager
			
			task.store_boosting_raw_results(iter_boosting)

			for bag_or_inst in ['bag', 'instance']:
				for train_or_test in ['train', 'test']:

					self.raw_predictions[bag_or_inst][train_or_test].append(task.get_predictions(bag_or_inst,train_or_test))
					for x in self.raw_predictions[bag_or_inst][train_or_test][-1].keys():
						self.raw_predictions[bag_or_inst][train_or_test][-1][x] = (self.raw_predictions[bag_or_inst][train_or_test][-1][x] > 0 ) +0

			#self.raw_predictions['bag']['test'].append(task.get_predictions('bag','test'))

			#self.raw_predictions['instance']['train'].append(task.get_predictions('instance','train'))
			#self.raw_predictions['instance']['test'].append(task.get_predictions('instance','test'))
			
			

			
			self.epsilons_positive.append( dict_average(self.raw_predictions['instance']['train'][-1],  inst_weight_temp, instance_ids_train['positive'] )    )
			self.epsilons_negative.append(  dict_average(self.raw_predictions['instance']['train'][-1],  inst_weight_temp, instance_ids_train['negative'] )     )

			if self.epsilons_negative[-1] == 0:
				self.alphas.append(20)
				break

			self.alphas.append(0.5*np.log(self.epsilons_positive[-1]/self.epsilons_negative[-1]))
			#self.alphas.append(0.5)
			
			self.Z_positive.append( 1-self.epsilons_positive[-1]+sqrt( self.epsilons_positive[-1]*self.epsilons_negative[-1] ) )
			self.Z_negative.append( 1-self.epsilons_negative[-1]+sqrt( self.epsilons_positive[-1]*self.epsilons_negative[-1] ) )

			for inst_id in instance_ids_train['positive']:
				inst_weight_temp[inst_id] = inst_weight_temp[inst_id]*exp(-self.alphas[-1]*self.raw_predictions['instance']['train'][-1][inst_id] )/self.Z_positive[-1]

			for inst_id in instance_ids_train['negative']:
				inst_weight_temp[inst_id] = inst_weight_temp[inst_id]*exp(self.alphas[-1]*self.raw_predictions['instance']['train'][-1][inst_id] )/self.Z_negative[-1]
			#import pdb;pdb.set_trace()

		self.num_iter_boosting=len(self.alphas)

	def predict(self):
		predictions_list={}
		predictions_list['bag']={}
		predictions_list['instance']={}
		predictions_list['bag']['train']=[]
		predictions_list['bag']['test']=[]
		predictions_list['instance']['train']=[]
		predictions_list['instance']['test']=[]

		for iter_index in range(self.num_iter_boosting):
			predictions_list['bag']['train'].append([self.raw_predictions['bag']['train'][iter_index][x] for x in self.train_dataset.bag_ids  ])
			predictions_list['bag']['test'].append([self.raw_predictions['bag']['test'][iter_index][x] for x in self.test_dataset.bag_ids  ])

			predictions_list['instance']['train'].append([self.raw_predictions['instance']['train'][iter_index][x] for x in self.train_dataset.instance_ids  ])
			predictions_list['instance']['test'].append([self.raw_predictions['instance']['test'][iter_index][x] for x in self.test_dataset.instance_ids  ])

		predictions_matrix={}
		predictions_matrix['bag']={}
		predictions_matrix['instance']={}
		#import pdb;pdb.set_trace()
		predictions_matrix['bag']['train']=np.matrix( np.vstack((predictions_list['bag']['train']))  )
		predictions_matrix['bag']['test']=np.matrix( np.vstack((predictions_list['bag']['test']))  )
		predictions_matrix['instance']['train']=np.matrix( np.vstack((predictions_list['instance']['train']))  )
		predictions_matrix['instance']['test']=np.matrix( np.vstack((predictions_list['instance']['test']))  )

		for iter_index in range(self.num_iter_boosting):
			#import pdb;pdb.set_trace()
			temp_train=np.matrix(self.alphas[0:iter_index+1])*predictions_matrix['instance']['train'][0:iter_index+1, :]
			self.accum_predictions['instance']['train'].append(dict(zip( self.train_dataset.instance_ids ,  temp_train.tolist()[0]     )))

			temp_test=np.matrix(self.alphas[0:iter_index+1])*predictions_matrix['instance']['test'][0:iter_index+1, :] 
			self.accum_predictions['instance']['test'].append(  dict(zip( self.test_dataset.instance_ids ,  temp_test.tolist()[0]     )) )

			temp_bag_train={}
			temp_bag_test={}
			for bag_id in self.train_dataset.bag_ids:
				#import pdb;pdb.set_trace()
				temp_bag_train[bag_id]=np.average([self.accum_predictions['instance']['train'][-1][x] for x in self.train_dataset.instance_ids if bag_id in x   ])
			for bag_id in self.test_dataset.bag_ids:
				temp_bag_test[bag_id]=np.average([self.accum_predictions['instance']['test'][-1][x] for x in self.test_dataset.instance_ids if bag_id in x   ])

			self.accum_predictions['bag']['train'].append(temp_bag_train)
			self.accum_predictions['bag']['test'].append(temp_bag_test)
			
		#import pdb;pdb.set_trace()			


	def store_boosting_results(self, boosting_rounds):
		#boosting_rounds starts from 1
		submission_boosting={}

		submission_boosting['accum']={}
		submission_boosting['accum']['instance']={}
		submission_boosting['accum']['bag']={}

		submission_boosting['accum']['instance']['train']=self.accum_predictions['instance']['train'][boosting_rounds-1]
		submission_boosting['accum']['instance']['test']=self.accum_predictions['instance']['test'][boosting_rounds-1]

		submission_boosting['accum']['bag']['train']=self.accum_predictions['bag']['train'][boosting_rounds-1]
		submission_boosting['accum']['bag']['test']=self.accum_predictions['bag']['test'][boosting_rounds-1]

		train_bag_labels=np.array([submission_boosting['accum']['bag']['train'][x] for x in self.train_dataset.bag_ids ])
		bag_predictions=np.array([submission_boosting['accum']['bag']['test'][x] for x in self.test_dataset.bag_ids ])

		train_instance_labels=np.array([submission_boosting['accum']['instance']['train'][x] for x in self.train_dataset.instance_ids ])
		instance_predictions=np.array([submission_boosting['accum']['instance']['test'][x] for x in self.test_dataset.instance_ids ])

		try:
            		from sklearn.metrics import roc_auc_score as score
        	except:
            		from sklearn.metrics import auc_score as score
        	scorename = 'AUC'

		submission_boosting['statistics_boosting']={}
		
		train=self.train_dataset
		test=self.test_dataset
		
		if train.bag_labels.size > 1:
	    		train_bag_accuracy = np.average( train.bag_labels== ( train_bag_labels > 0  )  )
	    		train_bag_balanced_accuracy= np.average( [ np.average( train_bag_labels[train.bag_labels]>0 ) ,   np.average( train_bag_labels[train.bag_labels==False]<0 ) ] )
            		print ('Training Bag %s score: %f, accuracy: %f, balanced accuracy: %f'
                   		% (scorename, score(train.bag_labels, train_bag_labels) ,train_bag_accuracy, train_bag_balanced_accuracy ))
	    		submission_boosting['statistics_boosting']['train_bag_'+scorename] = score(train.bag_labels, train_bag_labels)
	    		submission_boosting['statistics_boosting']['train_bag_accuracy']=train_bag_accuracy
	    		submission_boosting['statistics_boosting']['train_bag_balanced_accuracy']=train_bag_balanced_accuracy


        	if INSTANCE_PREDICTIONS and train.instance_labels.size > 1:
	    		train_instance_accuracy = np.average( train.instance_labels== ( train_instance_labels > 0  )  )
	    		train_instance_balanced_accuracy= np.average( [ np.average( train_instance_labels[train.instance_labels]>0 ) ,   np.average( train_instance_labels[train.instance_labels==False]<0 ) ]  )
            		print ('Training Inst. %s Score: %f, accuracy: %f, balanced accuracy: %f'
                   		% (scorename, score(train.instance_labels, train_instance_labels) ,train_instance_accuracy, train_instance_balanced_accuracy ))
            		submission_boosting['statistics_boosting']['train_instance_'+scorename] = score(train.instance_labels, train_instance_labels)
	    		submission_boosting['statistics_boosting']['train_instance_accuracy']=train_instance_accuracy
	    		submission_boosting['statistics_boosting']['train_instance_balanced_accuracy']=train_instance_balanced_accuracy

        	if test.bag_labels.size > 1:
	    		test_bag_accuracy = np.average( test.bag_labels== ( bag_predictions > 0  )  )
	    		test_bag_balanced_accuracy= np.average( [ np.average( bag_predictions[test.bag_labels]>0 ) ,   np.average( bag_predictions[test.bag_labels==False]<0 ) ]  )
            		   
	    		print ('Test Bag %s Score: %f, accuracy: %f, balanced accuracy: %f'
                   		% (scorename, score(test.bag_labels, bag_predictions), test_bag_accuracy, test_bag_balanced_accuracy ))

	    		submission_boosting['statistics_boosting']['test_bag_'+scorename] = score(test.bag_labels, bag_predictions)
  	    		submission_boosting['statistics_boosting']['test_bag_accuracy']=test_bag_accuracy
	    		submission_boosting['statistics_boosting']['test_bag_balanced_accuracy']=test_bag_balanced_accuracy

        	if INSTANCE_PREDICTIONS and test.instance_labels.size > 1:
   	    		test_instance_accuracy = np.average( test.instance_labels== ( instance_predictions > 0  )  )
	    		test_instance_balanced_accuracy= np.average( [ np.average( instance_predictions[test.instance_labels]>0 ) ,   np.average( instance_predictions[test.instance_labels==False]<0 ) ]  )

           	 	print ('Test Inst. %s Score: %f, accuracy: %f, balanced accuracy: %f'
                   		% (scorename, score(test.instance_labels, instance_predictions),test_instance_accuracy, test_instance_balanced_accuracy ))
	    		submission_boosting['statistics_boosting']['test_instance_'+scorename] = score(test.instance_labels, instance_predictions)
	    		submission_boosting['statistics_boosting']['test_instance_accuracy']=test_instance_accuracy
	    		submission_boosting['statistics_boosting']['test_instance_balanced_accuracy']=test_instance_balanced_accuracy

		self.results_manager.store_results_boosting(submission_boosting, boosting_rounds, self.train_dataset_name, self.test_dataset_name, 100, 100)



			

def dict_average(dict_predictions, dict_weights, ids):
	
	list_predictions =[ dict_predictions[id] for id in ids ]
	list_weights = [dict_weights[id] for id in ids  ]
	return np.average(list_predictions, weights = list_weights )


def compute_statistic(predictions, labels ,weights ,ids ,statistic_name):
	# predicitons, weights are dictionaries; lables are list which is ordered by ids
	# statistic_name is string which specified the name of statistic to be compute
	
	predictions_list=[ predictions[x] for x in ids]
	if weights == None:
		weights_list=None
	else:
		weights_list=[ weights[x] for x in ids]

	if statistic_name == 'accuracy':
		statistic = np.average(( np.array(predictions_list)>0) == labels, weights=weights_list )

	return statistic

def run_tune_parameter(train, test , auxiliary_struct, key_statistic  ,label_index=None):
    #train is the string for training dataset
    #test is the string for testing dataset
    #tasks is the all possible tasks in dictionary format, i.e. task_dict
    #shared_variables contains two conponents: one is the queue to be run, the second one is condition_lock that synchronize
    #label_index is the index of label with respect to which the optimal parameter combination is determined    
    #key_statistic is the string which specifies the statistic used to pick the best parameter

    #this function will return the optimal task on the training set/testing set pair
    
    tasks = auxiliary_struct['task_dict']
    shared_variables = auxiliary_struct['shared_variables']
    server = auxiliary_struct['server']
    
    if INNER_CROSS_VALIDATION == True:
    	#import pdb; pdb.set_trace()
    	#run the experiment train with the best parameter tuned on train
    	subtasks=dict((k, tasks[k] ) for k in tasks.keys()  if k[2].find(train+'.')==0    ) #subtasks is the dictionary which contains the tasks to tune the parameters for train
    	with server.status_lock:
    		for sub_key in subtasks.keys():
			subtasks[sub_key].finished = False
    	shared_variables['to_be_run'].put(subtasks)
    	shared_variables['condition_lock'].acquire()
    
    	#import pdb; pdb.set_trace()
    	while(not reduce(lambda x, y: x and y, [ tasks[z].finished for z in subtasks.keys()   ]   )):  #if all tasks are finished
    		print 'blocked by wait'
       		shared_variables['condition_lock'].wait()
		print 'awakened from wait'
    
    	shared_variables['condition_lock'].release()  
    	print 'all subtasks used for tuning parameters are finished'
    	print 'try to choose the optimal parameters for this training dataset'
    
    	num_para_combination=max([ subtasks.keys()[x][5] for x in range(len(subtasks) )  ])+1
    	statistic_avg_per_para={}
    
    	if label_index is None:
    		statisitic_name=key_statistic
    	else:
		statisitic_name=key_statistic+str(label_index)

    	for para_index in range(num_para_combination):
 		statistic_avg_per_para[para_index]=np.mean( [tasks[x].get_statistic(statisitic_name)[0] for x in subtasks.keys() if x[5]==para_index] ) 
    
    	para_index_optimal = np.argmax(statistic_avg_per_para.values())

    else:
	para_index_optimal = 0

    subtasks=dict((k, tasks[k] ) for k in tasks.keys()  if k[2]== train and k[5] == para_index_optimal    )
    
    with server.status_lock:
     	for sub_key in subtasks.keys():
		subtasks[sub_key].finished = False    
    shared_variables['to_be_run'].put(subtasks)
    shared_variables['condition_lock'].acquire()
    while(not reduce(lambda x, y: x and y, [ tasks[z].finished for z in subtasks.keys()   ]   )):  #if all tasks are finished
    	print 'blocked by wait'
       	shared_variables['condition_lock'].wait()
	print 'awakened from wait'
    
    shared_variables['condition_lock'].release()  
    print 'all subtasks are finished'
    
    print 'parameter tuning on training set'+train+' is finished'
    
    return subtasks.keys()[0]  #return the key of the optimal task for training set "train"
    #import pdb; pdb.set_trace()  
    
   

			