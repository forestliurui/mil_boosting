#This is the adaboost for server end

import string

class Adaboost(object):
	def __init__(self):
		pass
	def fit(self, train_dataset_name, auxiliary_struct):

		#get the name of train and test dataset 
		dataset_name=string.split(train_dataset_name,'.')[0]

		train_dataset_name_list=string.split(train_dataset_name,'.')

		test_dataset_name_list = train_dataset_name_list
		test_dataset_name_list[-1] = 'test'

		test_dataset_name = string.join(test_dataset_name_list,'.')\
		#get the name of train and test dataset 

		train_dataset=data.get_dataset(train_dataset_name)
		test_dataset=data.get_dataset(test_dataset_name)


		max_iter_boosting=10

		key_statistic='test_bag_balanced_accuracy'

		auxiliary_struct['shared_variables']['bag_weights'][dataset_name] = dict.fromkeys(train_dataset.bag_ids,1)


		for iter_boosting in range(max_iter_boosting):
			

			task=run_tune_parameter(train_dataset, test_dataset , auxiliary_struct, key_statistic  ,label_index=None):


	def predict(self):

	def store_boosting_results(self):	


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
    
   


