#This is the implementation of iterative SVM proposed by Rui Liu. It will apply SVM to set of instances with each one of them from exactly one bag.
 
import numpy as np
import math
from mi_svm import MIKernelSVM, MIKernelSVR, SVM
from cvxopt import matrix as cvxmat, sparse, spmatrix
from scipy.io import loadmat
import csv
import misvm

DEBUG_MODE=False

def  read_csv_with_true_instance_labels_into_bag_list(file_name):
	
	#file_name is ended with ".csv". All entries in it should be real numbers. 
	#In csv, row corresponds to instances
	#the first and second columns to bag and instance id
	#the last column to true instance labels (1 corresponds to positive, negative symbols doesn't matter)

	#The return variables:
	#@bags: the list of arrays. Each array corresponds to a bag and its row to instance, column to features
	#@labels_bag: the list of values taking on +1/-1, corresponding to the bag labels
	#@labels_inst : the list of 1-D row arrays.  Each array corresponds to a bag. It takes on values +1/-1


	data_raw=np.genfromtxt(file_name, delimiter=",")  
	num_instances=data_raw.shape[0]

	label_raw=( 2*(data_raw[:, -1]==1)-1 ).tolist() #label_raw is the list for all instances which takes on +1/-1
	label_raw_bool=	(data_raw[:, -1]==1 ).tolist()

	instance_raw=data_raw[:, 2:-1] #raw instance matrices

	bags=[]
	labels_bag=[]
	labels_inst=[]
	labels_bag_bool=[]
			
	for row_index in range(num_instances):
		if row_index==0:
			start_index=0
		else:
			if data_raw[row_index, 0]!=data_raw[row_index-1, 0]: #bag_id changed 
				
				bags.append(instance_raw[ start_index: row_index ,:])
				labels_inst.append(np.array(label_raw[start_index: row_index ]))
				labels_bag_bool.append( reduce( lambda x, y: x|y,    label_raw_bool[start_index: row_index ] )  )
				start_index = row_index	
					
	
	bags.append(instance_raw[ start_index: ,:])
	labels_inst.append(np.array(label_raw[start_index:  ]))
	labels_bag_bool.append( reduce( lambda x, y: x|y,    label_raw_bool[start_index: ] )  )
	labels_bag=map(lambda x: 2*x-1, labels_bag_bool)
	
	
	return bags, labels_bag, labels_inst


def beta4subgradient_max(classifier, dataset, num_instance_each_bag_train):
	#This is the function to get the instance with greatest prediction score from each bag
	dataset=np.matrix(dataset)
	num_example, num_feature=dataset.shape
	
	
	
	beta_current=np.zeros((num_example,1))
	max_each_bag=np.zeros((len(num_instance_each_bag_train), 1))
	
	for bag_index_temp in range(len(num_instance_each_bag_train)):
		if bag_index_temp ==0:
			max_each_bag[bag_index_temp]=np.max( classifier.predict( dataset[0: num_instance_each_bag_train[bag_index_temp], :]  ))
		else:
			max_each_bag[bag_index_temp]=np.max( classifier.predict( dataset[sum(num_instance_each_bag_train[0:bag_index_temp]):sum( num_instance_each_bag_train[0:bag_index_temp+1]), :] ))
	
		for inst_index_temp in range(num_instance_each_bag_train[bag_index_temp]):
			if bag_index_temp == 0:
				if classifier.predict(dataset[inst_index_temp, :]) == max_each_bag[bag_index_temp]:
					beta_current[inst_index_temp]=1
					break
			else:
				if   classifier.predict(dataset[int(sum(num_instance_each_bag_train[0:bag_index_temp]))+inst_index_temp, :])   == max_each_bag[bag_index_temp]:
					beta_current[int(sum(num_instance_each_bag_train[0:bag_index_temp]))+inst_index_temp]=1
					break
		#import pdb;pdb.set_trace()
		'''
		if bag_index_temp == 0:
			beta_current[0:int(num_instance_each_bag_train[bag_index_temp])]= (  beta_current[0:int(num_instance_each_bag_train[bag_index_temp])]  )/sum( beta_current[0:int(num_instance_each_bag_train[bag_index_temp])] )
		else:
			beta_current[int(sum(  num_instance_each_bag_train[0:bag_index_temp]  )):  int(sum(  num_instance_each_bag_train[0:bag_index_temp+1]  ))]= (  beta_current[int(sum(  num_instance_each_bag_train[0:bag_index_temp]  )):  int(sum(  num_instance_each_bag_train[0:bag_index_temp+1]  ))]  )/sum( beta_current[int(sum(  num_instance_each_bag_train[0:bag_index_temp]  )):  int(sum(  num_instance_each_bag_train[0:bag_index_temp+1]  ))] )
		'''	
		#import pdb;pdb.set_trace()
	#import pdb;pdb.set_trace()
	
	beta_final=beta_current
	return beta_final


class Iterative_SVM(object):
	def __init__(self, **parameters):
		self.parameters=parameters
		self.instance_classifiers=[] #the list containing all the itermediate instance_classifiers that are trained. Typically, we only use the last one for testing
	def fit(self, X_bags, y_labels, weights_bag = None):
		#X_bags is a list of arrays, each bag is an array in the list
		#The row of array corresponds to instances in the bag, column corresponds to feature
		#y_labels is a list which contains the labels of bags. Here, binary labels are assumed, i.e. +1/-1
		#weights_bag is a list which contains the weights for bags

		if type(y_labels)!=list:
			y_labels=y_labels.tolist()
		
		if(type( y_labels[0] )==bool):
			y_labels = 2*y_labels-1
		
		Iteration_number=20
		
		num_bags=len(X_bags)
				
		num_instance_each_bag=[x.shape[0] for x in X_bags]
		instances=np.vstack((X_bags))

		instance_labels_generated_from_bag_labels=[y_labels[bag_index]*np.ones(( num_instance_each_bag[bag_index])) for bag_index in range(num_bags)  ]
		instance_labels_generated_from_bag_labels=np.hstack((instance_labels_generated_from_bag_labels)) #1-D row array: apply the bag label to its instances
		
		if weights_bag==None:
			weights_instance=None
		else:
			weights_instance= [( weights_bag[bag_index]/float(num_instance_each_bag[bag_index]) )*np.ones((num_instance_each_bag[bag_index])) for bag_index in range(num_bags)]
			weights_instance=np.hstack((weights_instance))

		#import pdb;pdb.set_trace()
		#initial classifier
		instance_classifier=SVM(**self.parameters)
		instance_classifier.fit(instances, instance_labels_generated_from_bag_labels, weights_instance)
		self.instance_classifiers.append(instance_classifier)

		iteration=0
		while True:
        		iteration=iteration+1		
			print "iteration #%d" % iteration

			#beta is the coefficient for subgradient of max function
			beta=beta4subgradient_max(instance_classifier, instances, num_instance_each_bag)
			beta_row_array=beta.transpose()[0]
			beta_bool=(beta_row_array==1)
			
			if DEBUG_MODE==True:
				print beta_bool
	
			if iteration >1:
				if DEBUG_MODE==True:
					print "The number of disagreement on beta between CCCP iterations: %d/%d" % (sum(beta_bool_old!=beta_bool)/2, len(num_instance_each_bag))
				if sum(beta_bool_old!=beta_bool)/2<=1:			
					break

			if iteration >Iteration_number:
				break

			beta_bool_old=beta_bool
			#import pdb;pdb.set_trace()
			
			training_set_current= instances[beta_bool,:]
	
			
			#import pdb;pdb.set_trace()

			instance_classifier=SVM(**self.parameters)
			instance_classifier.fit(training_set_current, y_labels, weights_bag)
			self.instance_classifiers.append(instance_classifier)

			#print classifier.predict(test_instances)

	def predict(self, X_bags):
		#X_bags is a list of arrays, each bag is an array in the list
		#The row of array corresponds to instances in the bag, column corresponds to feature

		#predictions_bag is the returned list of predictions which are real values 

		num_bags=len(X_bags)
		predictions_bag=[ max( self.instance_classifiers[-1].predict(X_bags[index_bag]) ) for index_bag in range(num_bags) ]

		predictions_bag=np.array( predictions_bag )
		return predictions_bag


	def predict_inst(self, X_bags):	
		#X_bags is a list of arrays, each bag is an array in the list
		#The row of array corresponds to instances in the bag, column corresponds to feature	

		#predictions_inst is the returned list of arrays which are instance-level real-valued predictions for all test bags. Note the array in the list here is row array (1-D)
		
		num_bags=len(X_bags)
		predictions_inst=[  self.instance_classifiers[-1].predict(X_bags[index_bag])  for index_bag in range(num_bags) ]
		
		return predictions_inst

	