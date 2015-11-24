#This is the implementation of iterative SVM proposed by Rui Liu. It will apply SVM to set of instances that are selected from bags. Postive bags will give out 2 instances, one with greatest prediction and the other with smallest prediction
#Negative bags always give out 1 instance with greatest prediction from previous iteration.
 
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


def index_selector(classifier, X_bags, y_labels, bag_weights=None):
	#This is the function to get the instance with greatest prediction score from each bag and smallest prediction from positive bags
	
	num_bags=len(X_bags)

	positive_instances=[] #list for positive instances
	negative_instances=[] #list for negative instances
	negative_weights_final=[]
	positive_weights_final=[]

	for index_bag in range(num_bags):
		current_predictions=classifier.predict(X_bags[index_bag])
		max_predictions=np.max(current_predictions)
		min_predictions=np.min(current_predictions)

		num_inst=len( current_predictions.tolist() )

		for index_inst in range(num_inst):
			if current_predictions[index_inst]==max_predictions:
				if y_labels[index_bag]== 1:
					positive_instances.append(X_bags[index_bag][index_inst,:])
					if bag_weights!=None:
						positive_weights_final.append( bag_weights[index_bag])
				else:
					negative_instances.append(X_bags[index_bag][index_inst,:])
					if bag_weights!=None:
						negative_weights_final.append( bag_weights[index_bag])

			
			
			if current_predictions[index_inst]==min_predictions and max_predictions!=min_predictions:
				if y_labels[index_bag]== 1:
					negative_instances.append(X_bags[index_bag][index_inst,:])
					if bag_weights!=None:
						negative_weights_final.append( bag_weights[index_bag])


	instances_final=np.vstack((positive_instances+ negative_instances ))
	labels_final= np.hstack(( np.ones(  len(positive_instances)), -np.ones(  len(negative_instances))  ))
	if bag_weights!=None:
		weights_final=np.hstack((positive_weights_final, negative_weights_final ))
	else:
		weights_final=None
	
	return instances_final, labels_final, weights_final


class Iterative_SVM_pn(object):
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
			instances_current, labels_current, weights_current=index_selector(instance_classifier, X_bags, y_labels, weights_bag)
			

			if iteration >Iteration_number:
				break

			#import pdb;pdb.set_trace()
			
			
	
			
			#import pdb;pdb.set_trace()

			instance_classifier=SVM(**self.parameters)
			instance_classifier.fit(instances_current, labels_current, weights_current)
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

	