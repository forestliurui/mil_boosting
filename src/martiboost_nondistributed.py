"""
This is the implementation of Martingale Boosting in the paper "Philip Long, Rocco Servedio, Martingale Boosting"
"""

import copy

class TreeNode(object):
	def __init__(self, classifier, weights):
		self.classifier = copy.deepcopy( classifier )
		self.weights = dict(weights)
		self.left = None
		self.right = None
	


class MartiBoost(object):
	def __init__(self, **parameters):
		self.head = None

	def fit(self, X_bags, y_labels):
		'''
		X_bags is a list of arrays, each bag is an array in the list
		The row of array corresponds to instances in the bag, column corresponds to feature
		y_labels is the list which contains the labels of bags. Here, binary labels are assumed, i.e. +1/-1
		'''
		
		

		if type(y_labels)!=list:
			y_labels=y_labels.tolist()
		
		if type( y_labels[0] )==bool or 0 in y_labels:  #convert the boolean labels into +1/-1 labels
			y_labels = map(lambda x:2*x-1, y_labels )
		
		max_iter_boosting=self.max_iter_boosting

		if type(X_bags) == list:  # treat it as the SIL in MIL setting
			num_bags=len(X_bags)
				
			num_instance_each_bag=[x.shape[0] for x in X_bags]
		
			instances=np.vstack((X_bags))
			

			instance_labels_generated_from_bag_labels=[y_labels[bag_index]*np.ones((1, num_instance_each_bag[bag_index]))[0] for bag_index in range(num_bags)  ]
			instance_labels_generated_from_bag_labels=np.hstack((instance_labels_generated_from_bag_labels))		
		else:
			instances = X_bags
			instance_labels_generated_from_bag_labels = y_labels

		instance_labels_generated_from_bag_labels = np.array( instance_labels_generated_from_bag_labels )
		num_instances = instances.shape[0]
		num_instances_positive = np.sum(instance_labels_generated_from_bag_labels == 1)	
		num_instances_negative = np.sum(instance_labels_generated_from_bag_labels != 1)		

		self.c=[] #the list of weights for weak classifiers

		#import pdb;pdb.set_trace()

		#initial inst weights
		weights_inst= np.ones((num_instances))
		for inst_index in range(num_instances):
			if instance_labels_generated_from_bag_labels[inst_index]==1:
				weights_inst[inst_index] = float(1)/num_instances_positive
			else:
				weights_inst[inst_index] = float(1)/num_instances_negative

		instance_classifier=SVM(**self.parameters)
		
		self.head = TreeNode(instance_classifier, weights_inst)


		current_level_list = []
		current_level_list.append(self.head)

		for index_Boosting in range(max_iter_boosting):

			next_level_list = []
			
			for current in current_level_list:
		
				#import pdb;pdb.set_trace()

				current.classifier.fit(instances, instance_labels_generated_from_bag_labels.tolist(), current.weights)
						
				
			
				current.training_predictions = instance_classifier.predict(instances)
				current.errors_instance = {}
				current.errors_instance["positive"] = np.average(current.training_predictions[instance_labels_generated_from_bag_labels == 1]>0)
				current.errors_instance["negative"] = np.average(current.training_predictions[instance_labels_generated_from_bag_labels != 1]<0)
			
			
				current_prediction_num = {}
				current_prediction_num["positive"] = {}
				current_prediction_num["negative"] = {}
			
				current_prediction_num["positive"]["positive"] = np.sum( instance_labels_generated_from_bag_labels[ current.training_predictions >0 ] ==1 ) #the num of true positive
				current_prediction_num["positive"]["negative"] = np.sum( instance_labels_generated_from_bag_labels[ current.training_predictions >0 ] != 1) #the num of false positive

				current_prediction_num["negative"]["positive"] = np.sum( instance_labels_generated_from_bag_labels[ current.training_predictions <0 ] ==1 ) #the num of false negative
				current_prediction_num["negative"]["negative"] = np.sum( instance_labels_generated_from_bag_labels[ current.training_predictions <0 ] != 1) #the num of true negative

				for inst_index in range(num_instances):

					weights_inst[inst_index] = 0
					if current.training_predictions[inst_index] < 0:
						if instance_labels_generated_from_bag_labels >0 :
							weights_inst[inst_index] = float(1)/current_prediction_num["negative"]["positive"]
						else: 
							weights_inst[inst_index] = float(1)/current_prediction_num["negative"]["negative"]
					
				instance_classifier=SVM(**self.parameters)
				current.left = TreeNode(instance_classifier, weights_inst)	
				next_level_list.append(current.left)

				for inst_index in range(num_instances):
					weights_inst[inst_index] = 0

					if current.training_predictions[inst_index] > 0:
						if instance_labels_generated_from_bag_labels >0 :
							weights_inst[inst_index] = float(1)/current_prediction_num["positive"]["positive"]
						else: 
							weights_inst[inst_index] = float(1)/current_prediction_num["positive"]["negative"]
						
				instance_classifier=SVM(**self.parameters)
				current.right = TreeNode(instance_classifier, weights_inst)

				next_level_list.append(current.right)
			
			current_level_list = next_level_list
					
			
