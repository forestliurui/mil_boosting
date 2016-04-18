"""
This is used to generate MIL data from noise model which tries to measure the influence of overlapping over the performance of adaboost on SIL
"""

import numpy as np
import random
from RankBoost_modiII_nondistributed import RankBoost_modiII
from Adaboost_nondistributed import AdaBoost
from MIBoosting_Xu_nondistributed import MIBoosting_Xu

try:
	from sklearn.metrics import roc_auc_score as score
except:
	from sklearn.metrics import auc_score as score

BOOSTERS = {
	"adaboost": AdaBoost,
	"rankboost_modiII": RankBoost_modiII,
	"miboosting_xu": MIBoosting_Xu
}

def generate_dataset(overlapping_percent):
	bag_size = 10

	num_bag_positive = 100
	num_bag_negative = 100

	positive_instance_bound = {}
	positive_instance_bound["upper"] = 50
	positive_instance_bound["lower"] = 0

	negative_instance_bound = {}
	negative_instance_bound["upper"] = 100
	negative_instance_bound["lower"] = 50

	#overlapping_percent = 0.5
	
	negative_instance_mid = (negative_instance_bound["upper"] + negative_instance_bound["lower"])/float(2)
	
	negative_instance_bound_negative_bag = {}
	negative_instance_bound_negative_bag["upper"]  = negative_instance_bound["upper"]
	negative_instance_bound_negative_bag["lower"] = negative_instance_mid- (negative_instance_mid - negative_instance_bound["lower"])*overlapping_percent

	negative_instance_bound_positive_bag = {}
	negative_instance_bound_positive_bag["upper"] = negative_instance_mid+ ( negative_instance_bound["upper"] - negative_instance_mid)*overlapping_percent
	negative_instance_bound_positive_bag["lower"] =  negative_instance_bound["lower"]

	positive_bags = []
	negative_bags = []

	positive_bags_instance_labels = []
	negative_bags_instance_labels = []

	positive_bags_bag_labels = []
	negative_bags_bag_labels = []

	for bag_index in range(num_bag_positive):
		bag = []
		bag_instance_label = []

		bag.append(np.array([random.uniform(positive_instance_bound["lower"], positive_instance_bound["upper"])]))
		bag_instance_label.append(True)

		for inst_index in range(bag_size - 1):
			bag.append(np.array([random.uniform(negative_instance_bound_positive_bag["lower"] , negative_instance_bound_positive_bag["upper"])]))
			bag_instance_label.append(False)

		positive_bags.append(np.vstack(bag))
		positive_bags_instance_labels.append(np.array(bag_instance_label) )
		positive_bags_bag_labels.append(True)
	

	for bag_index in range(num_bag_negative):
		bag = []
		bag_instance_label = []


		for inst_index in range(bag_size ):
			bag.append(np.array([random.uniform(negative_instance_bound_negative_bag["lower"] , negative_instance_bound_negative_bag["upper"])]))
			bag_instance_label.append(False)

		negative_bags.append(np.vstack(bag))
		negative_bags_instance_labels.append(np.array(bag_instance_label) )
		negative_bags_bag_labels.append(False)
	

	return positive_bags+negative_bags, np.hstack(( np.hstack((positive_bags_instance_labels)), np.hstack((negative_bags_instance_labels)) )), positive_bags_bag_labels+negative_bags_bag_labels



def test_case():
	booster_names = ["rankboost_modiII", "adaboost", "miboosting_xu"]
	results = {}
  	for booster_name in booster_names:
		results[booster_name] = {}
		train_inst_accuracy = []
		test_inst_accuracy = []
		train_bag_accuracy = []
		test_bag_accuracy = []

		train_inst_AUC = []
		test_inst_AUC = []
		train_bag_AUC = []
		test_bag_AUC = []
    		for percentage_index in range(10):
			percentage = percentage_index/float(10)
			train_bags, train_bags_instance_labels, train_bags_bag_labels = generate_dataset(percentage)
			test_bags, test_bags_instance_labels, test_bags_bag_labels = generate_dataset(percentage)

			#rankboost_modiII + decision stump
			params = {'weak_classifier': 'dtree_stump','max_depth': 1,'max_iter_boosting': 50}
			#bdt = RankBoost_modiII(**params)
			#bdt = AdaBoost(**params)
			bdt = BOOSTERS[booster_name](**params)

			bdt.fit(train_bags,  train_bags_bag_labels)

			train_inst_accuracy.append( np.average((bdt.predict_train(getInstPrediction =  True)>0 )== train_bags_instance_labels))
			test_inst_accuracy.append( np.average((bdt.predict(X_bags=test_bags,getInstPrediction =  True)>0 )== test_bags_instance_labels))

			train_bag_accuracy.append( np.average((bdt.predict_train(getInstPrediction =  False)>0 )== train_bags_bag_labels))
			test_bag_accuracy.append( np.average((bdt.predict(X_bags=test_bags,getInstPrediction =  False)>0 )== test_bags_bag_labels))

			train_inst_AUC.append( score(train_bags_instance_labels, bdt.predict_train(getInstPrediction =  True)  ))
			test_inst_AUC.append( score( test_bags_instance_labels,  bdt.predict(X_bags=test_bags,getInstPrediction =  True)  ))

			train_bag_AUC.append( score(train_bags_bag_labels, bdt.predict_train(getInstPrediction =  False)  ))
			test_bag_AUC.append( score(test_bags_bag_labels, bdt.predict(X_bags=test_bags,getInstPrediction =  False) ))			
		

			"""
			print "overlapping rate: %f" % percentage
			print "train_inst_accuracy: %f" % train_inst_accuracy
			print "test_inst_accuracy: %f" %test_inst_accuracy
			print "train_bag_accuracy: %f" %train_bag_accuracy
			print "test_bag_accuracy: %f" %test_bag_accuracy

			print ""
			"""
		results[booster_name]["train_inst_accuracy"] = train_inst_accuracy
		results[booster_name]["test_inst_accuracy"] = test_inst_accuracy
		results[booster_name]["train_bag_accuracy"] = train_bag_accuracy
		results[booster_name]["test_bag_accuracy"] = test_bag_accuracy

		results[booster_name]["train_inst_AUC"] = train_inst_AUC
		results[booster_name]["test_inst_AUC"] = test_inst_AUC
		results[booster_name]["train_bag_AUC"] = train_bag_AUC
		results[booster_name]["test_bag_AUC"] = test_bag_AUC

    	import pdb;pdb.set_trace()


if __name__ == "__main__":
	test_case()


	