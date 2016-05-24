"""
This is used to process the movie len dataset
"""
import csv
import dill
import pickle
from RankBoost_modiIV_ranking_nondistributed import RankBoost_modiIV_ranking


def run_experiments():
	pkl_file = open('ranking/LETOR/LETOR.pkl', 'r')

	dataset = dill.load(pkl_file)
	parameter = {"max_iter_boosting":500}
	
	results = {}
	
	ranker = RankBoost_modiIV_ranking(**parameter)
	print "start training"
	ranker.fit(dataset.X, dataset.y_train)
	print "finish training"

	results["ranker"] = ranker
	results["train_error"] = []
	results["test_error"] = []
	for j in range(1, 100):
		print  " iteration: ", j
		predictions = ranker.predict_train(iter = j)
		error = ranker.getRankingError(predictions, dataset.y_train)
		results["train_error"].append(error)

		if j == 1:
			predictions = ranker.predict(dataset.X_test, iter = j)
		else:
			predictions = ranker.predict( iter = j)
		error = ranker.getRankingError(predictions, dataset.y_test)
		results["test_error"].append(error)

	#import pdb;pdb.set_trace()
	dill.dump(results, open("ranking/LETOR/results_modiIII.pkl", "wb"))
	import pdb;pdb.set_trace()
	

class ranking_data(object):
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X  = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test

if __name__ == "__main__":
	#process()
	run_experiments()
