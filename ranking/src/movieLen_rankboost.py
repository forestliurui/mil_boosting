"""
In this script, I will run rankboost on movieLen
"""

import dill
from process_movieLen_dataset import movieLenData
from RankBoost_ranking_nondistributed import RankBoost_ranking
import sys
sys.path.append("/home/rui/MIL_Boost/MIL_Boosting/MIL_Boost/MIL_Boost/src/")
from results import ResultsManager

def run_experiments():
	

	parameter = {"max_iter_boosting":500, 'weak_classifier': 'stump_ranker'}
	
	results_manager = ResultsManager("ranking/movieLen/results/rankboost_test.db")

	results = {}
	for index in range(50):
		if index != 2:
			continue

		user = str(index)
		movieLen = dill.load(open('ranking/movieLen/movieLen_user'+str(index)+'.pkl'))
		if results_manager.is_finished("train", user, '0', '1'):
			continue
		print "test user: ", index

		ranker = RankBoost_ranking(**parameter)

		print "user: ", index, " training begins"

		ranker.fit(movieLen.X_train[0], movieLen.p_train[0])
		
		print "user: ", index, " training ends"

		results['statistics_boosting'] = {}
		#results['statistics_boosting']["ranker"] = ranker
		results['statistics_boosting']["train_error"] = None
		results['statistics_boosting']["test_error"] = None
		for j in range( 1, min(101, ranker.actual_rounds_of_boosting+1) ):
			print "user: ", index, " iteration: ", j
			predictions = ranker.predict_train(iter = j)
			error = ranker.getRankingError(predictions, movieLen.p_train[0])
			results['statistics_boosting']["train_error"] = error
			if j == 1:
				predictions = ranker.predict(movieLen.X_test[0], iter = j)
			else:
				predictions = ranker.predict( iter = j)
			error = ranker.getRankingError(predictions, movieLen.p_test[0])
			results['statistics_boosting']["test_error"] = error 

			#import pdb;pdb.set_trace()

			results_manager.store_results_boosting(results, j, "test", user, '0', '1')
		#dill.dump(results, open("ranking/result/results_rankboost_user_"+user+".pkl", "wb"))

	#import pdb;pdb.set_trace()
	#dill.dump(results, open("ranking/results_rankboost.pkl", "wb"))
	import pdb;pdb.set_trace()


if __name__ == "__main__":
	run_experiments()