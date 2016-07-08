"""
This is used to process the movie len dataset
"""
import csv
import dill
import pickle
from RankBoost_modiIII_ranking_nondistributed import RankBoost_modiIII_ranking

def process():
	filename = "ranking/LETOR/LETOR_raw.csv"
	query_map = {}
	movie_map = {}
	inst_id = 0
	
	with open(filename, "rb") as f:
		
		reader = csv.reader(f, delimiter = " ")
		for row in reader:
			if row[-1] == "":
				row.pop()
			#print row
			
			inst_id += 1
			score = int(row[0])
			query_id = int(row[1].split(":")[1])

			instance = []

			if query_id not in query_map:
				query_map[query_id] = {}
			query_map[query_id][inst_id] = {}
			query_map[query_id][inst_id]["instance"] = [float(item.split(":")[1]) for item in row[2:]]
			query_map[query_id][inst_id]["score"] = score

	#import pdb;pdb.set_trace()
	num_query = len(query_map)

	num_query_train = int(4*(num_query/float(5)))
		
	X = {}
	y_train = []
	X_test = {}
	y_test = []
	for query_index in range(num_query):
		query_id = query_map.keys()[query_index]
		score_distribution_train = {0:[], 1:[], 2:[], 3:[], 4:[]}
		score_distribution_test = {0:[], 1:[], 2:[], 3:[], 4:[]}
		if query_index < num_query_train:
			for inst_id in query_map[query_id]:
				score = query_map[query_id][inst_id]["score"]
				
				X[inst_id] = query_map[query_id][inst_id]["instance"]
				
				if score!=0 and len(score_distribution_train[score-1])!= 0:
					y_train+= [(inst_id, x) for x in score_distribution_train[score-1]]
				score_distribution_train[score].append(inst_id)
		else:
			for inst_id in query_map[query_id]:
				score = query_map[query_id][inst_id]["score"]
				
				X_test[inst_id] = query_map[query_id][inst_id]["instance"]
				
				if score!=0 and len(score_distribution_test[score-1])!= 0:
					y_test+= [(inst_id, x) for x in score_distribution_test[score-1]]
				score_distribution_test[score].append(inst_id)
				

	LETOR = ranking_data(X, y_train, X_test, y_test)	
	
	dill.dump(LETOR, open('ranking/LETOR/LETOR.pkl', 'wb'))
	import pdb;pdb.set_trace()

def run_experiments():
	pkl_file = open('ranking/LETOR/LETOR.pkl', 'r')

	dataset = dill.load(pkl_file)
	parameter = {"max_iter_boosting":500}
	
	results = {}
	
	ranker = RankBoost_modiIII_ranking(**parameter)
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

