"""
This is used to process the movie len dataset
"""
import csv
import dill
import pickle
from RankBoost_ranking_nondistributed import RankBoost_ranking
import sys
sys.path.append("/home/rui/MIL_Boost/MIL_Boosting/MIL_Boost/MIL_Boost/src/")
from results import ResultsManager

MISSING_VAL = -1



def process():
	filename = "movieLen.csv"
	user_map = {}
	movie_map = {}
	with open(filename, "rb") as f:
		reader = csv.reader(f, delimiter = ",")
		for row in reader:
			if row[0] not in user_map:
				user_map[row[0]] = {}
			user_map[row[0]][row[1]] = int(row[2])

			if row[1] not in movie_map:
				movie_map[row[1]] = {}
			movie_map[row[1]][row[0]] = int(row[2])

	"""
	active_user = [user[0] for user in user_map.items() if len(user[1])]

	pop_movie = [movie[0] for movie in movie_map.items() if len(movie[1])]

	user_map1 = {}
	movie_map1 = {}
	with open(filename, "rb") as f:
		reader = csv.reader(f, delimiter = ",")
		for row in reader:
		    if row[0] in active_user and row[1] in pop_movie:
			if row[0] not in user_map1:
				user_map1[row[0]] = {}
			user_map1[row[0]][row[1]] = row[2]

			if row[1] not in movie_map1:
				movie_map1[row[1]] = {}
			movie_map1[row[1]][row[0]] = row[2]
	"""
	num_user = len(user_map)

	num_user_train = num_user/2
	num_movie = len(movie_map)
	
	X = {}
	for movie in movie_map.keys():
		X[movie] = []
		for user_index in range(num_user_train):
			user = user_map.keys()[user_index]
			#import pdb;pdb.set_trace()
			if movie in user_map[user]:
				X[movie].append(user_map[user][movie])
			else:
				X[movie].append(0)
	y = {}
	y_train = {}
	y_test = {}
	for user_index in range(num_user_train, num_user):
		user = user_map.keys()[user_index]
		y[user] = []

		for movie1 in user_map[user]:
			for movie2 in user_map[user]:
				if user_map[user][movie1] > user_map[user][movie2]:
					y[user].append( (movie1, movie2)  )
		y_train[user] = y[user][0:2*len(y[user])/3]
		y_test[user] =  y[user][2*len(y[user])/3:]
	movieLen = ranking_data(X, y_train, y_test)	
	output = open('movieLen.pkl', 'wb')
	dill.dump(movieLen, output)
	import pdb;pdb.set_trace()

def process_remove_missing():
	"""
	The missing values are removed for target user
	Target user is the user which is used to construct critical pairs
	"""
	filename = "ranking/movieLen.csv"
	user_map = {}
	movie_map = {}
	with open(filename, "rb") as f:
		reader = csv.reader(f, delimiter = ",")
		for row in reader:
			if row[0] not in user_map:
				user_map[row[0]] = {}
			user_map[row[0]][row[1]] = int(row[2])

			if row[1] not in movie_map:
				movie_map[row[1]] = {}
			movie_map[row[1]][row[0]] = int(row[2])
	num_user = len(user_map)
	num_movie = len(movie_map)


	X_total = {} #X_total is a dictionary mapping movie_id into a list which contains the rating from users other than target user 
	target_user = user_map.keys()[0]
	X_target = {} #X_target is a dictionary mapping movie_id into a value which is the rating from target user
	for movie in movie_map.keys():
		if movie in user_map[target_user]:
			X_total[movie] = []
			for user in user_map.keys():
				if user != target_user:
					if movie in user_map[user]:
						X_total[movie].append(user_map[user][movie])
					else:
						X_total[movie].append(MISSING_VAL)
				else:
					X_target[movie] = user_map[user][movie]

	movieLen = split_train_test(X_total, X_target)
	dill.dump(movieLen, open('ranking/movieLen1.pkl', 'wb'))
	import pdb;pdb.set_trace()
	
def split_train_test(X_total, X_target):
	#split X_total and X_target into training and testing part
	X_total_train = {}
	X_target_train = {}
	X_total_test = {}
	X_target_test = {}
	
	train_ratio = 0.8
		
	for movie_index in range(len(X_total)):
		movie = X_total.keys()[movie_index]
		if movie_index < train_ratio*len(X_total):
			X_total_train[movie] = X_total[movie]
			X_target_train[movie] = X_target[movie]
		else:
			X_total_test[movie] = X_total[movie]
			X_target_test[movie] = X_target[movie]

	#construct critical pairs from target_user
	y_train = []
	y_test = []	
	for movie1 in X_target_train.keys():
		for movie2 in X_target_train.keys():
			
			if X_target_train[movie1] > X_target_train[movie2]:
				y_train.append( (movie1, movie2) )
			elif X_target_train[movie1] < X_target_train[movie2]:
				y_train.append( (movie2, movie1) )
	
	for movie1 in X_target_test.keys():
		for movie2 in X_target_test.keys():
			if X_target_test[movie1] > X_target_test[movie2]:
				y_test.append( (movie1, movie2) )
			elif X_target_test[movie1] < X_target_test[movie2]:
				y_test.append( (movie2, movie1) )
	
	return ranking_data_split(X_total_train, X_total_test, y_train, y_test)

def run_experiments_remove_missing():
	pkl_file = open('ranking/movieLen1.pkl', 'r')

	movieLen = dill.load(pkl_file)
	parameter = {"max_iter_boosting":500, 'weak_classifier': 'stump_ranker'}
	
	results_manager = ResultsManager("ranking/result/database/rankboost_remove_missing.db")

	results = {}

	user = 0	

	ranker = RankBoost_ranking(**parameter)		

	print "training"
	ranker.fit(movieLen.X_train, movieLen.y_train)
	print "finish training"

	results['statistics_boosting'] = {}
	#results['statistics_boosting']["ranker"] = ranker
	results['statistics_boosting']["train_error"] = None
	results['statistics_boosting']["test_error"] = None
	for j in range( 1, min(101, ranker.actual_rounds_of_boosting+1) ):
		print "get prediction results for boosting round: ", j		
		predictions = ranker.predict_train(iter = j)
		error = ranker.getRankingError(predictions, movieLen.y_train)
		results['statistics_boosting']["train_error"] = error
		if j == 1:
			predictions = ranker.predict(movieLen.X_test, iter = j)
		else:
			predictions = ranker.predict( iter = j)
		error = ranker.getRankingError(predictions, movieLen.y_test)
		results['statistics_boosting']["test_error"] = error 
		results_manager.store_results_boosting(results, j, "test", user, '0', '1')

	#import pdb;pdb.set_trace()
	#dill.dump(results, open("ranking/results_rankboost.pkl", "wb"))
	import pdb;pdb.set_trace()

def run_experiments():
	pkl_file = open('ranking/movieLen.pkl', 'r')

	movieLen = dill.load(pkl_file)
	parameter = {"max_iter_boosting":500, 'weak_classifier': 'stump_ranker'}
	
	results_manager = ResultsManager("ranking/result/database/rankboost.db")

	results = {}
	for index in range(len(movieLen.y_train.keys())):
		user = movieLen.y_train.keys()[index]
		if results_manager.is_finished("train", user, '0', '1'):
			continue
		print "test user: ", index

		ranker = RankBoost_ranking(**parameter)		
		ranker.fit(movieLen.X, movieLen.y_train[user])
		
		results['statistics_boosting'] = {}
		#results['statistics_boosting']["ranker"] = ranker
		results['statistics_boosting']["train_error"] = None
		results['statistics_boosting']["test_error"] = None
		for j in range( 1, min(101, ranker.actual_rounds_of_boosting+1) ):
			print "user: ", index, " iteration: ", j
			predictions = ranker.predict_train(iter = j)
			error = ranker.getRankingError(predictions, movieLen.y_train[user])
			results['statistics_boosting']["train_error"] = error
			if j == 1:
				predictions = ranker.predict(movieLen.X, iter = j)
			else:
				predictions = ranker.predict( iter = j)
			error = ranker.getRankingError(predictions, movieLen.y_test[user])
			results['statistics_boosting']["test_error"] = error 
			results_manager.store_results_boosting(results, j, "train", user, '0', '1')
		#dill.dump(results, open("ranking/result/results_rankboost_user_"+user+".pkl", "wb"))

	#import pdb;pdb.set_trace()
	#dill.dump(results, open("ranking/results_rankboost.pkl", "wb"))
	import pdb;pdb.set_trace()

class ranking_data_split(object):
	def __init__(self, X_train, X_test, y_train, y_test):
		self.X_train  = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test

class ranking_data(object):
	def __init__(self, X, y_train, y_test):
		self.X  = X
		self.y_train = y_train
		self.y_test = y_test

if __name__ == "__main__":
	#process()
	#process_remove_missing()
	#run_experiments()
	
	run_experiments_remove_missing()

