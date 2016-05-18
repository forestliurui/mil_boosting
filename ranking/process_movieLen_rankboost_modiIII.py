"""
This is used to process the movie len dataset
"""
import csv
import dill
import pickle
from RankBoost_modiIII_ranking_nondistributed import RankBoost_modiIII_ranking

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

def run_experiments():
	pkl_file = open('ranking/movieLen.pkl', 'r')

	movieLen = dill.load(pkl_file)
	parameter = {"max_iter_boosting":500}
	
	results = {}
	for index in range(len(movieLen.y_train.keys())):
		print "test user: ", index
		ranker = RankBoost_modiIII_ranking(**parameter)
		user = movieLen.y_train.keys()[index]
		ranker.fit(movieLen.X, movieLen.y_train[user])
		
		results[user] = {}
		results[user]["ranker"] = ranker
		results[user]["train_error"] = []
		results[user]["test_error"] = []
		for j in range(1, 100):
			print "user: ", index, " iteration: ", j
			predictions = ranker.predict_train(iter = j)
			error = ranker.getRankingError(predictions, movieLen.y_train[user])
			results[user]["train_error"].append(error)
			if j == 1:
				predictions = ranker.predict(movieLen.X, iter = j)
			else:
				predictions = ranker.predict( iter = j)
			error = ranker.getRankingError(predictions, movieLen.y_test[user])
			results[user]["test_error"].append(error)

	#import pdb;pdb.set_trace()
	dill.dump(results, open("ranking/results_modiIII.pkl", "wb"))
	import pdb;pdb.set_trace()
	

class ranking_data(object):
	def __init__(self, X, y_train, y_test):
		self.X  = X
		self.y_train = y_train
		self.y_test = y_test

if __name__ == "__main__":
	#process()
	run_experiments()

