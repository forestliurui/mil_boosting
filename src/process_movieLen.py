"""
This is used to process the movie len dataset
"""
import csv
import pickle
from RankBoost_modiII_ranking_nondistributed import RankBoost_modiII_ranking

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
		y_train[user] = y[user][0:len(y[user])/2]
		y_test[user] =  y[user][len(y[user])/2:]
	movieLen = ranking_data(X, y_train, y_test)	
	output = open('movieLen.pkl', 'wb')
	pickle.dump(movieLen, output)
	import pdb;pdb.set_trace()

def run_experiments():

	movieLen = pickle.load('../movieLen/movieLen.pkl')
	ranker = RankBoost_modiII_ranking()
	ranker.fit(movieLen.X, movieLen.y_train)

	predictions = ranker.predict_train(iter = 5)
	error = ranker.getRankingError(predictions, movieLen.y_test)

	

class ranking_data(object):
	def __init__(self, X, y_train, y_test):
		self.X  = X
		self.y_train = y_train
		self.y_test = y_test

if __name__ == "__main__":
	#process()
	run_experiments()

