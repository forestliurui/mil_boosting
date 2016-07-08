"""
This script is used to process the raw data of movieLen and partition it into appropriate training and test sets

has passed tests, should be correct
"""

import unittest
import csv
import dill

def process(filename):
	"""
	This function is trying to partition the raw dataset, and store them into .pkl files
	"""

	#filename = "movieLen.csv"
	

	user_map, movie_map = getMap(filename)
	
	num_user = len(user_map)
	num_movie = len(movie_map)

	num_folds = 5

	for index_user in range(num_user): #take turn to choose user for critical pairs
		print "user index: ", index_user
		if index_user >50: #try not to get too many subdatasets
			break
		X, y = getXy(user_map, movie_map, index_user)

		X_train, y_train, X_test, y_test = getPartition(X, y, num_folds)

		p_train= []
		p_test = []
		for temp in y_train:
			p_train.append( getCriticalPair(temp) )
		for temp in y_test:
			p_test.append( getCriticalPair(temp) )
	
		movieLen = movieLenData(X_train, p_train, X_test, p_test)
		
		#import pdb;pdb.set_trace()
		dill.dump(movieLen, open('ranking/movieLen/movieLen_user'+str(index_user)+'.pkl', 'wb'))


def process_getStat(filename):
	"""
	This function is trying to get stats like sizeOfMovies and sizeOfUsers for each partition, and store them into a csv file.
	"""

	#filename = "movieLen.csv"
	
	output_filename = "ranking/stat_movieLen.csv"

	line = "indexOfDataset, sizeOfMovies, sizeOfUsers\n"

	with open(output_filename, 'a+') as f:
		f.write(line) 

	user_map, movie_map = getMap(filename)
	
	num_user = len(user_map)
	num_movie = len(movie_map)

	num_folds = 5
	stat = {'sizeOfMovies':[], 'sizeOfUsers': []}
	for index_user in range(num_user): #take turn to choose user for critical pairs
		print "user index: ", index_user
		#if index_user >50: #try not to get too many subdatasets
		#	break
		X, y = getXy(user_map, movie_map, index_user)

		X_train, y_train, X_test, y_test = getPartition(X, y, num_folds)

		p_train= []
		p_test = []
		for temp in y_train:
			p_train.append( getCriticalPair(temp) )
		for temp in y_test:
			p_test.append( getCriticalPair(temp) )
		stat['sizeOfMovies'].append(len(X))
		stat['sizeOfUsers'].append( len( X.values()[0] ) )
		#movieLen = movieLenData(X_train, p_train, X_test, p_test)
		
		#import pdb;pdb.set_trace()
		#dill.dump(movieLen, open('ranking/movieLen/movieLen_user'+str(index_user)+'.pkl', 'wb'))

		line = str(index_user)+","
		line += str(stat['sizeOfMovies'][-1])
		line += ","
		line += str(stat['sizeOfUsers'][-1])
		line += "\n"
		with open(output_filename, 'a+') as f:
			f.write(line)

class movieLenData(object):
	def __init__(self, X_train, p_train, X_test, p_test):
		self.X_train = X_train
		self.p_train = p_train
		
		self.X_test = X_test
		self.p_test = p_test



def getCriticalPair(y):
	"""
	y is a dictionary like {movie_index1: 2, movie_index2: 1} for the index_user

	return a list of tuples like [(movie_index1, movie_index2), (movie_index3, movie_index1) ]. For each tuple, the first item is assumed to be ranked higher than the second according to y
	"""
	
	movieList = y.keys()
	result = []
	for i in movieList:
		for j in movieList:
			if y[i] > y[j]:
				result.append((i, j))
	return result


def getPartition(X, y, num_folds):
	"""
	X as a dictionary like {movie_index1: list, movie_index2: list}, the list contains the rating of a specific movie from all other users except for the one in y. -1 is used to indicate missing value
	y as a dictionary like {movie_index1: 2, movie_index2: 1} for the index_user
	
	num_folds is the num of folds to be partitioned into

	return X_train, y_train, X_test, y_test
	
	X_train is a list of dic. The dic is like X. 
	y_train is a list of dic. The dic is like y
	...
	Each fold corresponds to one item in lists
	"""

	num_movie = len(X)

	X_train = []
	y_train = []
	X_test = []
	y_test = []

	for fold in range(num_folds):
		ub = (fold+1)*(num_movie/num_folds) 
		lb = fold*(num_movie/num_folds)

		#import pdb;pdb.set_trace()
		X_train.append({ key: list(X[key]) for key in X.keys() if key not  in range(lb, ub) })
		X_test.append({ key: list(X[key]) for key in X.keys() if key  in range(lb, ub) })
		y_train.append( { key: y[key] for key in y.keys() if key not in range(lb, ub) } )
		y_test.append( { key: y[key] for key in y.keys() if key  in range(lb, ub) } )
	return X_train, y_train, X_test, y_test

def getMap(filename):
	"""
	file_name is a string respresenting a .csv file, each role of which is like user_id, movie_id, rating

	return user_map and movie_map

	user_map is dictionary to store the movie rating for each user, like {user1:{movie1: 2, movie2:3, movie3: 1}, user2: {movie1: 1, movie2:2, movie3: 1}  }
	movie_map is dictionary to store the movie rating for each movie, like {movie1:{user1: 2, user2:3, user3: 1}, movie2: {user1: 1, user2:2, user3: 1}  }
	"""

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
	return user_map, movie_map

def getXy(user_map, movie_map, index_user):
	"""
	user_map is dictionary to store the movie rating for each user, like {user1:{movie1: 2, movie2:3, movie3: 1}, user2: {movie1: 1, movie2:2, movie3: 1}  }
	movie_map is dictionary to store the movie rating for each movie, like {movie1:{user1: 2, user2:3, user3: 1}, movie2: {user1: 1, user2:2, user3: 1}  }

	index_user is the index of user in user_map.keys() which will be used to construct critical pairs

	return X as a dictionary like {movie_index1: list, movie_index2: list} , the list contains the rating of a specific movie from all other users except for the one in y. -1 is used to indicate missing value
	      y as dictionary like {movie_index1: 2, movie_index2: 1} for the index_user

	Processing is applied to make sure there is no missing value in y
	"""	
	X = {}
	y = {}
	id_user = user_map.keys()[index_user]

	index_m_actual = -1
	for index_m in range(len(movie_map)):
		id_m = movie_map.keys()[index_m]

		if id_user in movie_map[id_m]: #there is a rating for id_user/index_user for movie id_m
			index_m_actual += 1
			X[index_m_actual] = []
		
			for index_u in range(len(user_map)):
				id_u = user_map.keys()[index_u]
				if index_u!=index_user:
					if id_u in movie_map[id_m]:
						X[index_m_actual].append(  movie_map[id_m][id_u] )
					else:
						X[index_m_actual].append(-1)
				else:
					y[index_m_actual] = movie_map[id_m][id_u]
			
	return X, y



class TestProcess(unittest.TestCase):
	def DA_test_getMap(self):
		file_name = "ranking/movieLen/test_process_movieLen.csv"
		user_map, movie_map = getMap(file_name)		
		print "user map"
		print user_map

		print "movie map"
		print movie_map
		
	def DA_test_getXy(self):
		file_name = "ranking/movieLen/test_process_movieLen.csv"
		user_map, movie_map = getMap(file_name)	
		print user_map.keys()[1]
		X, y = getXy(user_map, movie_map, 1)
		print X
		print y

	def DA_test_getPartition(self):
		file_name = "ranking/movieLen/test_process_movieLen.csv"
		user_map, movie_map = getMap(file_name)	
		print user_map.keys()[1]
		X, y = getXy(user_map, movie_map, 1)

		X_train, y_train, X_test, y_test = getPartition(X, y, 2)
		import pdb;pdb.set_trace()

	def test_getCriticalPair(self):
		file_name = "ranking/movieLen/test_process_movieLen.csv"
		user_map, movie_map = getMap(file_name)	
		print user_map.keys()[1]
		X, y = getXy(user_map, movie_map, 1)

		X_train, y_train, X_test, y_test = getPartition(X, y, 2)

		p = getCriticalPair(y_train[0])
		import pdb;pdb.set_trace()

if __name__ == "__main__":
	#unittest.main()
	#process("ranking/movieLen/movieLen.csv")
	process_getStat("ranking/movieLen/movieLen.csv")


