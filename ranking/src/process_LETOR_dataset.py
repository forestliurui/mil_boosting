"""
This script is used to process the raw data of Microsoft's Learning to Rank (LETOR) 10K dataset and partition it into appropriate training and test sets
"""

import unittest
import csv
import dill
import numpy as np

from process_movieLen_dataset import getPartition, getCriticalPair
from process_movieLen_dataset import movieLenData as RankingData

def process(filename):
	"""
	This function is trying to partition the raw dataset, and store them into .pkl files
	"""

	#filename = "movieLen.csv"
	

	data_map = getMap(filename) #data_map {query_id(int): { doc_id(int): {"feature":feature_list(float list); "score": rel_score(int) } } } 
	

	num_folds = 5



	output_index = 0

	for index_query in range(len(data_map)): #take turn to choose user for critical pairs
		print "user index: ", index_query

		qid = int(data_map.keys()[index_query])
		X, y = getXy(data_map, qid)

		X_train, y_train, X_test, y_test = getPartition(X, y, num_folds)

		p_train= []
		p_test = []
		for temp in y_train:
			p_train.append( getCriticalPair(temp) )
		for temp in y_test:
			p_test.append( getCriticalPair(temp) )
		#import pdb;pdb.set_trace()
		result = RankingData(X_train, p_train, X_test, p_test)
		
		#import pdb;pdb.set_trace()
		dill.dump(result, open('ranking/LETOR/LETOR_query_'+str(output_index)+'.pkl', 'wb'))
		output_index += 1

def getMap(filename):
	data_map = {}

	with open(filename, "rb") as f:
		reader = csv.reader(f, delimiter = ",")
		doc_id = 0
		for row in reader:
			qid = int(row[1].split(":")[1])
			score = int(row[0])
			
			if qid not in data_map:
				data_map[qid] = {}
			data_map[qid][doc_id] ={}

			data_map[qid][doc_id]["score"] = score
			data_map[qid][doc_id]["feature"] = [float(x) for x in row[2:138]]
		
			doc_id +=1 
	return data_map

def getXy(data_map, qid):
	X={}
	y ={}
  	for doc_id in data_map[qid].keys():
		X[doc_id] =	data_map[qid][doc_id]["feature"]
		y[doc_id] = data_map[qid][doc_id]["score"]
	return X, y


if __name__ == "__main__":
	#filename= "ranking/LETOR/LETOR_doc_count500.csv"
	filename= "ranking/LETOR/LETOR_doc_upperbound_600_lowerbound_400.csv"
	process(filename)
