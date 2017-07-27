"""
preprocess LETOR dataset, originally stored in three different files: train.txt, test.txt and vali.txt

merge them into one csv files and only store those queries whose number of associated docs is greater than a threshold (e.g 500)

Format conversion:
each row in train/test/vali.txt file: (like feature_id:feature_value)
score qid:131 1:3 2:1 ...

into

each row in csv: (remove feature_id)
score, qid:131,3,1,...

"""

import csv



def check_file(filename, qid_count_doc):

	#qid_count_doc = {}

	with open(filename, "rb") as f:
		reader = csv.reader(f, delimiter = " ")
		for row in reader:
			#import pdb;pdb.set_trace()
			score = row[0]
			qid = int(row[1].split(":")[1])
			if qid not in qid_count_doc:
				qid_count_doc[qid] = 0
		
 			qid_count_doc[qid] += 1
	#import pdb;pdb.set_trace()
	#return qid_count_doc

def merge_file(input_filename, output_filename, count, doc_count_upperbound = None, doc_count_lowerbound = None):
	
	if doc_count_upperbound is None:
		doc_count_upperbound =  max(count.values()) +1
	if doc_count_lowerbound is None:
		doc_count_lowerbound = -1

	#doc_count_threshold = 500
	num_feature = 136
	with open(input_filename, "rb") as f:
		reader = csv.reader(f, delimiter = " ")
		for row in reader:
			qid = int(row[1].split(":")[1])
			if count[qid] >= doc_count_lowerbound and count[qid] <= doc_count_upperbound:
				with open(output_filename, "a+") as f_out:
					row_out = ",".join(row[0:2])
					row_out += ","
					row_out += (",".join([ str(float(x.split(":")[1]) ) for x in  row[2:num_feature+2]  ]) ) #plus the first col for relevance score and the second col for qid
					row_out += "\n"
					f_out.write(row_out)

if __name__ == "__main__":
	
	count = {} #map from query id to number of docs associated with this query
	print "checking test.txt"
	filename = "test.txt"
	check_file(filename, count)

	print "checking train.txt"
	filename = "train.txt"
	check_file(filename, count)

	print "checking vali.txt"
	filename = "vali.txt"
	check_file(filename, count)
	#import pdb;pdb.set_trace()
	doc_count_lowerbound = 400 #the min number of doc for querys that are to be stored at .csv
	doc_count_upperbound = 600

	output_filename = "LETOR_doc_upperbound_"+str(doc_count_upperbound)+"_lowerbound_"+str(doc_count_lowerbound)+".csv"
	merge_file( "test.txt", output_filename, count, doc_count_upperbound, doc_count_lowerbound )
	merge_file( "train.txt", output_filename, count, doc_count_upperbound, doc_count_lowerbound)
	merge_file( "vali.txt", output_filename, count, doc_count_upperbound, doc_count_lowerbound)
	#import pdb;pdb.set_trace()
	

