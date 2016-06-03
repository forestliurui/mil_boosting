import csv

filename = "LETOR_raw.csv"

bag_id = 1
bag_ids =[]
filename_output = "LETOR.csv"

feature_size = None

with open(filename, "rb") as f:
	reader = csv.reader(f, delimiter = ",")
	for row in reader:
		#import pdb;pdb.set_trace()
		
		if feature_size is None:
			feature_size = len(row) - 1		
		bag_ids.append(str(bag_id))

		row_out = [str(bag_id)]*2 + list(row[1:])+[row[0]]
		if row_out[-1] == "1":
			row_out[-1] = "1"
		else:
			row_out[-1] = "0"
		
		num_missing = 0
		for index in range(len(row_out)):
			if row_out[index] == "?":
				num_missing  += 1
				row_out[index] = "-1"		
		if num_missing < feature_size/2:
			with open(filename_output, "a+") as f_out:
			
				f_out.write(",".join(row_out)+"\n")
		bag_id += 1

name_file_output = "Hepatitis.names"
with open(name_file_output, "w") as f_n:
	f_n.write("0,1\n")
	f_n.write("bag_id:"+",".join(bag_ids)+"\n")
	f_n.write("inst_id:"+",".join(bag_ids)+"\n")
	
	for j in range(feature_size):
		f_n.write("f"+str(j)+": continuous\n")



