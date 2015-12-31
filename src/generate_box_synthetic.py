#this script is used to generate box-like synthetic dataset
#Specifically, all points within the box [0, 10]*[0, 10] are positive; otherwise, negative
#in order to sample more data which is close to boundary, I use the distribution beta(0.5, 0.5)

import random
import itertools
import matplotlib.pyplot as plt

PLOT = False

outputfile_data = 'box_syn.data'
outputfile_names = 'box_syn.names'

random.seed(12345) #initialize random number generator
N = 50 #number of positive/negative instances
max_range = 100

samples_x = {}
samples_x['positive'] = []
samples_x['negative'] = []


for i in range(N):
	samples_x['positive'] .append( max_range*random.betavariate(0.5, 0.5)   )
	if samples_x['positive'][-1]  != 0:
		if samples_x['positive'][-1] < 5 :
			samples_x['negative'].append( -samples_x['positive'][-1]  )
		else:
			samples_x['negative'].append( 2*max_range - samples_x['positive'][-1]  )

samples_y = {}
samples_y['positive'] = samples_x['positive'] 
samples_y['negative'] = samples_x['negative'] 
random.shuffle( samples_y['positive'] )
random.shuffle( samples_y['negative'] )

if PLOT:
	plt.figure()

samples = {}
samples['positive'] = []
samples['negative'] = []
for i in itertools.product(samples_x['positive'], samples_y['positive']):
	samples['positive'].append(i)
	if PLOT:
		plt.plot(samples['positive'][-1][0], samples['positive'][-1][1], 'r+')

for i in itertools.product(samples_x['negative'], samples_y['negative']):
	samples['negative'].append(i)
	if PLOT:
		plt.plot(samples['negative'][-1][0], samples['negative'][-1][1], 'bo')
for i in itertools.product(samples_x['positive'], samples_y['negative']):
	samples['negative'].append(i)
	if PLOT:
		plt.plot(samples['negative'][-1][0], samples['negative'][-1][1], 'bo')

for i in itertools.product(samples_x['negative'], samples_y['positive']):
	samples['negative'].append(i)
	if PLOT:
		plt.plot(samples['negative'][-1][0], samples['negative'][-1][1], 'bo')

random.shuffle(samples['positive'])
random.shuffle(samples['negative'])


if PLOT:
	plt.axis([-max_range, 3*max_range, -max_range, 3*max_range])
	plt.show()

bag_index_list = []
inst_index_list = []
bag_index = 0
for inst_index in range(len(samples['positive'])):
	if inst_index%5 == 0:
		bag_index +=1
	line = '%d, %d' % (bag_index, inst_index+1)
	if len(bag_index_list)==0 or bag_index!=bag_index_list[-1]:
		bag_index_list.append(bag_index)
	inst_index_list.append( inst_index+1 )
	for i in samples['positive'][inst_index]:
		line += ', %f' % i
	line += ',1\n'
 	
	with open(outputfile_data, 'a+') as f:
		f.write(line)
positive_instance_num = len(samples['positive'])

for inst_index in range( len(samples['negative']) ):
	if inst_index%5 == 0:
		bag_index +=1
	line = '%d, %d' % (bag_index, inst_index+1+positive_instance_num)
	if len(bag_index_list) ==0 or bag_index!=bag_index_list[-1]:
		bag_index_list.append(bag_index)
	inst_index_list.append(  inst_index+1+positive_instance_num )
	for i in samples['negative'][inst_index]:
		line += ', %f' % i
	line += ',0\n'
 	
	with open(outputfile_data, 'a+') as f:
		f.write(line)

line = '0, 1\n'
with open(outputfile_names, 'a+') as f:
	f.write(line)
line = 'bag_id: '+','.join( map(lambda x: str(x),   bag_index_list)  ) 
line +='\n'
with open(outputfile_names, 'a+') as f:
	f.write(line)
line = 'inst_id: '+','.join( map(lambda x: str(x),   inst_index_list)  ) 
line +='\n'
with open(outputfile_names, 'a+') as f:
	f.write(line)

line = 'f1: continuous\n'
with open(outputfile_names, 'a+') as f:
	f.write(line)
line = 'f2: continuous\n'
with open(outputfile_names, 'a+') as f:
	f.write(line)
line = 'label: 0, 1'
with open(outputfile_names, 'a+') as f:
	f.write(line)

