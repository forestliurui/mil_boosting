from multiprocessing import Process, Queue, Array, Manager
import numpy as np
import time

class Res_float(object):
    def __init__(self, val):
        self.val = val
    def printVal(self):
        print(self.val)
   

def factorial_sum(num):
    res = 0
    for i in range(1, num+1):
        res += i
    return res

def factorial_batch(nums, index_list):
    res = 0
    for index in index_list:
        res += factorial_sum(nums[index])
    return res


def factorial_queue(nums, index_list, q):
    res = []
    for index in index_list:
        res.append(factorial_sum(nums[index]) )
    print("before put into queue")
    q.put(res)
    print("finish put into queue")


def factorial_share(nums, index_list, array):
    res = []
    for index in index_list:
        array[index] = factorial_sum(nums[index]) 

def factorial_server(nums, index_list, dictionary):
    for index in index_list:
        #dictionary[index] = factorial_sum(nums[index])
        dictionary[index] = Res_float(factorial_sum(nums[index]))

max_num = 10**4
size = 10**6
batch_size = size/1
array = Array('i', [0]*size)
nums = np.random.choice(max_num, size)
#print(nums)

import pdb;pdb.set_trace()
start_time = time.time()
processes = {}
manager = Manager()
d = manager.dict()
q = Queue()
indices = range(size)

if size == batch_size:
    d = {}
    factorial_server(nums, range(batch_size), d)
else:
  for batch_index in range(int(size/batch_size)):
    batch = indices[batch_size*batch_index: batch_size*(batch_index+1)]
    processes[batch_index] = Process(target = factorial_server, args = (nums, batch, d,))
    #processes[batch_index] = Process(target = factorial_share, args = (nums, batch, array,)) 
    #processes[batch_index] = Process(target = factorial_queue, args = (nums, batch, q,))
    #processes[batch_index] = Process(target = factorial_batch, args = (nums, batch,))
    print("start process for batch: %d"%batch_index)
    processes[batch_index].start()
  #print(q.get())
  for batch_index in range(int(size/batch_size)):
    print("join on process for batch: %d"%batch_index)
    processes[batch_index].join()
#print(q.get())
#print("begin to fetch result")
#for batch_index in range(int(size/batch_size)):
#    print(q.get())
#array_res = array[:]
dict_res = dict(d)
end_time = time.time()
print("finish all processes: %.3f"%(end_time - start_time))
import pdb;pdb.set_trace()

