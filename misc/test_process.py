from multiprocessing import Process
import numpy as np

def factorial_sum(num):
    res = 0
    for i in range(1, num+1):
        res += i
    return res

def factorial_sum(nums, index_list):
    res = 0
    for index in index_list:
        res += factorial_sum(nums[index])
    return res

max_num = 10**4
size = 10**6
batch_size = 10**5
nums = np.random.choice(max_num, size)

indices = range(size)
for batch_index in range(int(size/batch_size)):
    batch = indices[batch_size*batch_index: batch_size*(batch_index+1)]
    
    


