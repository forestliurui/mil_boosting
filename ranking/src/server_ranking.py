#!/usr/bin/env python

"""
This is the server-side of the program to run experiments on general ranking problems (not bipartite).

To run it, use command like (from the directory that contains the 'ranking' directory)

python ranking/src/server_ranking.py results_root_dir ranker_name dataset_category

where results_root_dir is the directory where database containing results will be stored, 
ranker_name is one of {'rankboost','rankboost_modiII', 'rankboost_modiIII'},
dataset_category is one of {'LETOR', 'UCI','MovieLen'}

e.g. 

python ranking/src/server_ranking.py ranking/results/LETOR rankboost_modiIII LETOR

"""

import os
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import time
import random
import cherrypy
from cherrypy import expose, HTTPError
from threading import RLock
from collections import defaultdict
from random import shuffle
import numpy as np

import sys
if os.path.exists("/home/rui/MIL_Boost/MIL_Boosting/MIL_Boost/MIL_Boost/src/"):
	sys.path.append("/home/rui/MIL_Boost/MIL_Boosting/MIL_Boost/MIL_Boost/src/")
elif os.path.exists("/home/rui/MIL_Boosting/src/"):
	sys.path.append("/home/rui/MIL_Boosting/src/")
else:
	sys.path.append("/home/rui/MIL_boosting/src/")

#from progress import ProgressMonitor
from results import get_result_manager

PORT = 2117
DEFAULT_TASK_EXPIRE = 120 # Seconds
TEMPLATE = """
<html>
<head>
  <META HTTP-EQUIV="REFRESH" CONTENT="60">
  <title>%s</title>
  <style type="text/css">
    table.status {
      border-width: 0px;
      border-spacing: 0px;
      border-style: none;
      border-color: black;
      border-collapse: collapse;
      background-color: white;
      margin-left: auto;
      margin-right: auto;
    }
    table.status td {
        border-width: 1px;
        padding: 1px;
        border-style: solid;
        border-color: black;
        text-align: center;
    }
    table.summary {
      border-width: 0px;
      border-spacing: 0px;
      border-style: none;
      border-color: none;
      border-collapse: collapse;
      background-color: white;
      margin-left: auto;
      margin-right: auto;
    }
    table.summary td {
        border-width: 0px;
        padding: 3px;
        border-style: none;
        border-color: black;
        text-align: center;
        width: 50px;
    }
    td.tech { width: 50px; }
    td.done {
      background-color: green;
    }
    td.pending {
      background-color: yellow;
    }
    td.failed {
      background-color: red;
    }
    td.na {
      background-color: gray;
    }
  </style>
</head>
<body>
<h1>Time Remaining: %s</h1>
%s
</body>
</html>
"""

class UnfinishedException(Exception): pass

def plaintext(f):
    f._cp_config = {'response.headers.Content-Type': 'text/plain'}
    return f

class ExperimentServer(object):

    def __init__(self, tasks,  render,
                 task_expire=DEFAULT_TASK_EXPIRE):
        self.status_lock = RLock()
        self.tasks = tasks
       
        self.render = render
        self.task_expire = task_expire

        self.unfinished = set(self.tasks.items())

    def clean(self):
        with self.status_lock:
            self.unfinished = filter(lambda x: (not x[1].finished),
                                     self.unfinished)
            for key, task in self.unfinished:
                if (task.in_progress and
                    task.staleness() > self.task_expire):
                    task.quit()

    @expose
    def index(self):
        with self.status_lock:
            self.clean()
            return self.render(self.tasks)

    @plaintext
    @expose
    def request(self):
        with self.status_lock:
            self.clean()
            # Select a job to perform
	    rand_select = random.randint(1,20)
	    dataset_category_all = ['LETOR',  'MovieLen']
	    
	    if rand_select<=5:
	    	dataset_category = dataset_category_all[0]
	    else:
		dataset_category = dataset_category_all[1]	

            unfinished_raw = list(self.unfinished)
            shuffle(unfinished_raw)

	    unfinished_category = [x for x in unfinished_raw if x[1].dataset_category == dataset_category ]
	    unfinished_no_category = [x for x in unfinished_raw if x[1].dataset_category != dataset_category ]
	    
            unfinished_category = sorted(unfinished_category, key = lambda x: x[1].fold_index)
	    unfinished_no_category = sorted(unfinished_no_category, key = lambda x: x[1].fold_index)

	    unfinished_category = sorted(unfinished_category, key = lambda x: x[1].user_id)
	    unfinished_no_category = sorted(unfinished_no_category, key = lambda x: x[1].user_id)

	    unfinished = unfinished_category + unfinished_no_category
	   
            #candidates = sorted(unfinished, key=lambda x: x[1].priority())
            #candidates = sorted(unfinished, key = lambda x:x[1].fold_index)
            #candidates = sorted(candidates, key = lambda x:x[1].user_id)
	    #candidates = sorted(candidates, key = lambda x:x[1].in_progress)

	    candidates = sorted(unfinished, key = lambda x:x[1].in_progress)

	    if len(candidates) == 0:
                raise HTTPError(404)
            key, task = candidates.pop(0)
            task.ping()
        param ={}
	param['ranker'] = task.ranker_name
	param['dataset_category'] = key[0]
        arguments = {'key': key, 'param': param}
        #import pdb;pdb.set_trace()
        return yaml.dump(arguments, Dumper=Dumper)

    @plaintext
    @expose
    def update(self, key_yaml=None):
        try:
            key = yaml.load(key_yaml, Loader=Loader)
        except:
            raise HTTPError(400)
        with self.status_lock:
            if not key in self.tasks:
                raise HTTPError(404)
            task = self.tasks[key]
            if not task.finished:
                task.ping()
            else:
                # Someone else already finished
                raise HTTPError(410)
        return "OK"

    @plaintext
    @expose
    def quit(self, key_yaml=None):
        try:
            key = yaml.load(key_yaml, Loader=Loader)
        except:
            raise HTTPError(400)
        with self.status_lock:
            if not key in self.tasks:
                raise HTTPError(404)
            task = self.tasks[key]
            if not task.finished:
                task.quit()
            else:
                # Someone else already finished
                raise HTTPError(410)
        return "OK"

    @plaintext
    @expose
    def fail(self, key_yaml=None):
        try:
            key = yaml.load(key_yaml, Loader=Loader)
        except:
            raise HTTPError(400)
        with self.status_lock:
            if not key in self.tasks:
                raise HTTPError(404)
            task = self.tasks[key]
            if not task.finished:
                task.fail()
            else:
                # Someone else already finished
                raise HTTPError(410)
        return "OK"

    @plaintext
    @expose
    def submit(self, key_yaml=None, sub_yaml=None):
        try:
            key = yaml.load(key_yaml, Loader=Loader)
            submission = yaml.load(sub_yaml, Loader=Loader)
        except:
            import pdb;pdb.set_trace()
            raise HTTPError(400)
        with self.status_lock:
            if not key in self.tasks:
                raise HTTPError(404)
            task = self.tasks[key]
            if not task.finished:
                #task.store_results(submission)
                task.store_boosting_accum_results_directly_from_client(submission)
		task.finish()
        return "OK"

def time_remaining_estimate(tasks, alpha=0.1):
    to_go = float(len([task for task in tasks if not task.finished]))
    finish_times = sorted([task.finish_time for task in tasks if task.finished])
    ewma = 0.0
    for interarrival in np.diff(finish_times):
        ewma = alpha*interarrival + (1.0 - alpha)*ewma

    if ewma == 0:
        return '???'

    remaining = to_go * ewma
    if remaining >= 604800:
        return '%.1f weeks' % (remaining/604800)
    elif remaining >= 86400:
        return '%.1f days' % (remaining/86400)
    elif remaining >= 3600:
        return '%.1f hours' % (remaining/3600)
    elif remaining >= 60:
        return '%.1f minutes' % (remaining/60)
    else:
        return '%.1f seconds' % remaining

def render(tasks):
    # Get dimensions
    experiment_names = set()
    experiment_ids = set()
    parameter_ids = set()
    for key in tasks.keys():
	user_id = key[1]
	fold_id = key[2]
        dataset_category = key[0]
        experiment_names.add(dataset_category+'.'+str(user_id)+'.'+str(fold_id))
        experiment_ids.add(dataset_category+'.'+str(user_id)+'.'+str(fold_id))
        parameter_ids.add('1')

    experiment_names = sorted(experiment_names)
    experiment_title = ('Status: %s' % ', '.join(experiment_names))

    time_est = time_remaining_estimate(tasks.values())

    reindexed = defaultdict(list)
    for k, v in tasks.items():
	user_id = k[1]
	fold_id = k[2]
	dataset_category = k[0]
        reindexed[dataset_category+'.'+str(user_id)+'.'+str(fold_id), '1'].append(v)

    tasks = reindexed

    table = '<table class="status">'
    # Experiment header row
    table += '<tr><td style="border:0" rowspan="1"></td>'
    for parameter_id in parameter_ids:
        table += ('<td class="tech">%s</td>' % str(parameter_id))
    table += '</tr>\n'

    # Data rows
    for experiment_id in sorted(experiment_ids):
        table += ('<tr><td class="data">%s</td>' % str(experiment_id))
        for parameter_id in parameter_ids:
            key = (experiment_id, parameter_id)
            title = ('%s, %s' % tuple(map(str, key)))
            if key in tasks:
                table += ('<td style="padding: 0px;">%s</td>' % render_task_summary(tasks[key]))
            else:
                table += ('<td class="na" title="%s"></td>' % title)
        table += '</tr>\n'

    table += '</table>'
    return (TEMPLATE % (experiment_title, time_est, table))

def render_task_summary(tasks):
    n = float(len(tasks))
    failed = 0
    finished = 0
    in_progress = 0
    waiting = 0
    for task in tasks:
        if task.finished:
            finished += 1
        elif task.failed:
            failed += 1
        elif task.in_progress:
            in_progress += 1
        else:
            waiting += 1

    if n == finished:
        table = '<table class="summary"><tr>'
        table += ('<td class="done" title="Finished">D</td>')
        table += ('<td class="done" title="Finished">O</td>')
        table += ('<td class="done" title="Finished">N</td>')
        table += ('<td class="done" title="Finished">E</td>')
        table += '</tr></table>'
    else:
        table = '<table class="summary"><tr>'
        table += ('<td title="Waiting">%.2f%%</td>' % (100*waiting/n))
        table += ('<td class="failed" title="Failed">%.2f%%</td>' % (100*failed/n))
        table += ('<td class="pending" title="In Progress">%.2f%%</td>' % (100*in_progress/n))
        table += ('<td class="done" title="Finished">%.2f%%</td>' % (100*finished/n))
        table += '</tr></table>'
    return table

class Task(object):

    def __init__(self, dataset_category, user_id,
                 fold_index):
	
      	self.key = (dataset_category, user_id, fold_index) #for UCI dataset, treat user_id as a string, representing the name of individual dataset
        self.dataset_category = dataset_category
	self.user_id = user_id
        self.fold_index = fold_index
	self.train = str(user_id)+'.'+str(fold_index)+'.train'  #its format is like '2.3.train', meaning training part of the 3rd fold of user with id being 2
	self.test = str(user_id)+'.'+str(fold_index)+'.test' #its format is like '2.3.test', meaning test part of the 3rd fold of user with id being 2
        self.priority_adjustment = 0
	self.parameter_id_str = '0'
        self.parameter_set = '1'
        
	self.grounded = False

        self.last_checkin = None
        self.finished = False
        self.failed = False
        self.in_progress = False

        self.finish_time = None

    def ground(self, results_directory, ranker_name):
        self.results_directory = results_directory
 
 	self.ranker_name = ranker_name
	results_subdir = self.results_directory
        #results_subdir = os.path.join(self.results_directory,
        #                              self.experiment_name)
	#self.results_path = os.path.join(results_subdir,
        #                                 'UCI_'+ ranker_name +'.db')
	self.results_path = os.path.join(results_subdir,
                                        self.dataset_category+'_'+ ranker_name +'.db')


        self.results_manager = get_result_manager(self.results_path)
        if self.results_manager.is_finished(self.train, self.test):
            self.finish()

        self.grounded = True

    def get_predictions(self, bag_or_inst, train_or_test):
        if not self.grounded:
            raise Exception('Task not grounded!')

        if not self.finished:
            raise UnfinishedException()

        if train_or_test == 'train':
            test_set_labels = False
        elif train_or_test == 'test':
            test_set_labels = True
        else:
            raise ValueError('"%s" neither "train" nor "test"' %
                             train_or_test)

        if bag_or_inst.startswith('b'):
            return self.results_manager.get_bag_predictions(
                self.train, self.test, self.parameter_id_str,
                self.parameter_set, test_set_labels)
        elif bag_or_inst.startswith('i'):
            return self.results_manager.get_instance_predictions(
                self.train, self.test, self.parameter_id_str,
                self.parameter_set, test_set_labels)
        else:
            raise ValueError('"%s" neither "bag" nor "instance"'
                             % bag_or_inst)

    def get_statistic(self, statistic_name):
        if not self.finished:
            raise UnfinishedException()

        return self.results_manager.get_statistic(statistic_name, self.train,
                    self.test, self.parameter_id_str, self.parameter_set)

    def store_boosting_accum_results_directly_from_client(self, submission_boosting):
	for boosting_round in submission_boosting.keys():
		self.results_manager.store_results_boosting(submission_boosting[boosting_round], boosting_round, self.train, self.test, self.parameter_id_str, self.parameter_set)



    def store_results(self, submission):
        """Write results to disk."""
        if not self.grounded:
            raise Exception('Task not grounded!')

        self.results_manager.store_results(submission,
            self.train, self.test, self.parameter_id_str, self.parameter_set)

    def ping(self):
        if not self.finished:
            self.in_progress = True
            self.last_checkin = time.time()

    def quit(self):
        if not self.finished:
            self.in_progress = False
            self.last_checkin = None

    def fail(self):
        if not self.finished:
            self.failed = True
            self.in_progress = False

    def staleness(self):
        return time.time() - self.last_checkin

    def priority(self):
        return (10000*int(self.in_progress) + 1000*int(self.failed)
                + self.priority_adjustment)

    def finish(self):
        self.finished = True
        self.in_progress = False
        self.failed = False
        self.finish_time = time.time()

def start_experiment(results_root_dir, ranker_name, dataset_category):

    #ranker_name = 'rankboost'
    task_dict = load_config( results_root_dir, ranker_name, dataset_category)

    #import pdb;pdb.set_trace() 

    server = ExperimentServer(task_dict, render)
    cherrypy.config.update({'server.socket_port': PORT,
                            'server.socket_host': '0.0.0.0'})
    cherrypy.quickstart(server)

def load_config(results_directory, ranker_name, dataset_category):
    #dataset_category = "LETOR"
    dataset_categories = [dataset_category]
    tasks = {}
    for dataset_category in dataset_categories:
	if dataset_category == "MovieLen":
		#user_id_set = range(303) #respresent the user ID in MovieLen dataset. After preprocessing, there are 303 users left
		user_id_set = range(51) #to make sure the amount of users is same as LETOR
		fold_index_set = range(5) #movies associated for each user ID is partitioned into 5 folds for cross validation

                #user_id_set = [0]
                #fold_index_set = [2]

	elif dataset_category == "UCI": 
		fold_index_set= range(10) #each dataset is partitioned into 10 folds for cross validation
	
		#For UCI, user_id is a string representing each dataset
		user_id_set = ['Ionosphere',
    			'Haberman',
    			'Hepatitis',
    			'Horse_colic',
    			'volcanoes',
    			'breast_cancer_wdbc',
    			'wine',
    			'car_evaluation',
    			'Iris',
    			'Poker_hand',
    			'Credit_card_clients']
	
		#user_id_set = ['Ionosphere']

	elif dataset_category == "LETOR":
		#user_id_set = range(21) 
		#user_id_set = range(40) 
		user_id_set = range(51) #for LETOR, the user_id is actually query ID

		fold_index_set = range(5) #URLs associated with each query is partitioned into 5 folds for cross-validation
	else:
		raise error("Do NOT support data category %s" %dataset_category)

	
	for user_id in user_id_set:
		for fold_index in fold_index_set:
			task_key = (dataset_category, user_id, fold_index)
			task = Task(dataset_category, user_id, fold_index)
			
			task.ground(results_directory, ranker_name)
			tasks[task_key]= task
    return tasks

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog configfile resultsdir")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
	
    num_args = 3

    if len(args) != num_args:
        parser.print_help()
        exit()
    start_experiment(*args, **options)
