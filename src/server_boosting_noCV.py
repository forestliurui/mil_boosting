#!/usr/bin/env python
import os
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import time
import random
import data
import cherrypy
from cherrypy import expose, HTTPError
from threading import RLock
from collections import defaultdict
from random import shuffle
import numpy as np
from math import log, exp
#import editdistance
import threading
import Queue
import sqlite3
import string
import gc

import evaluation_metric
from folds import FoldConfiguration
from progress import ProgressMonitor
from results import get_result_manager

INNER_CROSS_VALIDATION = False

PORT = 2118
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

    def __init__(self, tasks, params, render,
                 task_expire=DEFAULT_TASK_EXPIRE):
        self.status_lock = RLock()
        self.tasks = tasks
        self.params = params
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
	gc.collect()
        with self.status_lock:
            self.clean()
            # Select a job to perform
            unfinished = list(self.unfinished)
            shuffle(unfinished)
            candidates = sorted(unfinished, key=lambda x: x[1].priority())
            if len(candidates) == 0:
                raise HTTPError(404)
            key, task = candidates.pop(0)
            task.ping()

    	

        (experiment_name, experiment_id,
         train, test, parameter_id, parameter_set) = key
        parameters = self.params[experiment_id].get_parameters(
            parameter_id=parameter_id, parameter_set=parameter_set)
	#import pdb;pdb.set_trace()
	for k, v in parameters.items():
        	print '\t%s: %s' % (k, str(v))

        arguments = {'key': key, 'parameters': parameters}
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
        experiment_names.add(key[0])
        experiment_ids.add(key[1])
        parameter_ids.add(key[4])

    experiment_names = sorted(experiment_names)
    experiment_title = ('Status: %s' % ', '.join(experiment_names))

    time_est = time_remaining_estimate(tasks.values())

    reindexed = defaultdict(list)
    for k, v in tasks.items():
        reindexed[k[1], k[4]].append(v)

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

class ParameterConfiguration(object):

    def __init__(self, results_directory, experiment_name,
                 experiment_id, experiment_format,
                 parameter_key, parameter_format, parameter_configuration):
        self.results_directory = results_directory
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.experiment_format = experiment_format
        self.parameter_key = parameter_key
        self.parameter_format = parameter_format
        self.parameter_configuration = parameter_configuration

        self.param_directory = os.path.join(results_directory, experiment_name)
        if not os.path.exists(self.param_directory):
            os.mkdir(self.param_directory)

        self.settings = None
        self.param_dict = {}

    def _parameter_path(self, parameter_id):
        key = self.experiment_id
        key += parameter_id
        fmt = list(self.experiment_format)
        fmt += self.parameter_format
        format_str = '_'.join(fmt)
        filename = (format_str % key)
        filename += '.params'
        return os.path.join(self.param_directory, filename)

    def get_settings(self):
        if self.settings is None:
            self.settings = []

            for parameters in self.parameter_configuration:
                parameters = dict(**parameters)
                p_search = parameters.pop('search')
                search_type = p_search['type']
                if search_type != 'random':
                    raise ValueError('Unknown search type ""' % search_type)

                parameter_id = tuple(parameters[k] for k in self.parameter_key)
                param_path = self._parameter_path(parameter_id)

                # Load any parameters that already exist
                if os.path.exists(param_path):
                    with open(param_path, 'r') as f:
                        param_list = yaml.load(f)
                else:
                    param_list = []
		#import pdb;pdb.set_trace()
                # Add additional parameter sets as needed
                for i in range(p_search['n'] - len(param_list)):
		    
                    params = {}
                    for param, constraints in parameters.items():
                        if type(constraints) == list:
                            if (type(constraints[0]) == str
                                and constraints[0][0] == 'e'):
                                params[param] = 10**random.uniform(
                                                        *[float(c[1:])
                                                        for c in constraints])
                            else:
                                params[param] = random.uniform(*map(float, constraints))
                        else:
                            params[param] = constraints
                    #import pdb;pdb.set_trace()
                    param_list.append(params)

                with open(param_path, 'w+') as f:
                    f.write(yaml.dump(param_list, Dumper=Dumper))

                self.param_dict[parameter_id] = param_list
                for i in range(len(param_list)):
                    self.settings.append({'parameter_id': parameter_id,
                                          'parameter_set': i})
	#import pdb;pdb.set_trace()
        return self.settings

    def get_parameters(self, parameter_id=None, parameter_set=None):
        self.get_settings() # This must be called first
        return self.param_dict[parameter_id][parameter_set]

    def get_parameter_sets(self):
        sets = defaultdict(list)
        for s in self.get_settings():
            sets[s['parameter_id']].append(s['parameter_set'])
        return list(sets.items())

class Task(object):

    def __init__(self, experiment_name, experiment_id,
                 train, test,
                 parameter_id, parameter_set):
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.train = train
        self.test = test
        self.parameter_id = parameter_id
        self.parameter_set = parameter_set

        self.priority_adjustment = 0

        self.grounded = False

        self.last_checkin = None
        self.finished = False
        self.failed = False
        self.in_progress = False

        self.finish_time = None

    def ground(self, results_directory,
               experiment_format, parameter_format):
        self.results_directory = results_directory
        self.experiment_format = experiment_format
        self.parameter_format = parameter_format

        self.parameter_id_str = ('_'.join(parameter_format)
                                 % self.parameter_id)
        self.experiment_id_str = ('_'.join(experiment_format)
                                  % self.experiment_id)

        results_subdir = os.path.join(self.results_directory,
                                      self.experiment_name)
        self.results_path = os.path.join(results_subdir,
                                         self.experiment_id_str + '.db')

        self.results_manager = get_result_manager(self.results_path)
        if self.results_manager.is_finished(self.train, self.test,
                self.parameter_id_str, self.parameter_set):
            self.finish()

        self.grounded = True

    def get_predictions(self, bag_or_inst, train_or_test):
        if not self.grounded:
            raise Exception('Task not grounded!')
        #import pdb;pdb.set_trace()
        print 'try to get predictions'
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
    def get_prediction_true_matrix(self, bag_or_inst, train_or_test ):
	if not self.grounded:
            raise Exception('Task not grounded!')
        #import pdb;pdb.set_trace()
        print 'try to get predictions'
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
                
   		prediction_inst=self.get_predictions(bag_or_inst, train_or_test) #this is dictionary, with key as inst_id and value as list of scores for each label
        	if train_or_test == 'train':
			data_test_train=data.get_dataset(self.train)
		else:
			data_test_train=data.get_dataset(self.test)

                #test.instance_ids
        	prediction_matrix=reduce( lambda x, y :np.vstack((x, y)), [prediction_inst[x[1]]  for x in data_test_train.instance_ids   ]  )
        	label_matrix=data_test_train.instance_labels
	elif bag_or_inst.startswith('i'):
		raise ValueError('get_prediction_true_matrix for instance not implemented')
	else:
            raise ValueError('"%s" neither "bag" nor "instance"'
                             % bag_or_inst)
	return prediction_matrix, label_matrix

    def get_statistic(self, statistic_name):
        if not self.finished:
            raise UnfinishedException()

        return self.results_manager.get_statistic(statistic_name, self.train,
                    self.test, self.parameter_id_str, self.parameter_set)

    def store_results(self, submission):
        """Write results to disk."""
        if not self.grounded:
            raise Exception('Task not grounded!')

        self.results_manager.store_results(submission,
            self.train, self.test, self.parameter_id_str, self.parameter_set)

    def store_boosting_raw_results(self, boosting_rounds):

	submission_boosting={}
	submission_boosting['raw']={} #without accumulating results from previous rounds
	submission_boosting['raw']['instance_predictions']={}
	submission_boosting['raw']['bag_predictions']={}

	submission_boosting['raw']['instance_predictions']['test'] = self.get_predictions('instance','test')
	submission_boosting['raw']['instance_predictions']['train']= self.get_predictions('instance','train')
	submission_boosting['raw']['bag_predictions']['train'] = self.get_predictions('bag','train')
	submission_boosting['raw']['bag_predictions']['test'] = self.get_predictions('bag','test')

	
	self.results_manager.store_results_boosting(submission_boosting, boosting_rounds, self.train, self.test, self.parameter_id_str, self.parameter_set)

    def store_boosting_accum_results_directly_from_client(self, submission_boosting):
	for boosting_round in submission_boosting.keys():
		self.results_manager.store_results_boosting(submission_boosting[boosting_round], boosting_round, self.train, self.test, self.parameter_id_str, self.parameter_set)

	


    def store_boosting_accum_results(self, prediction_matrix_test_accumulated, boosting_rounds):
        #this is used to store the prediction results for test dataset's each label from boosting
        #bag_predictions = np.hstack((bag_predictions0[:,np.newaxis], bag_predictions1[:,np.newaxis],bag_predictions2[:,np.newaxis],bag_predictions3[:,np.newaxis],bag_predictions4[:,np.newaxis]  ))
        data_test=data.get_dataset(self.test)
	submission_boosting={}
        submission_boosting['instance_predictions']={}
        submission_boosting['instance_predictions']['test']={}
        for i, y in zip(data_test.instance_ids, map(tuple,prediction_matrix_test_accumulated)):
        	submission_boosting['instance_predictions']['test'][i] = map(float,y)
        
	eval_task=evaluation_metric.EvaluationMetric(self, prediction_matrix_test_accumulated)
    	eval_task.avg_prec()
	submission_boosting['statistics_boosting']={}
	submission_boosting['statistics_boosting']['hamm_loss']=eval_task.hamm_loss()
	submission_boosting['statistics_boosting']['one_error']=eval_task.one_error()
	submission_boosting['statistics_boosting']['coverage']=eval_task.coverage()
	submission_boosting['statistics_boosting']['average_precision']=eval_task.avg_prec()
	
	try:
            from sklearn.metrics import roc_auc_score as score
        except:
            from sklearn.metrics import auc_score as score
        scorename = 'AUC'

	AUC_list=[]
	for ii in range(5):
	    AUC_list.append(score(data_test.instance_labels[:,ii], prediction_matrix_test_accumulated[:,ii])) 
	    AUC_mean=np.mean(AUC_list)
	    submission_boosting['statistics_boosting'][scorename]=AUC_mean
	    print ('Test dataset: %s Boosting Rounds: %d  Average %s Score: %f'
                   % (self.test, boosting_rounds, scorename, AUC_mean ))
	    print( 'Its corresponding Individual %s Score: ' %scorename   +','.join(map(str, AUC_list))   )

	self.results_manager.store_results_boosting(submission_boosting, boosting_rounds, self.train, self.test, self.parameter_id_str, self.parameter_set)

	



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

class ExperimentConfiguration(object):

    def __init__(self, experiment_name, experiment_id,
                 fold_config, param_config, resampling_constructor):
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.fold_config = fold_config
        self.param_config = param_config
        self.resampling_constructor = resampling_constructor
        self.settings = None

    def get_settings(self):
        if self.settings is None:
            self.settings = []
            for train, test in self.fold_config.get_all_train_and_test():
                resampling_config = self.resampling_constructor(train)
                for resampled in resampling_config.get_all_resampled():
                    for pconfig in self.param_config.get_settings():
                        setting = {'train': resampled,
                                   'test': test}
                        setting.update(pconfig)
                        self.settings.append(setting)

        return self.settings

    def get_key(self, **setting):
        key = (self.experiment_name, self.experiment_id,
               setting['train'], setting['test'],
               setting['parameter_id'], setting['parameter_set'])
        return key

    def get_task(self, **setting):
        key = self.get_key(**setting)
        return Task(*key)

def start_experiment_noCV(configuration_file, results_root_dir):
    task_dict, param_dict = load_config(configuration_file, results_root_dir)

    server = ExperimentServer(task_dict, param_dict, render)
    cherrypy.config.update({'server.socket_port': PORT,
                            'server.socket_host': '0.0.0.0','response.timeout': 3000})
    cherrypy.quickstart(server)

def start_experiment(configuration_file, results_root_dir):
    task_dict, param_dict = load_config(configuration_file, results_root_dir)
    shared_variables={}  
    queue_tasks_to_be_run=Queue.Queue()
    #queue_tasks_finished=Queue.Queue() 
    shared_variables['to_be_run']=queue_tasks_to_be_run  #the queue containing the tasks to be run
    shared_variables['to_be_run'].put(dict())
    shared_variables['finished_set']={}    #the dictionary containing the finished tasks by client 
    #queues['finished']=queue_tasks_finished
    shared_variables['condition_lock']=threading.Condition() #condition variable used to synchronize server and controller
    shared_variables['bag_weights']={}
    shared_variables['inst_weights']={}
    

    server = ExperimentServer(task_dict, param_dict, render, shared_variables)
    cherrypy.config.update({'server.socket_port': PORT,
                            'server.socket_host': '0.0.0.0','response.timeout': 3000})
    
    #def wrapper_server(task, args):
    #     cherrp

    thread_server=threading.Thread(target=cherrypy.quickstart, args=(server,))
    thread_server.start()    
    #cherrypy.quickstart(server)
    experiment_dispatcher(configuration_file, task_dict, shared_variables, server)
    
def experiment_dispatcher(configuration_file, task_dict, shared_variables, server):
    print 'Loading configuration for experiments...'
    with open(configuration_file, 'r') as f:
        configuration = yaml.load(f)    

    num_dataset = len(configuration['experiments'])
    for index_dataset in range(num_dataset):
	
    	dataset_name=configuration['experiments'][index_dataset]['dataset']
	thread_dataset=threading.Thread(target=server_experiment, args=(dataset_name, configuration_file, task_dict, shared_variables, server))
    	thread_dataset.start()   


def server_experiment(dataset_name, configuration_file, task_dict, shared_variables, server):
    
    print 'Loading configuration for experiments...'
    with open(configuration_file, 'r') as f:
        configuration = yaml.load(f)    

    #dataset_name=configuration['experiments'][0]['dataset']

    outer_folds, inner_folds=configuration['folds']
    
    booster_name = configuration['booster_params'][0]['booster_name']
    max_iter_boosting = configuration['booster_params'][0]['max_iter']
    #shared_variables['bag_weights'][dataset_name]={}
    shared_variables['inst_weights'][dataset_name]={}


    auxiliary_structure ={}  #auxiliary structure to support parallelization
    auxiliary_structure['task_dict']=task_dict
    auxiliary_structure['shared_variables']=shared_variables
    auxiliary_structure['server']=server



    for set_index_boosting in range(outer_folds):
	train_dataset_name=string.replace( '%s.fold_%4d_of_%4d.train' % (dataset_name,set_index_boosting, outer_folds),' ','0'  )
    	test_dataset_name=string.replace( '%s.fold_%4d_of_%4d.test' % (dataset_name,set_index_boosting, outer_folds),' ','0'   )

	print 'The booster which is currently running: %s' % booster_name
	Ensemble_classifier = BOOSTERS[booster_name](max_iter_boosting)	
	#Ensemble_classifier=Adaboost()
	#Ensemble_classifier=Adaboost_instance()
	#Ensemble_classifier=MIBoosting_Xu()
	#Ensemble_classifier=RankBoost()
	#Ensemble_classifier=RankBoost_m3()
	Ensemble_classifier.fit(train_dataset_name, auxiliary_structure)
	Ensemble_classifier.predict()
	
	for iter_index in range(Ensemble_classifier.num_iter_boosting):
		Ensemble_classifier.store_boosting_results(iter_index+1)
	#import pdb;pdb.set_trace()


def server_experiment_backup(task_dict, shared_variables, server):
    for set_index_boosting in range(2):
    	train_dataset_name_to_be_tuned='natural_scene.fold_000%d_of_0002.train' % set_index_boosting
    	test_dataset_name_to_be_tuned='natural_scene.fold_000%d_of_0002.test' % set_index_boosting

    	train_dataset_to_be_tuned=data.get_dataset(train_dataset_name_to_be_tuned)
    	#import pdb; pdb.set_trace()
    	iteration_max=5
    	epsilon={}
    	alpha={}
        instance_weight={}
        
    	shared_variables['instance_weights']=dict.fromkeys(train_dataset_to_be_tuned.instance_ids,1)
    	for label_index in range(5):
		instance_weight[label_index]=shared_variables['instance_weights']
	
    	#shared_variables['instance_weights']=[1,1,1,1]
	prediction_matrix_test4label={}
    	label_matrix_test4label={}
	prediction_matrix_test_accumulated4label={}
    	for iteration in range(1, iteration_max+1):
        	print 'Boosting iteration NO. %d' % iteration
		for label_index in range(5):
			shared_variables['instance_weights']=instance_weight[label_index]
    			task1=run_tune_parameter(train_dataset_name_to_be_tuned,test_dataset_name_to_be_tuned, task_dict , shared_variables, server, label_index)

    			prediction_matrix, label_matrix =task_dict[task1].get_prediction_true_matrix('bag', 'train')
    			prediction_matrix_bool=(prediction_matrix > 0)
    			

			prediction_matrix_test4label[label_index], label_matrix_test4label[label_index] =task_dict[task1].get_prediction_true_matrix('bag', 'test')

			
    			#error_per_instance=[ editdistance.eval(prediction_matrix_bool[i,:], label_matrix[i,:])/float(prediction_matrix.shape[1])  for i in range(prediction_matrix.shape[0])  ]
			error_per_instance_bool=[ prediction_matrix_bool[i,label_index] != label_matrix[i,label_index]  for i in range(prediction_matrix.shape[0])  ]
			error_per_instance=map(lambda x: 1 if x else 0, error_per_instance_bool)

    			weight_per_instance=[ shared_variables['instance_weights'][  train_dataset_to_be_tuned.instance_ids[i] ]     for i in range(prediction_matrix.shape[0]) ] 
    			epsilon[iteration]=np.average( error_per_instance, weights= weight_per_instance   )
        		alpha[iteration]=log(( 1-epsilon[iteration])/float(epsilon[iteration]))
        		#import pdb; pdb.set_trace()
			

			if iteration == 1:
        			prediction_matrix_test_accumulated4label[label_index]=prediction_matrix_test4label[label_index]*alpha[iteration]
			else:
 				prediction_matrix_test_accumulated4label[label_index]=prediction_matrix_test_accumulated4label[label_index]+prediction_matrix_test4label[label_index]*alpha[iteration]

        		
        		#update weights
			instance_weight[label_index]={}
        		for error_per_instance_index in range(len(error_per_instance)):
                		weight_key=train_dataset_to_be_tuned.instance_ids[error_per_instance_index]
				instance_weight[label_index][weight_key]=shared_variables['instance_weights'][weight_key]*exp(alpha[iteration]*error_per_instance[error_per_instance_index])
        		#import pdb; pdb.set_trace()


		
		prediction_matrix_test_accumulated= np.vstack((prediction_matrix_test_accumulated4label[iii][:,iii] for iii in range(5))).transpose()
        	task_dict[task1].store_boosting_results(prediction_matrix_test_accumulated, iteration) #store the accumulated predictions and some evaluation metrics for boosting until current iteration


    
    import pdb; pdb.set_trace() #the end of boosting for one training dataset
    '''
    task_dict[task1].store_boosting_results(prediction_matrix_test_accumulated)
    #store_boosting_results(self, prediction_matrix_test_accumulated, boosting_rounds)
    eval_task1=evaluation_metric.EvaluationMetric(task_dict[task1])
    eval_task1.avg_prec()
    coverage_task1=evaluation_metric.coverage(task_dict[task1])

    shared_variables['instance_weights']=[2,2,2,2]
    task2=run_tune_parameter('natural_scene.fold_0000_of_0002.train','natural_scene.fold_0000_of_0002.test', task_dict , shared_variables, server)

    shared_variables['instance_weights']=[2,2,2,2]
    task2=run_tune_parameter('natural_scene.fold_0001_of_0002.train','natural_scene.fold_0001_of_0002.test', task_dict , shared_variables, server)

    #rettast=task_dict[task1].get_predictions('bag','test')
    
    
    import pdb; pdb.set_trace()
    coverage_task1=evaluation_metric.coverage(task_dict[task1])
    coverage_task2=evaluation_metric.coverage(task_dict[task2])
    '''
    import pdb; pdb.set_trace()

def run_tune_parameter(train, test , tasks, shared_variables, server, label_index=None):
    #train is the string for training dataset
    #test is the string for testing dataset
    #tasks is the all possible tasks in dictionary format, i.e. task_dict
    #shared_variables contains two conponents: one is the queue to be run, the second one is condition_lock that synchronize
    #label_index is the index of label with respect to which the optimal parameter combination is determined    

    #this function will return the optimal task on the training set/testing set pair
    
    #import pdb; pdb.set_trace()
    #run the experiment train with the best parameter tuned on train
    subtasks=dict((k, tasks[k] ) for k in tasks.keys()  if k[2].find(train+'.')==0    ) #subtasks is the dictionary which contains the tasks to tune the parameters for train
    with server.status_lock:
    	for sub_key in subtasks.keys():
		subtasks[sub_key].finished = False
    shared_variables['to_be_run'].put(subtasks)
    shared_variables['condition_lock'].acquire()
    
    #import pdb; pdb.set_trace()
    while(not reduce(lambda x, y: x and y, [ tasks[z].finished for z in subtasks.keys()   ]   )):  #if all tasks are finished
    	print 'blocked by wait'
       	shared_variables['condition_lock'].wait()
	print 'awakened from wait'
    
    shared_variables['condition_lock'].release()  
    print 'all subtasks used for tuning parameters are finished'
    print 'try to choose the optimal parameters for this training dataset'
    
    num_para_combination=max([ subtasks.keys()[x][5] for x in range(len(subtasks) )  ])+1
    statistic_avg_per_para={}
    
    if label_index is None:
    	statisitic_name='AUC'
    else:
	statisitic_name='AUC'+str(label_index)

    for para_index in range(num_para_combination):
 	statistic_avg_per_para[para_index]=np.mean( [tasks[x].get_statistic(statisitic_name)[0] for x in subtasks.keys() if x[5]==para_index] ) 
    
    para_index_optimal = np.argmax(statistic_avg_per_para.values())
    subtasks=dict((k, tasks[k] ) for k in tasks.keys()  if k[2]== train and k[5] == para_index_optimal    )
    
    with server.status_lock:
     	for sub_key in subtasks.keys():
		subtasks[sub_key].finished = False    
    shared_variables['to_be_run'].put(subtasks)
    shared_variables['condition_lock'].acquire()
    while(not reduce(lambda x, y: x and y, [ tasks[z].finished for z in subtasks.keys()   ]   )):  #if all tasks are finished
    	print 'blocked by wait'
       	shared_variables['condition_lock'].wait()
	print 'awakened from wait'
    
    shared_variables['condition_lock'].release()  
    print 'all subtasks are finished'
    
    print 'parameter tuning on training set'+train+' is finished'
    
    return subtasks.keys()[0]  #return the key of the optimal task for training set "train"
    #import pdb; pdb.set_trace()  
    
    



def load_config(configuration_file, results_root_dir):
    
    tasks = {}
    parameter_dict = {}

    print 'Loading configuration...'
    with open(configuration_file, 'r') as f:
        configuration = yaml.load(f)
    #import pdb;pdb.set_trace()
    experiment_key = configuration['experiment_key']
    experiment_name = configuration['experiment_name']
    
    #configuration.pop('booster_params')

    if experiment_name == 'mi_kernels':
        from resampling import NullResamplingConfiguration
        def constructor_from_experiment(experiment):
            return lambda dset: NullResamplingConfiguration(dset)
    else:
        raise ValueError('Unknown experiment name "%s"' % experiment_name)

    for experiment in configuration['experiments']:
        try:
            experiment_id = tuple(experiment[k] for k in experiment_key)
        except KeyError:
            raise KeyError('Experiment missing identifier "%s"'
                            % experiment_key)

        def _missing(pretty_name):
            raise KeyError('%s not specified for experiment "%s"'
                            % (pretty_name, str(experiment_id)))

        def _resolve(field_name, pretty_name):
            field = experiment.get(field_name,
                        configuration.get(field_name, None))
            if field is None: _missing(pretty_name)
            return field

        print 'Setting up experiment "%s"...' % str(experiment_id)

        try:
            dataset = experiment['dataset']
        except KeyError: _missing('Dataset')

        experiment_format = _resolve('experiment_key_format',
                                     'Experiment key format')

        parameter_key = _resolve('parameter_key', 'Parameter key')
        parameter_format = _resolve('parameter_key_format',
                                    'Parameter key format')
        parameters = _resolve('parameters', 'Parameters')
        param_config = ParameterConfiguration(results_root_dir,
                        experiment_name, experiment_id,
                        experiment_format, parameter_key,
                        parameter_format, parameters)
        parameter_dict[experiment_id] = param_config
	#import pdb;pdb.set_trace()
        folds = _resolve('folds', 'Folds')
        fold_config = FoldConfiguration(dataset, *folds)

        resampling_constructor = constructor_from_experiment(experiment)

        priority = experiment.get('priority', 0)

        experiment_config = ExperimentConfiguration(
                                experiment_name, experiment_id,
                                fold_config, param_config,
                                resampling_constructor)
        settings = experiment_config.get_settings()
        #import pdb;pdb.set_trace()
	prog = ProgressMonitor(total=len(settings), print_interval=10,
                               msg='\tGetting tasks')
        for setting in settings:
	    #import pdb;pdb.set_trace()
            key = experiment_config.get_key(**setting)
	    if not INNER_CROSS_VALIDATION:
		if key[2].count('fold_') > 1:  #key[2] corresponds to the name string for training set. We only add tasks for outer layer cross validation
		   continue
            task = experiment_config.get_task(**setting)
            task.priority_adjustment = priority
            task.ground(results_root_dir,
                experiment_format, parameter_format)
            tasks[key] = task
            prog.increment()

    import pdb;pdb.set_trace()
    return tasks, parameter_dict

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    parser = OptionParser(usage="Usage: %prog configfile resultsdir")
    options, args = parser.parse_args()
    options = dict(options.__dict__)
    if len(args) != 2:
        parser.print_help()
        exit()
    start_experiment_noCV(*args, **options)