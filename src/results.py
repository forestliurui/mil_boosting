"""
Handles reading/writing results
"""
import thread

RESULTS_MANAGERS = {}

def get_result_manager(results_path):
    if results_path not in RESULTS_MANAGERS:
        RESULTS_MANAGERS[results_path] = ResultsManager(results_path)
    return RESULTS_MANAGERS[results_path]

class ResultsManager(object):

    def __init__(self, results_path):
        self.results_path = results_path
        self.connection_thread = None
        self._connection = None
        self.dataset_ids = {}
        self.parameter_set_ids = {}
        self.statistic_ids = {}
        self._init_database()

    def _init_database(self):
        connection = self.get_connection()
        with connection:
            # Create tables and indices
            connection.execute(
                'CREATE TABLE IF NOT EXISTS datasets '
                '(dataset_id integer primary key, '
                'dataset_name text unique)'
            )
            connection.execute(
                'CREATE INDEX IF NOT EXISTS dataset_name_index '
                'ON datasets (dataset_name)'
            )
            connection.execute(
                'CREATE TABLE IF NOT EXISTS parameter_sets '
                '(parameter_set_id integer primary key, '
                'parameter_set_name text unique)'
            )
            connection.execute(
                'CREATE INDEX IF NOT EXISTS parameter_set_name_index '
                'ON parameter_sets (parameter_set_name)'
            )

            connection.execute(
                'CREATE TABLE IF NOT EXISTS instance_predictions '
                '(train_set_id integer, test_set_id integer, '
                'parameter_set_id integer, parameter_set_index integer, '
                'bag_id text, instance_id text, label real)'
            )
            connection.execute(
                'CREATE INDEX IF NOT EXISTS instance_prediction_index '
                'ON instance_predictions (train_set_id, test_set_id, '
                'parameter_set_id, parameter_set_index)'
            )

            connection.execute(
                'CREATE TABLE IF NOT EXISTS bag_predictions '
                '(train_set_id integer, test_set_id integer, '
                'parameter_set_id integer, parameter_set_index integer, '
                'bag_id text, label real)'
            )
            connection.execute(
                'CREATE INDEX IF NOT EXISTS bag_prediction_index '
                'ON bag_predictions (train_set_id, test_set_id, '
                'parameter_set_id, parameter_set_index)'
            )

            connection.execute(
                'CREATE TABLE IF NOT EXISTS statistic_names '
                '(statistic_name_id integer primary key, '
                'statistic_name text unique)'
            )
            connection.execute(
                'CREATE INDEX IF NOT EXISTS statistic_names_index '
                'ON statistic_names (statistic_name)'
            )
            connection.execute(
                'CREATE TABLE IF NOT EXISTS statistics '
                '(train_set_id integer, test_set_id integer, '
                'parameter_set_id integer, parameter_set_index integer, '
                'statistic_name_id integer, statistic_value real)'
            )
            connection.execute(
                'CREATE INDEX IF NOT EXISTS statistics_index '
                'ON statistics (train_set_id, test_set_id, '
                'parameter_set_id, parameter_set_index, statistic_name_id)'
            )
            #the following tables are especially for boosting 
            connection.execute(
                'CREATE TABLE IF NOT EXISTS statistics_boosting '
                '(train_set_id integer, test_set_id integer, '
                'parameter_set_id integer,  boosting_rounds integer, '
                'statistic_name_id integer, statistic_value real)'
            )
            connection.execute(
                'CREATE INDEX IF NOT EXISTS statistics_boosting_index '
                'ON statistics_boosting (train_set_id, test_set_id, '
                'parameter_set_id,  boosting_rounds, statistic_name_id)'
            )
	    connection.execute(
                'CREATE TABLE IF NOT EXISTS instance_raw_predictions_boosting '
                '(train_set_id integer, test_set_id integer, '
                'parameter_set_id integer, boosting_rounds integer,'
                'bag_id text, instance_id text, label real)'
            )
            connection.execute(
                'CREATE INDEX IF NOT EXISTS instance_raw_prediction_boosting_index '
                'ON instance_raw_predictions_boosting (train_set_id, test_set_id, '
                'parameter_set_id,  boosting_rounds)'
            )
 	    connection.execute(
                'CREATE TABLE IF NOT EXISTS bag_raw_predictions_boosting '
                '(train_set_id integer, test_set_id integer, '
                'parameter_set_id integer, boosting_rounds integer,'
                'bag_id text, label real)'
            )
            connection.execute(
                'CREATE INDEX IF NOT EXISTS bag_raw_prediction_boosting_index '
                'ON bag_raw_predictions_boosting (train_set_id, test_set_id, '
                'parameter_set_id,  boosting_rounds)'
            )
	    connection.execute(
                'CREATE TABLE IF NOT EXISTS instance_accum_predictions_boosting '
                '(train_set_id integer, test_set_id integer, '
                'parameter_set_id integer, boosting_rounds integer,'
                'bag_id text, instance_id text, label real)'
            )
            connection.execute(
                'CREATE INDEX IF NOT EXISTS instance_accum_prediction_boosting_index '
                'ON instance_accum_predictions_boosting (train_set_id, test_set_id, '
                'parameter_set_id,  boosting_rounds)'
            )
 	    connection.execute(
                'CREATE TABLE IF NOT EXISTS bag_accum_predictions_boosting '
                '(train_set_id integer, test_set_id integer, '
                'parameter_set_id integer, boosting_rounds integer,'
                'bag_id text, label real)'
            )
            connection.execute(
                'CREATE INDEX IF NOT EXISTS bag_accum_prediction_boosting_index '
                'ON bag_accum_predictions_boosting (train_set_id, test_set_id, '
                'parameter_set_id,  boosting_rounds)'
            )


		

        # Load existing name -> id mappings
        cursor = connection.cursor()

        cursor.execute('SELECT dataset_id, dataset_name FROM datasets')
        for did, dname in cursor.fetchall():
            self.dataset_ids[dname] = did

        cursor.execute(
            'SELECT parameter_set_id, parameter_set_name '
            'FROM parameter_sets'
        )
        for pid, pname in cursor.fetchall():
            self.parameter_set_ids[pname] = pid

        cursor.execute(
            'SELECT statistic_name_id, statistic_name '
            'FROM statistic_names'
        )
        for sid, sname in cursor.fetchall():
            self.statistic_ids[sname] = sid

    def get_statistic_id(self, statistic_name):
        if statistic_name not in self.statistic_ids:
            connection = self.get_connection()
            with connection:
                connection.execute(
                    'INSERT INTO statistic_names '
                    'VALUES (NULL, ?)',
                    (statistic_name,)
                )
                cursor = connection.cursor()
                cursor.execute(
                    'SELECT statistic_name_id '
                    'FROM statistic_names '
                    'WHERE statistic_name=?',
                    (statistic_name,)
                )
                for sid in cursor.fetchall():
                    # Should only ever be 1 result by uniqueness
                    self.statistic_ids[statistic_name] = sid[0]

        return self.statistic_ids[statistic_name]

    def get_dataset_id(self, dataset_name):
        if dataset_name not in self.dataset_ids:
            connection = self.get_connection()
            with connection:
                connection.execute(
                    'INSERT INTO datasets '
                    'VALUES (NULL, ?)',
                    (dataset_name,)
                )
                cursor = connection.cursor()
                cursor.execute(
                    'SELECT dataset_id '
                    'FROM datasets '
                    'WHERE dataset_name=?',
                    (dataset_name,)
                )
                for did in cursor.fetchall():
                    # Should only ever be 1 result by uniqueness
                    self.dataset_ids[dataset_name] = did[0]

        return self.dataset_ids[dataset_name]

    def get_parameter_set_id(self, parameter_set_name):
        if parameter_set_name not in self.parameter_set_ids:
            connection = self.get_connection()
            with connection:
                connection.execute(
                    'INSERT INTO parameter_sets '
                    'VALUES (NULL, ?)',
                    (parameter_set_name,)
                )
                cursor = connection.cursor()
                cursor.execute(
                    'SELECT parameter_set_id '
                    'FROM parameter_sets '
                    'WHERE parameter_set_name=?',
                    (parameter_set_name,)
                )
                for pid in cursor.fetchall():
                    # Should only ever be 1 result by uniqueness
                    self.parameter_set_ids[parameter_set_name] = pid[0]

        return self.parameter_set_ids[parameter_set_name]

    def get_connection(self):
        current_thread = thread.get_ident()
        if (self._connection is None
            or self.connection_thread != current_thread):
            import sqlite3
            self.connection_thread = current_thread
            self._connection = sqlite3.connect(self.results_path)
        return self._connection

    def get_statistic(self, statistic_name, train, test, parameter_set, parameter_set_index):
        connection = self.get_connection()
        cursor = connection.cursor()
        train_id = self.get_dataset_id(train)
        test_id = self.get_dataset_id(test)
        parameter_set_id = self.get_parameter_set_id(parameter_set)
        stat_id = self.get_statistic_id(statistic_name)
        cursor.execute(
            'SELECT statistic_value FROM statistics WHERE '
            'train_set_id=? AND test_set_id=? AND '
            'parameter_set_id=? AND parameter_set_index=? '
            'AND statistic_name_id=?',
            (train_id, test_id, parameter_set_id,
             parameter_set_index, stat_id)
        )
        return cursor.fetchone()

    def is_finished(self, train, test, parameter_set = None, parameter_set_index = None, boosting_round = None):
        connection = self.get_connection()
        cursor = connection.cursor()
        train_id = self.get_dataset_id(train)
        test_id = self.get_dataset_id(test)
	stat_id = self.get_statistic_id('FINISHED')

	execute_string = 'SELECT * FROM statistics_boosting WHERE train_set_id=%d AND test_set_id=%d AND statistic_name_id=%d' %  (train_id, test_id, stat_id)

	if parameter_set is not None:
		parameter_set_id = self.get_parameter_set_id(parameter_set)
		execute_string += 'AND parameter_set_index=%d ' % parameter_set_id
	if boosting_round is not None:
		execute_string += 'AND boosting_rounds=%d ' % boosting_round
        	
        cursor.execute(execute_string)

	"""
	cursor.execute(
            'SELECT * FROM statistics_boosting WHERE '
            'train_set_id=? AND test_set_id=? AND '
            'parameter_set_id=? '
            'AND statistic_name_id=?',
            (train_id, test_id, parameter_set_id,
              stat_id)
        )
	"""

        return (cursor.fetchone() is not None)

    def get_bag_predictions(self, train, test,
            parameter_set, parameter_set_index, test_set_labels=True):
        #import pdb;pdb.set_trace()
        train_id = self.get_dataset_id(train)
        if test_set_labels:
            test_id = self.get_dataset_id(test)
        else:
            test_id = train_id
        parameter_set_id = self.get_parameter_set_id(parameter_set)

        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute(
            'SELECT bag_id, label FROM bag_predictions '
            'WHERE train_set_id=? AND test_set_id=? AND '
            'parameter_set_id=? AND parameter_set_index=?',
            (train_id, test_id, parameter_set_id, parameter_set_index)
        )
        predictions = dict([(bid, label) for bid, label in cursor.fetchall()])
        return predictions


    def get_instance_predictions(self, train, test,
            parameter_set, parameter_set_index, test_set_labels=True):

        train_id = self.get_dataset_id(train)
        if test_set_labels:
            test_id = self.get_dataset_id(test)
        else:
            test_id = train_id
        parameter_set_id = self.get_parameter_set_id(parameter_set)

        connection = self.get_connection()
        cursor = connection.cursor()
        cursor.execute(
            'SELECT bag_id, instance_id, label FROM instance_predictions '
            'WHERE train_set_id=? AND test_set_id=? AND '
            'parameter_set_id=? AND parameter_set_index=?',
            (train_id, test_id, parameter_set_id, parameter_set_index)
        )
        predictions = dict([((bid, iid), label)
                            for bid, iid, label in cursor.fetchall()])
        return predictions

    def store_results(self, submission, train, test,
            parameter_set, parameter_set_index):

        train_id = self.get_dataset_id(train)
        test_id = self.get_dataset_id(test)
        parameter_set_id = self.get_parameter_set_id(parameter_set)
        
        instance_records=[]
        if 'instance_predictions' in submission.keys():
           instance_records = [
            (train_id, test_id,
             parameter_set_id, parameter_set_index,
             bid, iid, label)
            for (bid, iid), label in
                submission['instance_predictions']['test'].items()
           ]
	   if 'train' in submission['instance_predictions'].keys():

           	instance_records += [
            		(train_id, train_id,
             		parameter_set_id, parameter_set_index,
             		bid, iid, label)
            		for (bid, iid), label in
                	submission['instance_predictions']['train'].items()
           	]

	if 'bag_predictions' in submission.keys():
        	bag_records = [
            		(train_id, test_id,
             		parameter_set_id, parameter_set_index,
             		bid, label)
            		for bid, label in
                		submission['bag_predictions']['test'].items()
        	]
         	if 'train' in submission['bag_predictions'].keys():
          		bag_records += [
            			(train_id, train_id,
             			parameter_set_id, parameter_set_index,
             			bid, label)
            			for bid, label in
                			submission['bag_predictions']['train'].items()
          		]

	'''
        bag_records += [
            (train_id, train_id,
             parameter_set_id, parameter_set_index,
             bid, label)
            for bid, label in
                submission['bag_predictions']['train'].items()
        ]
	'''
        statistics_records=[]
        if 'statistics' in submission.keys():
           statistics_records = [
            (train_id, test_id,
             parameter_set_id, parameter_set_index,
             self.get_statistic_id(sname), value)
            for sname, value in submission['statistics'].items()
           ]
           statistics_records.append(
            (train_id, test_id,
             parameter_set_id, parameter_set_index,
             self.get_statistic_id('FINISHED'), 1.0)
           )

        connection = self.get_connection()

	cc=connection.cursor()	
	string_delete='delete from statistics where train_set_id = %d and test_set_id = %d and parameter_set_id = %d and parameter_set_index = %d and statistic_name_id = %d' % (train_id, test_id,
             parameter_set_id, parameter_set_index,
             self.get_statistic_id('AUC'))
	cc.execute(string_delete  )	
	connection.commit()

        with connection:
            if len(instance_records) > 0:
                connection.executemany(
                    'INSERT INTO instance_predictions '
                    'VALUES (?, ?, ?, ?, ?, ?, ?)', instance_records
                )
	    #import pdb;pdb.set_trace() 
	    if len(bag_records) > 0:

                connection.executemany(
                    'INSERT INTO bag_predictions '
                    'VALUES (?, ?, ?, ?, ?, ?)', bag_records
                )
            if len(statistics_records) > 0:
               connection.executemany(
                'INSERT INTO statistics '
                'VALUES (?, ?, ?, ?, ?, ?)', statistics_records
               )
     
     
    def store_results_boosting(self, submission, boosting_rounds, train, test, parameter_set, parameter_set_index):
	#import pdb;pdb.set_trace()

        train_id = self.get_dataset_id(train)
        test_id = self.get_dataset_id(test)
        parameter_set_id = self.get_parameter_set_id(parameter_set)
	#import pdb;pdb.set_trace()

        instance_records_raw=[]
	bag_records_raw=[]
        if 'raw' in submission.keys():
		if 'instance_predictions' in submission['raw'].keys():
           		instance_records_raw = [
            			(train_id, test_id,
             			parameter_set_id,  boosting_rounds, 
             			bid, iid, label)
            			for (bid, iid), label in
                			submission['raw']['instance_predictions']['test'].items()
           		]
	   		if 'train' in submission['raw']['instance_predictions'].keys():
            			instance_records_raw += [
              				(train_id, train_id,
                			parameter_set_id, boosting_rounds,
                			bid, iid, label)
                			for (bid, iid), label in
                				submission['raw']['instance_predictions']['train'].items()
              			]
		#import pdb;pdb.set_trace()
		if 'bag_predictions' in submission['raw'].keys():

        		bag_records_raw = [
            			(train_id, test_id,
             			parameter_set_id, boosting_rounds,
             			bid, label)
            			for bid, label in
                			submission['raw']['bag_predictions']['test'].items()
        		]
        		if 'train' in submission['raw']['bag_predictions'].keys():
          			bag_records_raw += [
            				(train_id, train_id,
             				parameter_set_id, boosting_rounds,
             				bid, label)
            				for bid, label in
                				submission['raw']['bag_predictions']['train'].items()
          			]
	#import pdb;pdb.set_trace()

	instance_records_accum=[]
	bag_records_accum=[]
	if 'accum' in submission.keys():
		if 'instance_predictions' in submission['accum'].keys():
           		instance_records_accum = [
            			(train_id, test_id,
             			parameter_set_id,  boosting_rounds, 
             			bid, iid, label)
            			for (bid, iid), label in
                			submission['accum']['instance_predictions']['test'].items()
           		]
	   		if 'train' in submission['accum']['instance_predictions'].keys():
            			instance_records_accum += [
              				(train_id, train_id,
                			parameter_set_id, boosting_rounds,
                			bid, iid, label)
                			for (bid, iid), label in
                				submission['accum']['instance_predictions']['train'].items()
              			]
		if 'bag_predictions' in submission['accum'].keys():

        		bag_records_accum = [
            			(train_id, test_id,
             			parameter_set_id, boosting_rounds,
             			bid, label)
            			for bid, label in
                			submission['accum']['bag_predictions']['test'].items()
        		]
        		if 'train' in submission['accum']['bag_predictions'].keys():
          			bag_records_accum += [
            				(train_id, train_id,
             				parameter_set_id,boosting_rounds,
             				bid, label)
            				for bid, label in
                				submission['accum']['bag_predictions']['train'].items()
          			]

	

        statistics_records=[]
        if 'statistics_boosting' in submission.keys():
           statistics_records = [
            (train_id, test_id,
             parameter_set_id,  boosting_rounds,
             self.get_statistic_id(sname), value)
            for sname, value in submission['statistics_boosting'].items()
           ]
           statistics_records.append(
            (train_id, test_id,
             parameter_set_id, boosting_rounds,
             self.get_statistic_id('FINISHED'), 1.0)
           )

        connection = self.get_connection()

 	#import pdb;pdb.set_trace()

        with connection:
            if len(instance_records_raw) > 0:
                connection.executemany(
                    'INSERT INTO instance_raw_predictions_boosting '
                    'VALUES (?, ?, ?, ?, ?, ?, ?)', instance_records_raw
                )
	    #import pdb;pdb.set_trace() 
            if len(instance_records_raw) > 0:
            	connection.executemany(
                	'INSERT INTO bag_raw_predictions_boosting '
                		'VALUES (?, ?, ?, ?, ?, ?)', bag_records_raw
            	)

     	    if len(instance_records_accum) > 0:
                connection.executemany(
                    'INSERT INTO instance_accum_predictions_boosting '
                    'VALUES (?, ?, ?, ?, ?, ?, ?)', instance_records_accum
                )
	    #import pdb;pdb.set_trace() 
            if len(instance_records_accum) > 0:
            	connection.executemany(
                	'INSERT INTO bag_accum_predictions_boosting '
                		'VALUES (?, ?, ?, ?, ?, ?)', bag_records_accum
            	)

            if len(statistics_records) > 0:
               connection.executemany(
                'INSERT INTO statistics_boosting '
                'VALUES (?, ?, ?, ?, ?, ?)', statistics_records
               )