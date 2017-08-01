"""
This is the query engine for sqlite3 database
"""

import sqlite3
import unittest

class QueryEngine(object):
     def __init__(self, database_file):
          self.connection = sqlite3.connect(database_file)
          self.c = self.connection.cursor()
          
     def query(self, targets, table, conditions = None):
          """
          @targets: the list of strings representing the names of the target columns
          @table:  the string representing the name of the table from which target is to be selected
          @conditions: the dictionary for which every column indicated by key should be equal to its corresponding value
          @return: a list of tuples; every element in the tuple correponds to every element in targets. Every tuple is one record/row selected
          """
          query_str = "SELECT %s FROM %s" %( ",".join(targets), table )
          condition_str_list = []
          if conditions is not None:
               for item in conditions.items():
                   if type(item[1]) is str:
                       condition_str_list.append( '%s = "%s"'%( item[0], item[1]  ) )
                   else:
                       condition_str_list.append('%s = %s'%(item[0], item[1])   )
               query_str += " WHERE "
               query_str += (" AND ".join(condition_str_list) )
          #import pdb;pdb.set_trace() 
          res = [ row for row in self.c.execute(query_str)   ]
          return res
     def querySingleTarget(self, target, table, conditions = None):
          """
          @target: a string representing the name of target column
          @return:a list of values
          the rest is the same with query()
          """
          targets = [target]
          res_general = self.query(targets, table, conditions)
          res = [ row[0] for row in res_general  ]
          return res

class TestQueryEngine(unittest.TestCase):
     def test1(self):
         database_file = "ranking/results/LETOR_MovieLen/LETOR_lowerbound_400_upperbound_600_rankboost_modiII.db"
         engine = QueryEngine(database_file)
         targets = ["*"]
         table = "statistic_names"
         conditions = {"statistic_name": "train_error"}
         #conditions = None
         print(engine.query(targets, table, conditions))
     def test2(self):
         database_file = "ranking/results/LETOR_MovieLen/LETOR_lowerbound_400_upperbound_600_rankboost_modiII.db"
         engine = QueryEngine(database_file)
         targets = ["*"]
         table = "statistic_names"
         conditions = {"statistic_name_id": 4}
         #conditions = None
         print(engine.query(targets, table, conditions))

     def test3(self):
         database_file = "ranking/results/LETOR_MovieLen/LETOR_lowerbound_400_upperbound_600_rankboost_modiII.db"
         engine = QueryEngine(database_file)
         targets = ["statistic_name", "statistic_name_id"]
         table = "statistic_names"
         conditions = {"statistic_name_id": 4}
         #conditions = None
         print(engine.query(targets, table, conditions))
 
     def test4(self):
         database_file = "ranking/results/LETOR_MovieLen/LETOR_lowerbound_400_upperbound_600_rankboost_modiII.db"
         engine = QueryEngine(database_file)
         targets = "statistic_name"
         table = "statistic_names"
         conditions = {"statistic_name_id": 4}
         #conditions = None
         print(engine.querySingleTarget(targets, table, conditions))

     def test5(self):
         database_file = "ranking/results/LETOR_MovieLen/LETOR_lowerbound_400_upperbound_600_rankboost_modiII.db"
         engine = QueryEngine(database_file)
         targets = "statistic_name"
         table = "statistic_names"
         conditions = {"statistic_name_id": 400}
         #conditions = None
         print(engine.querySingleTarget(targets, table, conditions))
         #should return empty list, i.e. []

if __name__ == "__main__":
     unittest.main()
