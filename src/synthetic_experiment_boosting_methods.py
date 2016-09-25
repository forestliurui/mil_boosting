"""
This contains all the data that could be used in synthetic_experiment_boost.py
"""

import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, "~/.lib/scikit-learn/sklearn")

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron

from sklearn.datasets import make_gaussian_quantiles
from sklearn.svm import SVC
from RankBoost_nondistributed import RankBoost
from martiboost_nondistributed import MartiBoost
from MIBoosting_Xu_nondistributed import MIBoosting_Xu
from RankBoost_m3_nondistributed import RankBoost_m3
from RankBoost_modiII_nondistributed import RankBoost_modiII
from rBoost_nondistributed import RBoost

from mi_svm import SVM
import data
import dill



def getMethod1():
	#self-made Adaboost + decision stump
	from Adaboost_nondistributed import AdaBoost
	param = {"max_iter_boosting": 200}
	bdt = AdaBoost(**param)
	return bdt

def getMethod2():
	#AdaBoosted decision tree
	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)
	return bdt

def getMethod3():
	#linear svm+adaboost
	params = {'C': 100000000, 'kernel': 'linear'}
	bdt = AdaBoostClassifier(SVC(**params),
                         algorithm="SAMME",
                         n_estimators=2)
	return bdt

def getMethod4():
	#rbf svm+adaboost
	params = {'C': 10000, 'kernel': 'rbf','gamma':1000}
	bdt = AdaBoostClassifier(SVC(**params),
                         algorithm="SAMME",
                         n_estimators=15)
	return bdt

def getMethod5():
	#rankboost
	params = {'C': 10, 'kernel': 'linear', 'max_iter_boosting':10}
	bdt = RankBoost(**params)
	return bdt

def getMethod6():
	#linear svm
	params = {'C': 10, 'kernel': 'linear'}
	bdt = SVC(**params)
	return bdt

def getMethod7():
	#rbf svm
	params = {'C': 10, 'kernel': 'rbf', 'gamma': 10000}
	bdt = SVC(**params)
	return bdt


def getMethod8():
	#martiboost + linear svm
	params = {'C': 10, 'kernel': 'linear'}
	bdt = MartiBoost(**params)
	return bdt

def getMethod9():
	#martiboost + balanced_decision_stump
	params = {'weak_classifier': 'dtree_stump_balanced', 'max_iter_boosting': 200}
	bdt = MartiBoost(**params)
	return bdt

def getMethod10():
	#MIBoosting + decision_stump
	params = {'weak_classifier': 'dtree_stump','max_depth': 1,'max_iter_boosting': 200}
	bdt1 = MIBoosting_Xu(**params)
	return bdt


def getMethod11():
	#rankboost_m3
	params = {'weak_classifier': 'dtree_stump','max_depth': 1,'max_iter_boosting': 2000}
	bdt = RankBoost_m3(**params)
	return bdt

def getMethod12():
	#rankboost + decision stump
	params = {'weak_classifier': 'dtree_stump','max_depth': 1,'max_iter_boosting': 20}
	bdt = RankBoost(**params)
	return bdt


def getMethod13():
	#rankboost_modiII + decision stump
	params = {'weak_classifier': 'dtree_stump','max_depth': 1,'max_iter_boosting': 20}
	bdt = RankBoost_modiII(**params)
	return bdt

def getMethod14():
	#rboost + decision stump
	params = {'weak_classifier': 'dtree_stump','max_depth': 1,'max_iter_boosting': 20}
	bdt = RBoost(**params)
	return bdt

def getMethod15():
	#AdaBoosted Perceptron
	bdt = AdaBoostClassifier(Perceptron(penalty = 'l1', n_iter = 50),
                         algorithm="SAMME",
                         n_estimators=200)
	return bdt
