"""
this is the testing file for boosting algorithms
"""


import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.svm import SVC
from RankBoost_nondistributed import RankBoost

from mi_svm import SVM

try:
	from sklearn.metrics import roc_auc_score as score
except:
	from sklearn.metrics import auc_score as score

# Construct dataset
X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=200, n_features=2,
                                 n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                 n_samples=300, n_features=2,
                                 n_classes=2, random_state=1)
X = np.concatenate((X1, X2))
y = np.concatenate((y1, - y2 + 1))

max_iter_boosting = 30

params = {'C': 1000, 'kernel': 'linear', 'max_iter_boosting':max_iter_boosting}
bdt = RankBoost(**params)
print "fitting the training set"
bdt.fit(X, y)
print "fitting completed"
auc=[]
for iter in range(1, max_iter_boosting+1):
	#import pdb;pdb.set_trace()
	Z = bdt.predict(X, iter=iter)
	#import pdb;pdb.set_trace()
	auc.append(score(y==1, Z))

import pdb;pdb.set_trace()

	


