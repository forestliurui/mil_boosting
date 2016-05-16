"""
This is the test file for RankBoost
"""

from RankBoost_modiII_ranking_nondistributed import RankBoost_modiII_ranking

X={1:[0,0], 2:[0,1], 3:[0,2], 4:[0, -1], 5:[0, -2], 6:[2,0], 7:[-1, 0], 8:[1,2], 9:[1,1], 10:[1,0], 11:[1,-1], 12:[1,-2]}
y = [(a,b) for a in range(7,13) for b in range(1,7)]

ranker = RankBoost_modiII_ranking()
ranker.fit(X,y)

predictions = ranker.predict_train(iter = 3)

import pdb;pdb.set_trace()