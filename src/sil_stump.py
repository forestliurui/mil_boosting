"""
Implements Single Instance Learning SVM
"""
import numpy as np
import inspect
from mi_svm import SVM
from sklearn.tree import DecisionTreeClassifier



class SIL_Stump(DecisionTreeClassifier):
    """
    Single-Instance Learning applied to MI data
    """

    def __init__(self, **kwargs):
        """
        @param kernel : the desired kernel function; can be linear, quadratic,
                        polynomial, or rbf [default: linear]
        @param C : the loss/regularization tradeoff constant [default: 1.0]
        @param scale_C : if True [default], scale C by the number of examples
        @param p : polynomial degree when a 'polynomial' kernel is used
                   [default: 3]
        @param gamma : RBF scale parameter when an 'rbf' kernel is used
                      [default: 1.0]
        @param verbose : print optimization status messages [default: True]
        @param sv_cutoff : the numerical cutoff for an example to be considered
                           a support vector [default: 1e-7]
        """
	kwargs.update({'max_depth': 1})
	del kwargs['kernel']
	del kwargs['C']
	del kwargs['base_kernel']
	del kwargs['normalization']
        super(SIL_Stump, self).__init__(**kwargs)
        self._bags = None
        self._bag_predictions = None

    def fit(self, bags, y, weights = None):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @param y : an array-like object of length n containing -1/+1 labels
	@param weights: an array-like object of length n containing weights for each bag or each instance
        """
        self._bags = [np.asmatrix(bag) for bag in bags]
        y = np.asmatrix(y).reshape((-1, 1))
        svm_X = np.vstack(self._bags)
        svm_y = np.vstack([float(cls) * np.matrix(np.ones((len(bag), 1)))
                           for bag, cls in zip(self._bags, y)])

	if weights is None:
		svm_weights = None
	elif len(bags) == len(weights.tolist()):
		weights = np.asmatrix(weights).reshape((-1, 1))
		svm_weights = np.vstack([float(cls) * np.matrix(np.ones((len(bag), 1)))
                           for bag, cls in zip(self._bags, weights)])
		svm_weights = np.ravel(svm_weights)
	else:
		svm_weights = weights
	#import pdb;pdb.set_trace()
        super(SIL_Stump, self).fit(svm_X, np.ravel(svm_y), svm_weights)
	#import pdb;pdb.set_trace()

    def _compute_separator(self, K):
        super(SIL_Stump, self)._compute_separator(K)
        self._bag_predictions = _inst_to_bag_preds(self._predictions, self._bags)

    def predict(self, bags):
        """
        @param bags : a sequence of n bags; each bag is an m-by-k array-like
                      object containing m instances with k features
        @return : an array of length n containing real-valued label predictions
                  (threshold at zero to produce binary predictions)
        """
        bags = [np.asmatrix(bag) for bag in bags]
        inst_preds = super(SIL_Stump, self).predict(np.vstack(bags))
        return _inst_to_bag_preds(inst_preds, bags)

    def get_params(self, deep=True):
        """
        return params
        """
        args, _, _, _ = inspect.getargspec(super(SIL, self).__init__)
        args.pop(0)
        return {key: getattr(self, key, None) for key in args}


def _inst_to_bag_preds(inst_preds, bags):
    return np.array([np.max(inst_preds[slice(*bidx)])
                     for bidx in slices(map(len, bags))])

def slices(groups):
    """
    Generate slices to select
    groups of the given sizes
    within a list/matrix
    """
    i = 0
    for group in groups:
        yield i, i + group
        i += group