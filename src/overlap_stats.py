#!/usr/bin/env python

import numpy as np
from data import get_dataset
#import dimensionality_reduction as dr
from scipy.spatial.distance import cdist
from sklearn.mixture import GMM
from scipy.stats import entropy

def inst_labeled_lists(bags, y, inst_labels):
    """ Returns lists of true positives, false positives, and true negatives """
    #  Lists of positive bags and negative bags, respectively
    pos_list = [bags[i] for i in range(len(y)) if y[i] == True]
    pos_inst_labels = [inst_labels[i] for i in range(len(y)) if y[i] == True]
    neg_list = [bags[i] for i in range(len(y)) if y[i] == False]
    neg_inst_labels = [inst_labels[i] for i in range(len(y)) if y[i] == False]

    #  List of instances in positive bags that are negative labeled.
    #  Basically, go through each bag in the positive list, and grab all the instances
    #  with false labels.  Then stack them all in one big array.

    false_positives = np.vstack([np.vstack([pos_list[i][j] for j in range(len(pos_list[i])) if pos_inst_labels[i][j] == False]) for i in range(len(pos_list))])
    
    true_positives = np.vstack([np.vstack([pos_list[i][j] for j in range(len(pos_list[i])) if pos_inst_labels[i][j] == True]) for i in range(len(pos_list))])

    negatives = np.vstack([np.vstack([neg_list[i][j] for j in range(len(neg_list[i])) if neg_inst_labels[i][j] == False]) for i in range(len(neg_list))])

    """
    false_positives = np.vstack([
        np.vstack( [np.vstack([
        pos_list[i][j] for j in range(len(pos_list[i])) if pos_inst_labels[i][j] == False]
        )] for i in range(len(pos_list)) )
    ])

    print false_positives

    negatives = np.vstack([np.vstack(
        np.vstack([neg_list[i][j] for j in range(len(neg_list[i]))])
        for i in range(len(neg_list)))])
    
    true_positives = np.vstack([np.vstack(
        np.vstack([pos_list[i][j] for j in range(len(pos_list[i])) if pos_inst_labels[i][j] == True])
        for i in range(len(pos_list)))])
"""
    return true_positives, false_positives, negatives
    

def k_nearest_in_negative(bags, y, inst_labels, k=5, percent=0, min=True, include_fp=False):
    """ For each negative instance p in a positive bag,
        find the k nearest neighbors (x1, ... xk)
        Count negative instances.  If include_fp is False, then only count
        instances in negative bags.  Divide by k.
        Returns minimum % or average % per instance depending on
        whether min is true.
        k can be a parameter or calculated as a % of the size
        of the dataset if percent is set.  
        The closer this measure is to 1, the better.  """

    true_positives, false_positives, negatives = inst_labeled_lists(bags, y, inst_labels)

    if include_fp:
        all_instances = np.vstack(bags)
        extended_labels = [[y[i]] * len(bags[i]) for i in range(len(bags))]
    else:
        all_instances = np.vstack([true_positives, negatives])
        extended_labels = [[True] * len(true_positives) + [False] * len(negatives)]
    extended_labels = [label for bag in extended_labels for label in bag]   # Flatten labels
    distances = cdist(false_positives, all_instances, 'euclidean')

    if percent:
        k = len(all_instances) * (percent / 100.)
    knneg = np.zeros(len(false_positives))
    for i in range(len(false_positives)):
        indices = np.argsort(distances[i])[:k]
        i_labels = [extended_labels[idx] for idx in indices]
        #  Assumption is that labels are True/False...
        knneg[i] = float(len(i_labels) - sum(i_labels)) / len(i_labels)

    if min:
        return np.min(knneg)
    else:
        return np.average(knneg)
    


def epsilon_hyperspheres(bags, y, inst_labels, epsilon=0.5):
    """ For each negative instance p in a positive bag,
        For each negative instance n in a negative bag,
            if ||p-n|| < epsilon, then count(p)++
        Returns average over all ratios count(p)/sum(n) """

    true_positives, false_positives, negatives = inst_labeled_lists(bags, y, inst_labels)

    dists = cdist(false_positives, true_positives, 'euclidean')
    sum_p = sum((dists < epsilon).T)    # dists is p x n, but sum is column-wise, so .T
    sum_n = true_positives.shape[0]
    return np.average(sum_p) / sum_n    # Average of count(p), divided by sum(n)


def radius_ratio(bags, y, inst_labels, percent=-1, k=-1):
    """ Find the ratio of the euclidean distance between a false positive point f
        and the n%th nearest negative point and the same distance to the n%th nearest
        positive point, averaged over all false positive points. """

    true_positives, false_positives, negatives = inst_labeled_lists(bags,y, inst_labels)
    
    if k == -1:
        pos_idx = len(true_positives) * percent
        neg_idx = len(negatives) * percent
    else:
        pos_idx = k
        neg_idx = k

    pos_dists = cdist(false_positives, true_positives, 'euclidean')
    neg_dists = cdist(false_positives, negatives, 'euclidean')
    ratios = np.zeros(len(false_positives))
    for i in range(len(false_positives)):
        nd = np.sort(neg_dists[i])
        pd = np.sort(pos_dists[i])
        ratios[i] = nd[pos_idx] / pd[neg_idx]
    return np.average(ratios)
    

def distribution_separation(bags, bag_dict, sample_size=1e2, n_components=10):
    """ Figure out the distributions of:
            a)  The negative instances n
            b)  The false-positive instances p (negative instances in positive bags)
        Then find the kl-divergence between distribution A and distribution B, returned.
        Distributions are modeled as Mixtures-of-Gaussians.
        This is the 
        """
    #  Lists of positive bags and negative bags, respectively
    pos_list = filter(lambda x: x[2] == True, bag_dict.itervalues())
    neg_list = filter(lambda x: x[2] == False, bag_dict.itervalues())

    #  List of arrays of instances in positive bags that are negative labeled.
    false_positives = [np.vstack([bag[1][i] for i in range(len(bag[1])) if not bag[3][i]])
                        for bag in pos_list]

    #  Arrays of negative instances in negative bags and in positive bags
    n_list = np.vstack([np.vstack(x[1] for x in neg_list)])
    p_list = np.vstack(false_positives)

    p_model = GMM(n_components=10)
    n_model = GMM(n_components=10)
    p_model.fit(p_list)
    n_model.fit(n_list)
    p_samples = p_model.sample(sample_size)
    n_samples = n_model.sample(sample_size)
    
    p_predict_p_samples, _ = p_model.score_samples(p_samples)
    n_predict_p_samples, _ = n_model.score_samples(p_samples)
    kl_pn = p_predict_p_samples.mean() - n_predict_p_samples.mean()

    p_predict_n_samples, _ = p_model.score_samples(n_samples)
    n_predict_n_samples, _ = n_model.score_samples(n_samples)
    kl_np = p_predict_n_samples.mean() - n_predict_n_samples.mean()
    
    return (kl_pn + kl_np)/2
