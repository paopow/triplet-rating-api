import numpy as np
import pandas as pd
import scipy.optimize as optimize
import pickle, sys
from itertools import combinations
from operator import itemgetter



def distance(x1,x2):
    return np.sqrt(np.dot(x1-x2, x1-x2))

def odd_to_sim(comparison):
        comp1 = [comparison[0], comparison[2], comparison[1]]
        comp2 = [comparison[2], comparison[0], comparison[1]]
        return [comp1, comp2]


def tste_kernel(positions, alpha):
    """Compute Student-t kernel for given ``positions`` and ``alpha`` value
    """
    X = positions
    Xt = X.T

    sum_X = np.sum(X*X, axis=1)
    sum_X_dup = np.repeat(sum_X, len(sum_X)).reshape(len(sum_X), len(sum_X))
    sum_X_dup_t = sum_X_dup.T
    K = (1 + (sum_X_dup + sum_X_dup_t - 2*np.dot(X,Xt))/alpha)**(-0.5*(alpha + 1))

    return K


def tste_cost(X, comparisons, K, l, uselog=True):
    """Calculate the cost of current positions ``X` with given ``comparisons``
        Returns: cost of the positions and the matrix of probability of each triplet
    """
    P = np.array([K[a,b]/(K[a,b] + K[a,c]) for (a, b, c) in comparisons])
    if uselog:
        #logP = np.max(np.log(P), np.finfo(float).min)
        logP = np.log(P) # Let's hope that there is not super small log value

        C = -sum(np.log(P)) + l*sum(np.ravel(X**2))
    else:
        C = -sum(P) + l*sum(np.ravel(X**2))

    return C, P



def tste_grad(X, comparisons, alpha, l=0, uselog=True):
        """Gradient of t-Distributed Stochastic Triplet Embedding.

            Returns: C cost and dC gradient
        """
        # compute Student-t kernel
        K = tste_kernel(X, alpha)

        # compute value of cost function
        C, P = tste_cost(X, comparisons, K, l, uselog)

        def term(a,b, i):
            return K[a,b]*(X[a,i]-X[b,i])

        # compute gradient
        dC = np.zeros(X.shape)


        const = (alpha + 1)/alpha
        triplets_array = np.array(comparisons)
        triplets_flat = np.reshape(triplets_array,(triplets_array.size,1))
        #TODO: vectorize this operation to speed it up
        for i in range(X.shape[1]):
            val = np.array([[term(a,b,i)-term(a,c,i), -term(a,b,i), term(a,c,i)] for a,b,c in comparisons])
            if uselog:
                val = -const*(1-np.reshape(P,(P.shape[0],1)))*val
            else:
                val = -const*np.reshape(P*(1-P), (P.shape[0],1))*val



            val_flat = np.reshape(val, (val.size,1))
            triplet_val = np.concatenate((triplets_flat, val_flat), axis=1)
            df = pd.DataFrame(triplet_val, columns=('id','value'))
            df_groupby = df.groupby('id')
            df_group_sum = df_groupby.sum()
            df_index_int = [int(index) for index,value in df_group_sum.value.iteritems()]
            idxed_df = df_group_sum.set_index([df_index_int])
            idx = idxed_df.index

            dC[idx,i:(i+1)] = dC[idx,i:(i+1)] + df_group_sum.values


        dC = -dC + 2*l*X

        return C, dC



def tste_impute(is_landmarks, comparisons, positions, mapping, alpha, l=0, uselog=True, verbose=False, grad_func = tste_grad, cost_func = tste_cost):
    """Calculate best euclidean embedding based on given comparison using t-STE.

        Returns: best position matrix
    """
    C = np.finfo(float).max
    num_item = positions.shape[0]
    num_dim = positions.shape[1]
    # print comparisons
    # print mapping
    # t_comparisons = comparisons
    t_comparisons = [(mapping.index(a), mapping.index(b), mapping.index(c)) for a,b,c in comparisons]
    num_triplet = len(t_comparisons)

    max_iter = 1000
    tol = 1e-7
    eta = 2.0  # learnning rate
    best_C = np.finfo(float).max
    X = positions
    best_X = positions

    num_increment = 0

    for i in range(max_iter):
        # compute value of slack variables, cost function, and gradient
        old_C = C
        C, G = tste_grad(X, t_comparisons, alpha, l, uselog)

        mask = np.repeat(np.reshape(is_landmarks,(num_item,1)), num_dim, axis=1)
        G = G*(1-mask)

        if C < best_C:
            best_C = C
            best_X = X

        # perform gradient update
        X = X - (eta / num_triplet)*num_item*G

        # update learning rate
        if old_C > C + tol:
            num_increment = 0
            eta = eta * 1.01
        else:
            num_increment = num_increment + 1
            eta = eta * 0.5

        if num_increment >= 5:
            break;

        if verbose:
            if i%10 == 0:
                num_violate = sum([1 for comp in t_comparisons if distance(X[comp[0]], X[comp[1]]) > distance(X[comp[0]], X[comp[2]])])
                progress_rep = ['Iteration:', str(i),
                                'error:', str(C),
                                'num constraints:', str(num_violate*1.0/num_triplet)]

                print ' '.join(progress_rep)

    return best_X


def what_to_ask(X, comparisons, mapping, is_landmarks, item_to_ask, num_triplet_wanted=1):
    """Give the next triplet to ask to get more information about ``item_to_ask`` using a simple heurisitc.
        Return a tuple of triplet to ask
    """
    item_to_ask_idx = mapping.index(item_to_ask)
    num_item = X.shape[0]
    p1 = X[item_to_ask_idx,:]
    mask = np.array(is_landmarks)
    X_lm = X[mask,:]
    diff_distances = [((b, c), abs(distance(p1,X_lm[b])-distance(p1,X_lm[c]))) for b,c in combinations(range(sum(is_landmarks)),2)]
    triplets = []
    while len(diff_distances) > 0 and len(triplets)<num_triplet_wanted:
        min_pair, min_diff =  min(diff_distances, key=itemgetter(1))

        b, c = mapping[min_pair[0]], mapping[min_pair[1]]
        ask_before = [(item_to_ask in _ and b in _ and c in _) for _ in comparisons]
        diff_distances.remove((min_pair,min_diff))
        if sum(ask_before) == 0 and (b != item_to_ask) and (c != item_to_ask):
            triplets.append((item_to_ask, b, c))
            comparisons.append([item_to_ask,b,c])
    # TODO: What about when everything is already asked? Need to throw and exception
    return triplets

def init_random_pos(num_item,landmark_pos=None, num_dim=2):
    X = np.random.normal(0,1,(num_item,num_dim))
    if not (landmark_pos==None):
        M = np.concatenate([landmark_pos, X])
    else:
        M = X
    return M


def calculate_indv_diversity(item_list,M, is_landmarks):
    diverse_score = 0
    for i in range(len(M)):
        item_list[i]['x'] = M[i][0]
        item_list[i]['y'] = M[i][1]

    part_pairs = combinations(range(len(item_list)),2)
    distances = [distance(M[p[0]],M[p[1]]) for p in part_pairs if (not is_landmarks[p[0]] and not is_landmarks[p[1]])]
    if not len(distances) == 0:
        diverse_score = np.mean(distances)

    return diverse_score

def calculate_indv_uniqueness(item_idx, M, num_neighbor):
    distances = []
    for i in range(len(M)):
        if i != item_idx:
            d = distance(M[item_idx],M[i])
            distances.append(d)
    distances = sorted(distances, reverse=True)
    avg = np.mean(distances[:num_neighbor])
    return avg

def calculate_set_uniqueness(item_idx, M_list, num_neighbor):
    dist = [calculate_indv_uniqueness(item_idx,M,num_neighbor) for M in M_list]
    return np.mean(dist)


def scale_idea_space(item_pos, box_range=[-2.,2.]):
    max_xy = np.max(item_pos, axis=0)
    min_xy = np.min(item_pos, axis=0)

    scale = (box_range[1]-(box_range[0]))/(max_xy - min_xy)

    return scale*(item_pos-min_xy) - box_range[1]

def bound_idea_space(item_pos, box_range=[-2.,2.]):
    upper_bound = np.array([[box_range[1],box_range[1]] for _ in range(item_pos.shape[0])])
    lower_bound = np.array([[box_range[0],box_range[0]] for _ in range(item_pos.shape[0])])
    item_pos = np.minimum(item_pos, upper_bound)
    item_pos = np.maximum(item_pos, lower_bound)
    return item_pos

def get_mean_distance(item_idx, M):
    part_pairs = combinations(item_idx,2)
    distances = [distance(M[p[0]],M[p[1]]) for p in part_pairs]
    return np.mean(distances)

def get_mean_distance_id(item_id, M, mapping):
    item_idx = [mapping.index(i) for i in item_id]
    return get_mean_distance(item_idx,M)


