from twitter import *
import time
import json
import pickle
import pandas as pd
from collections import Counter


## This uses the twitter API. I store my info in a file c'onfig.py'

config = {}
execfile("config.py", config)
twitter = Twitter(auth = OAuth(config["access_key"], config["access_secret"], config["consumer_key"], config["consumer_secret"]))


users = []
m = 1031884008005857280
sources = []

for i in range(4):
    a = twitter.search.tweets(q = '#FreeMariaButina', count=100, max_id = m)
    users+=[p['user']['screen_name'] for p in a['statuses']]
    sources+=[p['source'] for p in a['statuses']]
    ids = [p['id'] for p in a['statuses']]
    print users
    try : m = min(ids)
    except: break
    print[p['created_at'] for p in a['statuses'][-5:]]
    time.sleep(5)

u = Counter(users)
pobots = [a[0] for a in u.most_common(1500) if a[1]>2]   #choosing those that used hashtag 3 times 
len(pobots)  #556 perfect. 


thefile = open('Maria.txt', 'w')
for item in users:
  thefile.write("%s\n" % item)

df9 = pd.DataFrame(columns = ['screen_name', 'retweeted_ids'])


#Next get a bunch of data for each names
MID = 1020000000000000000
SID = 1016000000000000000
def get_timeline(sn):   #This pickles all the trimmed status data, doesn't save a lot of user info
    all_statuses = []
    howmany = 0 
    statuses = twitter.statuses.user_timeline(screen_name=sn, count = 200, max_id = MID, include_rts = True, since_id = SID, trim_user = True, exclude_replies = False)
    all_statuses+=statuses
    if len(all_statuses)==0 : return('empty')
    ids = [k['id'] for k in statuses]
    mid = min(ids)   #this throws error if empty
    time.sleep(.5)
    while(True):
        statuses = twitter.statuses.user_timeline(screen_name=sn, count = 200, max_id = mid, include_rts = True, since_id = SID, trim_user = True, exclude_replies = False)
        all_statuses+=statuses
        ids = [k['id'] for k in statuses]
        if(len(ids))<10: break
        mid = min(ids)
        time.sleep(.4)
        print mid, sn, howmany  
        howmany+=1
    retweets=[]
    for j in all_statuses:
        try: retweets+=[j['retweeted_status']['id']]  #each j is a json object with lots of data.  This simply adds the tweet id if retweeted. 
        except : continue
    df9.loc[len(df9)] = [sn, retweets] 
    
for ss in pobots[115:]:
    try: get_timeline(ss)
    except: 
        print 'error'
        time.sleep(5)
    print pobots.index(ss)
    if pobots.index(ss)%12==2:
        print(twitter.application.rate_limit_status(resources = 'statuses'))
        time.sleep(5)



pickle.dump(df9, open("maria_retweets.pickle","w"))


#now for the metric part 
import pandas as pd
import numpy as np
import scipy
from sklearn import preprocessing
from collections import Counter 
from sklearn import manifold
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pickle
import ot


def lists_to_metric(LL, top):  
    exponent = 3   #We add an a small eta and an exponent of the markov matrix to get ride of 0 transition probability 
    eta = 0.05
    flat_list = [item for sublist in LL for item in sublist]   
    flc = Counter(flat_list)
    common_rt = set([hh[0] for hh in flc.most_common(top)])
    threshold = flc.most_common(top)[top-1][1]
    LS = [set(l) for l in LL]
    LSS = [l.intersection(common_rt) for l in LS] 
    BL = [] 
    labels = list(common_rt)
    M = np.zeros([len(LL),len(labels)])
    for i in range(len(LL)):
        for p in LL[i]:
            if p in labels: M[i,labels.index(p)]=1
    SM = preprocessing.normalize(M, norm='l1')   #Now it's a right stochastic matrix, ie. each row sums to 1.  
    MSM = M.transpose().dot(SM) #This matrix is now the transition probability matrix.
    MSM = preprocessing.normalize(MSM, norm='l1')
    Mexp = np.linalg.matrix_power(MSM,exponent)
    NMSM = preprocessing.normalize(MSM+eta*Mexp, norm='max')
    distance_metric = -np.log(NMSM)/2
    return(NMSM,distance_metric,labels,threshold)

listlist = list(df9.retweeted_ids)
tweet_metric = lists_to_metric(listlist, 200)
## the threshold was 43  so quite a number of retweets here but not quite as many as the other group 
labels = tweet_metric[2]

def convert_hot(l, labels):                 # each user now has a binary vector indexed by retweet, with '1' if user retweeted
    cl = [k in l for k in labels]
    return(np.array([int(c) for c in cl]))
    
retweet_list = list(df9.retweeted_ids)
hotvecs = [convert_hot(i, labels) for i in retweet_list]
total_selected = [a.sum() for a in hotvecs]
df9['hotvecs'] = hotvecs
df9['hot_rts'] = total_selected


#Now subselect those that have at least minimum number of common retweets
df8=df9[df9.hot_rts>6]  #291 of these

list_of_hot_vecs = df8.hotvecs


#here, we use the Wasserstein metric to compare tweet vectors 

def get_W(v1,v2,dsq):  #vector1, vector2, and the metric to be used
    L = len(v1)
    inds=[i for i in range(L) if (v1[i]+v2[i])>0]
    lv1 = np.asfarray(v1[inds])
    lv2 = np.asfarray(v2[inds])
    lv1 = lv1/lv1.sum()
    lv2 = lv2/lv2.sum()
    ldsq = dsq[:,inds][inds]
    try : mm = ot.emd(lv1,lv2,list(ldsq))
    except : return(25)  #I believe the errors are for when vecs are empty
    value = (mm*ldsq).sum()
    return(value)


metric_on_users = [[get_W(p,q, tweet_metric[1]) for p in list_of_hot_vecs] for q in list_of_hot_vecs]
#took about a minute

#floyd warshall may or may not do anything but run it anyways 
fw = scipy.sparse.csgraph.floyd_warshall(metric_on_users)
sfw = (fw+fw.transpose())/2
view_metric = pd.DataFrame(metric_on_users, columns = df8.screen_name)
view_metric.index=df8.screen_name



mds = manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='precomputed')
pos = mds.fit(sfw).embedding_

plt.scatter(pos[:,0], pos[:,1]) 
plt.show()

#this is the original 2d projection, without using an Ricci methods.  

clust =  KMeans(n_clusters=2)
clust.fit(pos)
clust.labels_
LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'b',
                   }
label_color = [LABEL_COLOR_MAP[l] for l in clust.labels_]
plt.scatter(pos[:,0], pos[:,1],c=label_color )
view_metric['cluster'] = clust.labels_
view_metric.to_csv("view_metric.csv")  #this output the metric to a csv for easy perusing


# the following is more or less copied from https://github.com/micah541/Ricci


import numexpr as ne
import scipy.linalg as sl
def add_AB_to_C(A, B, C):
    """
    Compute C += AB in-place.
    This uses gemm from whatever BLAS is available.
    MKL requires Fortran ordered arrays to avoid copies.
    Hence we work with transpositions of default c-style arrays.
    This function throws error if computation is not in-place.
    """
    gemm = sl.get_blas_funcs("gemm", (A, B, C))
    assert np.isfortran(C.T) and np.isfortran(A.T) and np.isfortran(B.T)
    D = gemm(1.0, B.T, A.T, beta=1, c=C.T, overwrite_c=1)
    assert D.base is C or D.base is C.bas


def applyRicci(sqdist, eta, T, Ricci, mode='sym'):
    """
    Apply coarse Ricci to a squared distance matrix.
    Can handle symmetric, max, and nonsymmetric modes.
    Gaussian localizing kernel is used with T as variance parameter.
    """
    if 'sym' in mode:
        ne.evaluate('sqdist - (eta/2)*exp(-sqdist/T)*(Ricci+RicciT)',
                    global_dict={'RicciT': Ricci.T}, out=sqdist)
    elif 'max' in mode:
        ne.evaluate(
            'sqdist - eta*exp(-sqdist/T)*where(Ricci<RicciT, RicciT, Ricci)',
            global_dict={'RicciT': Ricci.T}, out=sqdist)
    elif 'dumb' in mode:
        ne.evaluate('sqdist*(1 - eta*exp(-sqdist/T))', out=sqdist)
    else:
        ne.evaluate('sqdist - eta*exp(-sqdist/T)*Ricci',
                    global_dict={'RicciT': Ricci.T}, out=sqdist)


def coarseRicci(L, sqdist, R, temp1=None, temp2=None):
    """
    Fully optimized Ricci matrix computation.
    Requires 7 matrix multiplications and many entrywise operations.
    Only 2 temporary matrices are needed, and can be provided as arguments.
    Uses full gemm functionality to avoid creating intermediate matrices.
    R is the output array, while temp1 and temp2 are temporary matrices.
    """
    D = sqdist
    if temp1 is None:
        temp1 = np.zeros(sqdist.shape)
    if temp2 is None:
        temp2 = np.zeros(sqdist.shape)
    A = temp1
    B = temp2
    # this C should not exist
    B = ne.evaluate("D*D/4.0")
    L.dot(B, out=A)
    L.dot(D, out=B)
    ne.evaluate("A-D*B", out=A)
    L.dot(A, out=R)
    # the first two terms done
    L.dot(B, out=A)
    ne.evaluate("R+0.5*(D*A+B*B)", out=R)
    # Now R contains everything under overline
    ne.evaluate("R+dR-0.5*dA*D-dB*B",
                global_dict={'dA': np.diag(A).copy()[:, None],
                             'dB': np.diag(B).copy()[:, None],
                             'dR': np.diag(R).copy()[:, None]}, out=R)
    # Now R contains all but two matrix products from line 2
    L.dot(L, out=A)
    ne.evaluate("L*BT-0.5*A*D", global_dict={'BT': B.T}, out=A)
    add_AB_to_C(A, D, R)
    ne.evaluate("L*D", out=A)
    add_AB_to_C(A, B, R)
    # done!
    np.fill_diagonal(R, 0.0)



def computeLaplaceMatrix(sqdist, t, L):
    """ Compute heat approximation to Laplacian matrix using logarithms. """
    lt = np.log(2.0 / t)
    ne.evaluate('sqdist / (-2.0 * t)', out=L)
    # sqdist is nonnegative, with 0 on the diagonal
    # so the largest element of each row of L is 0
    # no logsumexp needed
    density = ne.evaluate("sum(exp(L), axis=1)")[:, None]
    ne.evaluate("log(density)", out=density)
    # sum in rows must be 1, except for 2/t factor
    ne.evaluate('exp(L - density + lt)', out=L)
    L[np.diag_indices(len(L))] -= 2.0 / t

#Now in order to run "Ricci Flow"....

L = np.zeros(sfw.shape)
R = np.zeros(sfw.shape)

computeLaplaceMatrix(sfw, 1, L)  #note the way we had this set up (with Bartek Suideja) it was more efficient to pass to matrices to the function and have them modified.  We (mostly just Bartek) worked really hard to get the algorithm running efficiently. 
coarseRicci(L, sfw, R)
m = sfw.copy()
for ij in range(1500):
    applyRicci(m, 0.001, 1, R)


mds = manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='precomputed')
pos = mds.fit(m).embedding_

plt.scatter(pos[:,0], pos[:,1]) 
plt.show()


clust =  KMeans(n_clusters=2)
clust.fit(pos)
clust.labels_
LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'b',
                   }
label_color = [LABEL_COLOR_MAP[l] for l in clust.labels_]
plt.scatter(pos[:,0], pos[:,1],c=label_color )



def lists_to_metric(LL, top):  
    exponent = 3   #We add an a small eta and an exponent of the markov matrix to get ride of 0 transition probability 
    eta = 0.05
    flat_list = [item for sublist in LL for item in sublist]   
    flc = Counter(flat_list)
    common_rt = set([hh[0] for hh in flc.most_common(top)])
    threshold = flc.most_common(top)[top-1][1]
    LS = [set(l) for l in LL]
    LSS = [l.intersection(common_rt) for l in LS] 
    BL = [] 
    labels = list(common_rt)
    M = np.zeros([len(LL),len(labels)])
    for i in range(len(LL)):
        for p in LL[i]:
            if p in labels: M[i,labels.index(p)]=1
    SM = preprocessing.normalize(M, norm='l1')   #Now it's a right stochastic matrix, ie. each row sums to 1.  
    MSM = M.transpose().dot(SM) #This matrix is now the transition probability matrix.
    MSM = preprocessing.normalize(MSM, norm='l1')
    Mexp = np.linalg.matrix_power(MSM,exponent)
    NMSM = preprocessing.normalize(MSM+eta*Mexp, norm='max')
    distance_metric = -np.log(NMSM)/2
    return(NMSM,distance_metric,labels,threshold)

listlist = list(df9.retweeted_ids)
tweet_metric = lists_to_metric(listlist, 600)
## the threshold was 52  so quite a number of retweets here 
labels = tweet_metric[2]

def convert_hot(l, labels):                 # each user now has a binary vector indexed by retweet, with '1' if user retweeted
    cl = [k in l for k in labels]
    return(np.array([int(c) for c in cl]))
    
retweet_list = list(df9.retweeted_ids)
hotvecs = [convert_hot(i, labels) for i in retweet_list]
total_selected = [a.sum() for a in hotvecs]
df9['hotvecs'] = hotvecs
df9['hot_rts'] = total_selected


#Now subselect those that have at least minimum number of common retweets
df8=df9[df9.hot_rts>9]  #948 of these

list_of_hot_vecs = df8.hotvecs


#here, we use the Wasserstein metric to compare tweet vectors 

def get_W(v1,v2,dsq):  #vector1, vector2, and the metric to be used
    L = len(v1)
    inds=[i for i in range(L) if (v1[i]+v2[i])>0]
    lv1 = np.asfarray(v1[inds])
    lv2 = np.asfarray(v2[inds])
    lv1 = lv1/lv1.sum()
    lv2 = lv2/lv2.sum()
    ldsq = dsq[:,inds][inds]
    try : mm = ot.emd(lv1,lv2,list(ldsq))
    except : return(25)  #I believe the errors are for when vecs are empty
    value = (mm*ldsq).sum()
    return(value)


metric_on_users = [[get_W(p,q, tweet_metric[1]) for p in list_of_hot_vecs] for q in list_of_hot_vecs]
#took about ten minutes


#floyd warshall may or may not do anything but run it anyways 
fw = scipy.sparse.csgraph.floyd_warshall(metric_on_users)
sfw = (fw+fw.transpose())/2
view_metric = pd.DataFrame(metric_on_users, columns = df8.screen_name)
view_metric.index=df8.screen_name



mds = manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='precomputed')
pos = mds.fit(sfw).embedding_

plt.scatter(pos[:,0], pos[:,1]) 
plt.show()

#this is the original 2d projection, without using an Ricci methods.  

clust =  KMeans(n_clusters=2)
clust.fit(pos)
clust.labels_
LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'b',
                   }
label_color = [LABEL_COLOR_MAP[l] for l in clust.labels_]
plt.scatter(pos[:,0], pos[:,1],c=label_color )
view_metric['cluster'] = clust.labels_
view_metric.to_csv("view_metric.csv")  #this output the metric to a csv for easy perusing


# the following is more or less copied from https://github.com/micah541/Ricci


import numexpr as ne
import scipy.linalg as sl
def add_AB_to_C(A, B, C):
    """
    Compute C += AB in-place.
    This uses gemm from whatever BLAS is available.
    MKL requires Fortran ordered arrays to avoid copies.
    Hence we work with transpositions of default c-style arrays.
    This function throws error if computation is not in-place.
    """
    gemm = sl.get_blas_funcs("gemm", (A, B, C))
    assert np.isfortran(C.T) and np.isfortran(A.T) and np.isfortran(B.T)
    D = gemm(1.0, B.T, A.T, beta=1, c=C.T, overwrite_c=1)
    assert D.base is C or D.base is C.bas


def applyRicci(sqdist, eta, T, Ricci, mode='sym'):
    """
    Apply coarse Ricci to a squared distance matrix.
    Can handle symmetric, max, and nonsymmetric modes.
    Gaussian localizing kernel is used with T as variance parameter.
    """
    if 'sym' in mode:
        ne.evaluate('sqdist - (eta/2)*exp(-sqdist/T)*(Ricci+RicciT)',
                    global_dict={'RicciT': Ricci.T}, out=sqdist)
    elif 'max' in mode:
        ne.evaluate(
            'sqdist - eta*exp(-sqdist/T)*where(Ricci<RicciT, RicciT, Ricci)',
            global_dict={'RicciT': Ricci.T}, out=sqdist)
    elif 'dumb' in mode:
        ne.evaluate('sqdist*(1 - eta*exp(-sqdist/T))', out=sqdist)
    else:
        ne.evaluate('sqdist - eta*exp(-sqdist/T)*Ricci',
                    global_dict={'RicciT': Ricci.T}, out=sqdist)


def coarseRicci(L, sqdist, R, temp1=None, temp2=None):
    """
    Fully optimized Ricci matrix computation.
    Requires 7 matrix multiplications and many entrywise operations.
    Only 2 temporary matrices are needed, and can be provided as arguments.
    Uses full gemm functionality to avoid creating intermediate matrices.
    R is the output array, while temp1 and temp2 are temporary matrices.
    """
    D = sqdist
    if temp1 is None:
        temp1 = np.zeros(sqdist.shape)
    if temp2 is None:
        temp2 = np.zeros(sqdist.shape)
    A = temp1
    B = temp2
    # this C should not exist
    B = ne.evaluate("D*D/4.0")
    L.dot(B, out=A)
    L.dot(D, out=B)
    ne.evaluate("A-D*B", out=A)
    L.dot(A, out=R)
    # the first two terms done
    L.dot(B, out=A)
    ne.evaluate("R+0.5*(D*A+B*B)", out=R)
    # Now R contains everything under overline
    ne.evaluate("R+dR-0.5*dA*D-dB*B",
                global_dict={'dA': np.diag(A).copy()[:, None],
                             'dB': np.diag(B).copy()[:, None],
                             'dR': np.diag(R).copy()[:, None]}, out=R)
    # Now R contains all but two matrix products from line 2
    L.dot(L, out=A)
    ne.evaluate("L*BT-0.5*A*D", global_dict={'BT': B.T}, out=A)
    add_AB_to_C(A, D, R)
    ne.evaluate("L*D", out=A)
    add_AB_to_C(A, B, R)
    # done!
    np.fill_diagonal(R, 0.0)



def computeLaplaceMatrix(sqdist, t, L):
    """ Compute heat approximation to Laplacian matrix using logarithms. """
    lt = np.log(2.0 / t)
    ne.evaluate('sqdist / (-2.0 * t)', out=L)
    # sqdist is nonnegative, with 0 on the diagonal
    # so the largest element of each row of L is 0
    # no logsumexp needed
    density = ne.evaluate("sum(exp(L), axis=1)")[:, None]
    ne.evaluate("log(density)", out=density)
    # sum in rows must be 1, except for 2/t factor
    ne.evaluate('exp(L - density + lt)', out=L)
    L[np.diag_indices(len(L))] -= 2.0 / t

#Now in order to run "Ricci Flow"....

L = np.zeros(sfw.shape)
R = np.zeros(sfw.shape)

computeLaplaceMatrix(sfw, 1, L)  #note the way we had this set up (with Bartek Suideja) it was more efficient to pass to matrices to the function and have them modified.  We (mostly just Bartek) worked really hard to get the algorithm running efficiently. 
coarseRicci(L, sfw, R)
m = sfw.copy()
for ij in range(1000):
    applyRicci(m, 0.001, 1, R)

m = np.maximum.reduce([m, np.zeros(m.shape)])



mds = manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='precomputed')
pos = mds.fit(m).embedding_

plt.scatter(pos[:,0], pos[:,1]) 
plt.show()


clust =  KMeans(n_clusters=2)
clust.fit(pos)
clust.labels_
LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'b',
                   }
label_color = [LABEL_COLOR_MAP[l] for l in clust.labels_]
plt.scatter(pos[:,0], pos[:,1],c=label_color )


