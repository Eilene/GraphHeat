# -*- coding:UTF-8 -*-
from sklearn.preprocessing import normalize
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import warnings
import random
warnings.filterwarnings("ignore")

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_wiki_data(dataset_str="wiki",alldata=False):
    names = ['x', 'y', 'graph']
    objects = []

    for i in range(len(names)):
        with open("data/wiki-data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, graph = tuple(objects)

    class_index = {}
    for i in range(y.shape[1]):
        class_index[i] = []

    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if(y[i,j]!=0):
                class_index[j].append(i)

    train_index = []
    small_num = 5
    big_num = 20
    for i in range(y.shape[1]):
        if(len(class_index[i])<big_num):
            train_index.extend(random.sample(class_index[i],small_num))
        else:
            train_index.extend(random.sample(class_index[i], big_num))

    index = [i for i in range(y.shape[0])]

    now_index = list(set(index).difference(set(train_index)))
    val_index = random.sample(now_index,500)
    now_index = list(set(now_index).difference(set(val_index)))
    test_index = random.sample(now_index,500)

    features = x.tolil()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = y

    train_mask = sample_mask(train_index, labels.shape[0])
    val_mask = sample_mask(val_index, labels.shape[0])
    test_mask = sample_mask(test_index, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return labels,adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_data(dataset_str,alldata = True):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    if(alldata == True):
        features = sp.vstack((allx, tx)).tolil()
        labels = np.vstack((ally,ty))
        num = labels.shape[0]
        idx_train = range(num/5*3)
        idx_val = range(num/5*3, num/5*4)
        idx_test = range(num/5*4, num)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    return labels,adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    # print rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv,0)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt,0)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # return adj_normalized
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    # print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    # for i in range(len(t_k)):
    #     t_k[i] = normalize(t_k[i],norm='l1',axis=1)
    # # return t_k
    return sparse_to_tuple(t_k)

def to_weighting_function(order,threshold,mask, data_normalize,weight_normalize, adj,Weight,sparse_ness=True):
    # print Weight

    # adj_normalized = normalize_adj(adj)
    # laplacian = sp.eye(adj.shape[0]) - adj_normalized
    # largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    # scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    # mask 需要保留自身，为了使得自身的影响和其他可比，对adj加单位阵
    if(mask == True):
        if(order == 0):
            adj = sp.csr_matrix(np.eye(adj.shape[0]))
        elif(order == 1):
            adj = adj + sp.csr_matrix(np.eye(adj.shape[0]))
        elif(order == 2):
            second_order = adj.dot(adj)
            adj = adj + sp.csr_matrix(np.eye(adj.shape[0])) + second_order
            del second_order
        elif(order == 3):
            second_order = adj.dot(adj)
            third_order = adj.dot(second_order)
            adj = adj + sp.csr_matrix(np.eye(adj.shape[0])) + second_order + third_order
            del second_order,third_order
        elif(order == 4):
            second_order = adj.dot(adj)
            third_order = adj.dot(second_order)
            four_order = adj.dot(third_order)
            adj = adj + sp.csr_matrix(np.eye(adj.shape[0])) + second_order + third_order + four_order
            del second_order, third_order,four_order

    add_matrix = None
    # 做完乘法低于1e-10，则可以先删除
    print "Wavelet before sparseness: ", len(np.nonzero(Weight)[0])
    if(sparse_ness):
        Weight[Weight < 1e-5] = 0.0 # 乘完小于1e-10
    print "Wavelet after sparseness: ",len(np.nonzero(Weight)[0])

    for i in range(Weight.shape[1]):
        tmp = Weight[:, i]
        tmp = np.reshape(tmp, [Weight[:, i].shape[0], 1])
        # s = sp.csr_matrix(tmp).dot(sp.csr_matrix(np.transpose(tmp)))
        s = np.dot(tmp, np.transpose(tmp))

        # 在每个weighting function 上先粗略筛选
        # for i in range(s.data.shape[0]):
        #     if(s.data[i] < 1e-10):
        #         s.data[i] = 0.0
        #     s[s < 1e-10] = 0

        # mask
        if (mask == True):
            s = adj.multiply(s)
            # s = adj * s

        # normalize,每一个weighting func的normalize
        if (data_normalize == True):
            s = normalize(s, norm='l1', axis=1)

        # t_k.append(s)
        if(i == 0):
            add_matrix = s
        else:
            # add_matrix = add_matrix.__add__(s)
            add_matrix = add_matrix + s

        if(i%500 ==0):
            print i
            print len(np.nonzero(add_matrix)[0])


    if (sparse_ness):
        # for i in range(add_matrix.data.shape[0]):
        #     if(add_matrix.data[i] < threshold):
        #         add_matrix.data[i] = 0.0
        add_matrix[add_matrix < threshold] = 0

    add_matrix = sp.csr_matrix(add_matrix)
    add_matrix.setdiag([0.0] * add_matrix.shape[0])
    if(weight_normalize):
        add_matrix = normalize(add_matrix,axis=1,norm='l1')
    # np.set_printoptions(threshold=np.inf)
    print add_matrix
    print len(np.nonzero(add_matrix)[0])

    t_k = [add_matrix]
    t_k.append(sp.eye(adj.shape[0]))
    # t_k.append(scaled_laplacian)

    return t_k

def wave(dataset,order,threshold,s,mask,data_normalize,weight_normalize,adj,laplacian_normalize,sparse_ness = False):
    from weighting_func import laplacian,fourier,weight_wavelet
    # wavelet community
    # adj = adj_matrix()
    L = laplacian(adj, normalized=laplacian_normalize)
    lamb, U = fourier(dataset,L)
    # print np.dot(np.dot(U,np.diag(lamb)),np.transpose(U))
    # U 列 eigen function
    Weight = weight_wavelet(s,lamb, U)
    del U,lamb
    t_k = to_weighting_function(order,threshold,mask,data_normalize,weight_normalize,adj,Weight,sparse_ness=sparse_ness)
    # return  U,Weight, t_k
    return sparse_to_tuple(t_k)

def to_gcn_weighting_function(order,threshold,mask, data_normalize,weight_normalize, adj,Weight,sparse_ness=True):
    # print Weight

    # adj_normalized = normalize_adj(adj)
    # laplacian = sp.eye(adj.shape[0]) - adj_normalized
    # largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    # scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    # mask 需要保留自身，为了使得自身的影响和其他可比，对adj加单位阵
    if(mask == True):
        if(order == 0):
            adj = sp.csr_matrix(np.eye(adj.shape[0]))
        elif(order == 1):
            adj = adj + sp.csr_matrix(np.eye(adj.shape[0]))
        elif(order == 2):
            second_order = adj.dot(adj)
            adj = adj + sp.csr_matrix(np.eye(adj.shape[0])) + second_order
            del second_order
        elif(order == 3):
            second_order = adj.dot(adj)
            third_order = adj.dot(second_order)
            adj = adj + sp.csr_matrix(np.eye(adj.shape[0])) + second_order + third_order
            del second_order,third_order
        elif(order == 4):
            second_order = adj.dot(adj)
            third_order = adj.dot(second_order)
            four_order = adj.dot(third_order)
            adj = adj + sp.csr_matrix(np.eye(adj.shape[0])) + second_order + third_order + four_order
            del second_order, third_order,four_order

    add_matrix = None
    # 做完乘法低于1e-10，则可以先删除
    print "Wavelet before sparseness: ", len(np.nonzero(Weight)[0])
    if(sparse_ness):
        Weight[Weight < 1e-5] = 0.0 # 乘完小于1e-10
    print "Wavelet after sparseness: ",len(np.nonzero(Weight)[0])

    for i in range(Weight.shape[1]):
        tmp = Weight[:, i]
        tmp = np.reshape(tmp, [Weight[:, i].shape[0], 1])
        # s = sp.csr_matrix(tmp).dot(sp.csr_matrix(np.transpose(tmp)))
        s = np.dot(tmp, np.transpose(tmp))

        # 在每个weighting function 上先粗略筛选
        # for i in range(s.data.shape[0]):
        #     if(s.data[i] < 1e-10):
        #         s.data[i] = 0.0
        #     s[s < 1e-10] = 0

        # mask
        if (mask == True):
            s = adj.multiply(s)
            # s = adj * s

        # normalize,每一个weighting func的normalize
        if (data_normalize == True):
            s = normalize(s, norm='l1', axis=1)

        # t_k.append(s)
        if(i == 0):
            add_matrix = s
        else:
            # add_matrix = add_matrix.__add__(s)
            add_matrix = add_matrix + s

        if(i%500 ==0):
            print i
            print len(np.nonzero(add_matrix)[0])


    if (sparse_ness):
        # for i in range(add_matrix.data.shape[0]):
        #     if(add_matrix.data[i] < threshold):
        #         add_matrix.data[i] = 0.0
        add_matrix[add_matrix < threshold] = 0

    add_matrix = np.multiply(add_matrix, (np.ones(shape=adj.shape) - adj))

    add_matrix = sp.csr_matrix(add_matrix)
    add_matrix.setdiag([0.0] * add_matrix.shape[0])
    if(weight_normalize):
        add_matrix = normalize(add_matrix,axis=1,norm='l1')
    # np.set_printoptions(threshold=np.inf)
    print add_matrix
    print len(np.nonzero(add_matrix)[0])


    # printsd

    t_k = [add_matrix]
    t_k.append(preprocess_adj(adj))
    # t_k.append(sp.eye(adj.shape[0]))
    # t_k.append(scaled_laplacian)

    return t_k

def wave_gcn(order,threshold,s,mask,data_normalize,weight_normalize,adj,laplacian_normalize,sparse_ness = False):
    from weighting_func import laplacian,fourier,weight_wavelet
    # wavelet community
    # adj = adj_matrix()
    L = laplacian(adj, normalized=laplacian_normalize)
    lamb, U = fourier(L)
    # print np.dot(np.dot(U,np.diag(lamb)),np.transpose(U))
    # U 列 eigen function
    Weight = weight_wavelet(s,lamb, U)
    t_k = to_gcn_weighting_function(order,threshold,mask,data_normalize,weight_normalize,adj,Weight,sparse_ness=sparse_ness)
    # return  U,Weight, t_k
    return sparse_to_tuple(t_k)

def wavelet_basis_chebyshev(dataset,adj,s,laplacian_normalize,sparse_ness,threshold,weight_normalize):
    from weighting_func import laplacian, fourier, weight_wavelet, weight_wavelet_inverse
    from pygsp import graphs, filters
    G = graphs.Graph(adj)
    taus = [s]
    g = filters.Heat(G, taus)
    signal_matrix = np.identity(G.N)
    Weight = g.filter(signal_matrix, method='chebyshev', order=50)

    # taus = [-s]
    # g = filters.Heat(G, taus)
    # signal_matrix = np.identity(G.N)
    # inverse_Weight = g.filter(signal_matrix, method='chebyshev', order=30)

    # 逆变换和正变换s不能共享，否则不可逆
    L = laplacian(adj, normalized=laplacian_normalize)
    lamb, U = fourier(dataset, L)
    # Weight = weight_wavelet(s, lamb, U)
    inverse_Weight = weight_wavelet_inverse(1.0, lamb, U)
    del U, lamb


    if (sparse_ness):
        Weight[Weight < threshold] = 0.0
        inverse_Weight[inverse_Weight < threshold] = 0.0
    print len(np.nonzero(Weight)[0])
    print len(np.nonzero(inverse_Weight)[0])

    if (weight_normalize == True):
        Weight = normalize(Weight, norm='l1', axis=1)
        inverse_Weight = normalize(inverse_Weight, norm='l1', axis=1)

    Weight = sp.csr_matrix(Weight)
    inverse_Weight = sp.csr_matrix(inverse_Weight)
    print Weight,inverse_Weight
    t_k = [inverse_Weight, Weight]
    # t_k = [Weight]
    return sparse_to_tuple(t_k)

def wave_basis(dataset,adj,s,laplacian_normalize,sparse_ness,threshold,weight_normalize):
    from weighting_func import laplacian,fourier,weight_wavelet,weight_wavelet_inverse
    L = laplacian(adj,normalized=laplacian_normalize)
    lamb, U = fourier(dataset,L)
    Weight = weight_wavelet(s,lamb,U)
    inverse_Weight = weight_wavelet_inverse(s,lamb,U)
    del U,lamb

    # positive_num = Weight[Weight > 0.0].shape[0]
    # nonzero_num = len(np.nonzero(Weight)[0])
    # print positive_num,nonzero_num
    # positive_num = inverse_Weight[inverse_Weight > 0.0].shape[0]
    # nonzero_num = len(np.nonzero(inverse_Weight)[0])
    # print positive_num, nonzero_num
    # inverse_Weight[inverse_Weight > -threshold] = 0.0
    # print len(np.nonzero(inverse_Weight)[0])
    # Weight[Weight > -threshold] = 0.0
    # print len(np.nonzero(Weight)[0])

    if (sparse_ness):
        Weight[Weight < threshold] = 0.0
        inverse_Weight[inverse_Weight < threshold] = 0.0
    print len(np.nonzero(Weight)[0])
    print len(np.nonzero(inverse_Weight)[0])

    if (weight_normalize == True):
        Weight = normalize(Weight, norm='l1', axis=1)
        inverse_Weight = normalize(inverse_Weight, norm='l1', axis=1)

    Weight = sp.csr_matrix(Weight)
    inverse_Weight = sp.csr_matrix(inverse_Weight)
    print Weight
    t_k = [inverse_Weight,Weight]
    # t_k = [Weight,inverse_Weight]
    # t_k = [Weight]
    return sparse_to_tuple(t_k)

def spectral_basis(dataset,adj,s,laplacian_normalize,sparse_ness,threshold,weight_normalize):
    from weighting_func import laplacian,fourier,weight_wavelet,weight_wavelet_inverse
    L = laplacian(adj,normalized=laplacian_normalize)
    lamb, U = fourier(dataset,L)

    U = sp.csr_matrix(U)
    # U_transpose = sp.csr_matrix(np.transpose(U))
    t_k = [U]
    return sparse_to_tuple(t_k)

def NMF_community(order,threshold,mask,data_normalize,adj, community_num=100,sparse_ness = True):
    from weighting_func import NMF
    # NMF community
    # adj = adj_matrix()
    W = NMF(adj, compon=community_num)
    t_k = to_weighting_function(order,threshold,mask,data_normalize,adj,W,sparse_ness = sparse_ness)
    return sparse_to_tuple(t_k)

def wavelet_origin(dataset,order,threshold,s,mask,data_normalize,adj,laplacian_normalize,sparse_ness = False):
    from weighting_func import laplacian, fourier, weight_wavelet
    L = laplacian(adj, normalized=laplacian_normalize)
    lamb, U = fourier(dataset,L)

    Weight = weight_wavelet(s, lamb, U)
    del lamb,U,L

    if (sparse_ness):
        # for i in range(Weight.data.shape[0]):
        #     if(Weight.data[i] < threshold):
        #         Weight.data[i] = 0.0
        Weight[Weight < threshold] = 0

    Weight = sp.csr_matrix(Weight)
    Weight.setdiag([0.0] * Weight.shape[0])
    if (data_normalize == True):
        Weight = normalize(Weight, norm='l1', axis=1)

    t_k = [Weight]
    t_k.append(sp.eye(adj.shape[0]))
    return sparse_to_tuple(t_k)

def wavelet_gcn_origin(order,threshold,s,mask,data_normalize,adj,laplacian_normalize,sparse_ness = False):
    from weighting_func import laplacian, fourier, weight_wavelet
    L = laplacian(adj, normalized=laplacian_normalize)
    lamb, U = fourier(L)

    print U.shape,lamb[0],lamb[1]
    # print np.dot(np.dot(U,np.diag(lamb)),np.transpose(U))
    # U 列 eigen function
    Weight = weight_wavelet(s, lamb, U)

    if (mask == True):
        # mask 需要保留自身，为了使得自身的影响和其他可比，对adj加单位阵
        if (order == 0):
            adj = sp.csr_matrix(np.eye(adj.shape[0]))
        elif (order == 1):
            adj = adj + sp.csr_matrix(np.eye(adj.shape[0]))
        elif (order == 2):
            second_order = adj.dot(adj)
            adj = adj + sp.csr_matrix(np.eye(adj.shape[0])) + second_order
            del second_order
        elif (order == 3):
            second_order = adj.dot(adj)
            third_order = adj.dot(second_order)
            adj = adj + sp.csr_matrix(np.eye(adj.shape[0])) +  second_order + third_order
            del second_order, third_order
        elif (order == 4):
            second_order = adj.dot(adj)
            third_order = adj.dot(second_order)
            four_order = adj.dot(third_order)
            adj = adj + sp.csr_matrix(
                np.eye(adj.shape[0])) + second_order + third_order + four_order
            del second_order, third_order, four_order

        Weight = adj.multiply(Weight)

    if (sparse_ness):
        # for i in range(Weight.data.shape[0]):
        #     if(Weight.data[i] < threshold):
        #         Weight.data[i] = 0.0
        Weight[Weight < threshold] = 0

    # 一阶关系利用GCN拟合，其他关系利用wavelet
    print len(np.nonzero(Weight)[0])
    Weight = np.multiply(Weight,(np.ones(shape=adj.shape)-adj))
    # Weight = Weight * (np.ones(shape=adj.shape) - adj)
    print len(np.nonzero(Weight)[0])

    Weight = sp.csr_matrix(Weight)
    Weight.setdiag([0.0] * Weight.shape[0])
    print len(np.nonzero(Weight)[0])
    if (data_normalize == True):
        Weight = normalize(Weight, norm='l1', axis=1)


    print Weight.shape
    print len(np.nonzero(Weight)[0])
    print Weight

    # print Weight.todense()

    # Weight = normalize(Weight + sp.eye(adj.shape[0]),norm='l1',axis = 1)
    t_k = [Weight]
    # 一阶和自身关系利用GCN拟合
    t_k.append(preprocess_adj(adj))
    # t_k.append(sp.eye(adj.shape[0]))
    return sparse_to_tuple(t_k)

def feature_similarity(adj,feature):
    # for i in range(adj.shape[0]):
    #     print adj[i],adj[i].shape[0],adj[i].nnz
    # print type(feature)
    # feature = feature.tocsr().astype(np.float)
    print feature.shape

    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(feature)

    similarities[similarities<0.95] = 0
    # 14 和 5392相同
    # similarities[similarities < 1-1.0e-10]=0

    # import math
    # precision = 1.0e-10
    # for i in range(similarities.shape[0]):
    #     if(math.fabs(similarities[i,i]-1.0)>precision):
    #         print similarities[i,i]
    #
    # print "******************"

    # threshold , normalize

    print len(np.nonzero(similarities)[0])
    similarities = sp.csr_matrix(normalize(similarities,norm='l1',axis=1))
    print similarities

    # similarities = sp.csr_matrix(similarities)
    # print('pairwise dense output:\n {}\n'.format(similarities))
    t_k = [similarities]

    # adj_normalized = normalize_adj(adj)
    # laplacian = sp.eye(adj.shape[0]) - adj_normalized
    # largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    # scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    # t_k = list()
    #t_k.append(sp.eye(adj.shape[0]))
    # t_k.append(scaled_laplacian)


    return sparse_to_tuple(t_k)