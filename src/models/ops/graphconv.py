'''
layers.py

This file contains tensorflow defintions for layers used in Euclidean Graph
Convolution Neural Networks (EGCNNs). These layers were written to operate using tensorflow
1.5 and above.

'''
import numpy as np
import tensorflow as tf


################################################################################

def VCAInput(nb_nodes, nb_coords, nb_features, e_radius=1, dialations=1, a_mask=None):
    '''
    Method creates placeholders for inputs V and C, and constructs pairwise distance
    representation A from values of C.

    Params:
        nb_nodes - int32; number of nodes in input
        nb_coords - int32; number of dimensions in node coordinates
        nb_features - int32; number of features per node
        e_radius - float32; outerbound radius for exponential normalization
        a_mask - np.array((nb_nodes,nb_nodes)); mask applied to A

    Returns:
        v - Rank 3 placeholder tensor defining node features.
        c - Rank 3 placeholder tensor defining coordinates of nodes in n-euclidean space.
        a - Rank 3 tensor defining pairwise adjacency matrix of nodes. [See L2PDistENorm for more information]

    '''
    # Define node feature vector 'V'
    v = tf.placeholder(tf.float32, [None, nb_nodes, nb_features])

    # Define node coordinate vector 'C'
    c = tf.placeholder(tf.float32, [None, nb_nodes, nb_coords])

    # Define pairwise adjaceny matrix set 'A'
    if dialations > 1: d = np.arange(0,e_radius+e_radius/(dialations-1),e_radius/(dialations-1))
    else: d = [0.0,]
    a_ = L2PDist(c)
    a = [ENorm(a_, e_radius, _, a_mask) for _ in d]

    return v, c, a

def VCAInputVanilla(nb_nodes, nb_coords, nb_features, a_mask=None):
    '''

    '''
    # Define node feature vector 'V'
    v = tf.placeholder(tf.float32, [None, nb_nodes, nb_features])

    # Define node coordinate vector 'C'
    c = tf.placeholder(tf.float32, [None, nb_nodes, nb_coords])

    # Define pairwise adjaceny matrix set 'A'
    a = L2PDist(c)

    # Mask Empty
    emp = tf.expand_dims(tf.clip_by_value(tf.reduce_sum(v,axis=-1), 0, 1),axis=-1)
    a = a * emp
    a = a * tf.transpose(emp, [0,2,1])

    # Apply mask
    if a_mask is not None:
        mask_ = tf.convert_to_tensor(a_mask, dtype=tf.float32)
        mask_ = tf.reshape(mask_, [-1, a_mask.shape[0], a_mask.shape[1]])
        a = tf.multiply(a, mask_)

    return v, c, a

def LearnedENorm(v,a,nb_filters):
    '''
    '''
    # Dimensions
    batch_size = tf.shape(v)[0]
    nb_features = int(v.shape[2])
    nb_nodes = int(v.shape[1])

    # Define trainable parameters for learned e
    x_i = tf.contrib.layers.xavier_initializer()
    u = tf.Variable(x_i([nb_features, nb_filters]))
    u = tf.tile(tf.expand_dims(u, axis=0), [batch_size, 1, 1])

    es = tf.split(u,[1 for i in range(nb_filters)],axis=-1)
    a_prime = []
    for _ in es:
        # Get e
        u_ = tf.tile(_, [1,1,nb_nodes])
        e = tf.nn.relu(tf.matmul(v, u_))
        e = tf.transpose(e, [0,2,1])
        #e = tf.clip_by_value(e, 0, d)
        a_ = tf.exp((-a[0])/(e+0.0001))
        a_prime.append(a_)

    return a_prime


def L2PDist(c):
    '''

    Params:
        c - Rank 3 tensor defining coordinates of nodes in n-euclidean space.

    Returns:
        a - Rank 3 tensor defining pairwise adjacency matrix of nodes.

    '''
    # Calculate L2 pairwise distances
    l2 = tf.reduce_sum(c*c, 2)
    l2 = tf.reshape(l2, [-1, 1, l2.shape[-1]])
    a = l2 - 2*tf.matmul(c, tf.transpose(c, [0,2,1])) + tf.transpose(l2, [0,2,1])
    a = tf.abs(a)

    return a

def ENorm(a, e_radius=1, d_radius=0, mask=None):
    '''
    Params:
        a - Rank 3 tensor defining pairwise adjacency matrix of nodes.
        e_radius - float32; outerbound radius for exponential normalization. If distances are close
            to this value, they will approach a normalized value of 0.01. This parameter should be changed
            according to fit the typical radius from the origin of input points.
        mask - np.array((nb_nodes,nb_nodes)); mask applied to pairwise distance matrix.
            Commonly used to supress interaction between specific node pairs.

    Returns:
        a - Rank 3 tensor defining exponentiallly normalized pairwise adjacency matrix of nodes.

    '''
    # Apply exponential normalization
    C = e_radius/np.log(40)
    if d_radius > 0.0:
        a = a - d_radius
        a = 1 / tf.exp(a/C)
        a = a * tf.to_float(tf.less(a,tf.fill(tf.shape(a), 1.0)))
    else:
        a = 1 / tf.exp(a/C)

    # Apply mask
    if mask is not None:
        mask_ = tf.convert_to_tensor(mask, dtype=tf.float32)
        mask_ = tf.reshape(mask_, [-1, mask.shape[0], mask.shape[1]],'valid')
        a = tf.multiply(a, mask_)

    return a

def GraphPool(v,c, pool_size):
    '''
    '''
    v_ = tf.reduce_sum(v,axis=-1)
    nb_nodes = int(v.shape[1])
    nb_features = int(v.shape[-1])
    nb_coords = int(c.shape[-1])
    batch_size = tf.shape(v)[0]
    slices = [pool_size for i in range(nb_nodes//pool_size)]
    if nb_nodes%pool_size> 0: slices = slices+[nb_nodes%pool_size,]
    vs = tf.split(v_,slices, axis=1)
    argmaxs = []
    for i,_ in enumerate(vs):
        argmax = tf.argmax(_, axis=-1, output_type=tf.int32) + i*pool_size
        argmax = tf.reshape(argmax, [-1,1, 1])
        r = tf.reshape(tf.range(0, batch_size, 1), [-1, 1, 1])
        argmax = tf.concat([r,argmax], axis=-1)
        argmaxs.append(argmax)
    argmaxs = tf.concat(argmaxs, axis=-2)
    #argmaxs = tf.reshape(argmaxs, [1,batch_size, argmaxs.shape[-2], argmaxs.shape[-1]])
    v_prime = tf.gather_nd(v,argmaxs)
    c_prime = tf.gather_nd(c,argmaxs)
    a_prime = L2PDist(c_prime)

    return v_prime, c_prime, a_prime

def GraphConv(v, a, nb_filters, batch_norm=True, activation=tf.nn.softsign, dropout=0.0):
    '''
    Method defines basic graph convolution operation using inputs V and A. First,
    features between nodes are progated through the graph according to adjacency matrix
    A. Next, new features are computed according to trainable parameters to produce
    node feature tensor V'. If applicable, batch normalization, activation and dropout
    are applied to V' in that order.

    It's important to note that this implemenation of graph convolutions does not
    parameterize each edge in the graph, rather a single weight is shared between all edges
    of any given node. Parameterizing all edges at the moment requires the generation
    of seperate adjacency matricies for each edge independently which is severly inefficient
    due to sparsity. This problem is currently being researched.That being said, single-weight
    graph convolutions still preform reasonably well using euclidean pairwise distance representations.

    Params:
        v - Rank 3 tensor defining node features.
        a - Rank 3 tensor defining pairwise adjacency matrix of nodes.
        nb_filters - int32; number of features in V'
        batch_norm - boolean; whether to use batch normalization
        activation - tf.activation; desired activation function. So far, softsign has shown
                best performance and has prevented gradient explosion.
        dropout - float32; dropout rate.

    Returns:
        v_prime - Rank 3 tensor defining node features for V'

    '''
    # Dimensions
    batch_size = tf.shape(v)[0]
    nb_features = int(v.shape[2])
    nb_nodes = int(v.shape[1])
    support = len(a)

    # Define trainable parameters for feature weights
    x_i = tf.contrib.layers.xavier_initializer()
    w = tf.Variable(x_i([nb_features*support, nb_filters]))
    w = tf.tile(tf.expand_dims(w, axis=0), [batch_size, 1, 1])
    b = tf.Variable(tf.zeros([nb_filters]))

    # Update node features according to graph
    v_ = []
    for _ in a:
        v_.append(tf.matmul(_, v)/nb_nodes)
    v_ = tf.concat(v_, axis=-1)

    # Apply feature weights
    v_prime = tf.add(tf.matmul(v_, w), b)

    # Apply batch normalization, activation, and dropout
    if batch_norm: v_prime = tf.layers.batch_normalization(v_prime)
    if activation: v_prime = activation(v_prime)
    if dropout > 0.0: v_prime = tf.layers.dropout(v_prime, dropout)

    return v_prime

def GraphAveragePool(v, c, nb_nodes_pool, temp=None, e_radius=1, dialations=1, a_mask=None):
    '''
    Method defines a globally-aware euclidean graph pooling operation using inputs
    V, C. First, the contribution of each node in respect to the new set of
    nodes are computed using an attention-based operation dependent of node features
    V. Using the node contribution matrix, features in V are pooled into V_Pool and
    node coordinates C are pooled into C_Pool. Next, coordinates in C_Pool are centered
    around the new centroid. Finally, a new pairwise adjacency matrix A_Pool is computed from
    the values of C_Pool.

    This implementation was designed to work around the limitations of traditional graph
    pooling operations. By addressing the problem in euclidean space, the resulting
    pooled graphs are inherently tied to the original distribution of points. This is done
    by using an attention-based operation to learn where a new node lies in respect to
    the location of all original nodes. As a result, the new node is a weighted centroid of
    the original set of nodes and will always fall in some location within the convex hull of the original
    set of nodes.

    NOTE: Intitial exploration on datasets found that pooling node features of the first
    input features (i.e no convolution V) using the convolved representations of V to compute
    contribution improves performance of the model. There is no current explanation as to why
    the model benefits from this architectural decision.

    Attention can be tuned by adjusting the softmax temperature of the operation,
    but a common hueristic for temperature is defined as the sqrt(nb_features).

    Params:
        v - Rank 3 tensor defining defining node features which will be pooled.
        c - Rank 3 tensor defining coordinates of nodes in n-euclidean space.
        nb_nodes_pool - int32; number of nodes in pooled graph.
        temp - float32; temperature/scaling-factor used when calculating softmax attention.
            High temperature (ie. >1) = low varience between softmax probabilities.
            Low temperature (ie. >0 and <1) = high varience between softmax probabilities
        e_radius - float32; outerbound radius for exponential normalization.
        a_mask - np.array((nb_nodes,nb_nodes)); mask applied to pooled pairwise distance matrix.

    Returns:
        v_pool - Rank 3 placeholder tensor defining node features of pooled graph.
        c_pool - Rank 3 placeholder tensor defining coordinates of pooled nodes in n-euclidean space.
        a_pool - Rank 3 tensor defining pairwise adjacency matrix of pooled nodes. [See L2PDistENorm for more information]

    '''
    # Dimensions
    batch_size = tf.shape(v)[0]
    nb_features = int(v.shape[2])
    nb_nodes = int(v.shape[1])

    if not temp: temp = np.sqrt(nb_features)

    # Define trainable parameters for attention weights
    x_i = tf.contrib.layers.xavier_initializer()
    u = tf.Variable(x_i([nb_features, nb_nodes_pool]))
    u = tf.tile(tf.expand_dims(u, axis=0), [batch_size, 1, 1])

    # Calculate node_pool contribution using attention
    node_pool_atten = tf.nn.softmax(tf.transpose(tf.matmul(v, u), [0,2,1])/temp, axis=-1)

    # Pool features using node_pool_atten
    v_pool = tf.matmul(node_pool_atten, v) / nb_nodes

    # Pool coordinates using weighted averaging for each new node
    c_pool = tf.matmul(node_pool_atten, c)
    c_pool_ = tf.reshape(tf.reduce_mean(c_pool, axis=1), [-1, 1, c_pool.shape[-1]])
    c_pool = c_pool - c_pool_

    # Generate new pairwise adjaceny matrix
    if dialations > 1: d = np.arange(0,e_radius+e_radius/(dialations-1),e_radius/(dialations-1))
    else: d = [0.0,]
    a_ = L2PDist(c_pool)
    a_pool = [ENorm(a_, e_radius, _, a_mask) for _ in d]

    return v_pool, c_pool, a_pool
