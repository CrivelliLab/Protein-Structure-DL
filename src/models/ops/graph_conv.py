'''
graph_conv.py

This file contains tensorflow defintions for layers used in Graph
Convolution Neural Networks (GCNNs). These layers were written to operate using tensorflow
1.5 and above.

'''
import numpy as np
import tensorflow as tf

################################################################################

def VCAInput(nb_nodes, nb_coords, nb_features, a_mask=None):
    '''
    Method returns Tensorflow placeholders for graph convolutional networks.

    Params:
        nb_nodes - int; number of nodes in graphs
        nb_coords - int; number of spatial dimensions for points
        nb_features - int; number of features per node
        a_mask - np.array(); mask to apply on adjacency matrix

    Returns:
        v - Rank 3 tensor defining the node features; BATCHxNxF
        c - Rank 3 tensor defining coordinates of nodes in n-euclidean space; BATCHxNxC
        a - Rank 3 tensor defining L2 distances between nodes according to tensor c; BATCHxNxN

    '''
    # Define node feature vector 'V'
    v = tf.placeholder(tf.float32, [None, nb_nodes, nb_features])

    # Define node coordinate vector 'C'
    c = tf.placeholder(tf.float32, [None, nb_nodes, nb_coords])

    # Define mask placeholder
    m = tf.placeholder(tf.float32, [None, nb_nodes, nb_nodes])

    # Define pairwise adjaceny matrix set 'A'
    a = L2PDist(c)
    a_c = CosinePDist(c)

    '''
    # Mask Empty Nodes
    emp = tf.expand_dims(tf.clip_by_value(tf.reduce_sum(v,axis=-1), 0, 1),axis=-1)
    a_c = a_c * emp
    a = a * emp
    a = a * tf.transpose(emp, [0,2,1])
    a_c = a_c * tf.transpose(emp, [0,2,1])

    # Apply mask
    if a_mask is not None:
        mask_ = tf.convert_to_tensor(a_mask, dtype=tf.float32)
        mask_ = tf.reshape(mask_, [-1, a_mask.shape[0], a_mask.shape[1]])
        a = tf.multiply(a, mask_)
        a_c = tf.multiply(a, mask_)
    '''

    return v, c, [a,a_c], m

def L2PDist(c, namespace='l2pdist_a'):
    '''
    Method calculate L2 pariwise distances between coordinates in tensor c.

    Params:
        c - Rank 3 tensor defining coordinates of nodes in n-euclidean space.

    Returns:
        a - Rank 3 tensor defining pairwise adjacency matrix of nodes.

    '''
    # Calculate L2 pairwise distances
    l2 = tf.reduce_sum(c*c, 2)
    l2 = tf.reshape(l2, [-1, 1, l2.shape[-1]])
    a = l2 - 2*tf.matmul(c, tf.transpose(c, [0,2,1])) + tf.transpose(l2, [0,2,1])
    a = tf.abs(a, name=namespace)

    return a

def CosinePDist(c):
    '''
    Method calculates cosine pairwise distances between coordinates in tensor c.

    Params:
        c - Rank 3 tensor defining coordinates of nodes in n-euclidean space.

    Returns:
        a - rak 3 tensor defining cosine pairwise adjacency matrix of nodes.

    '''
    # Calculate cosine similarity
    normalized = tf.nn.l2_normalize(c, axis=-1)
    prod = tf.matmul(normalized, normalized, adjoint_b=True)
    a = (1 - prod) / 2.0
    a = 1 - a

    return a

def GraphKernels(v, a, nb_kernels, training=None, namespace='graphkernel_'):
    '''
    Method defines tensorflow layer which learns a graph kernel in order to normailize
    pairwise euclidean distances found in tensor a, according to node features in
    tensor v. The final set of normalized adjacency tensors a_prime have values
    ranging from [0,1].

    Params:
        v - Rank 3 tensor defining the node features; BATCHxNxF
        a - Rank 3 tensor defining L2 distances between nodes according to tensor c; BATCHxNxN
        nb_kernels - int; number of learned kernels

    Returns:
        a_prime - list(tf.arrays); list of normalized adjacency tensors; BATCHxNxN

    '''
    # Dimensions
    batch_size = tf.shape(v)[0]
    nb_features = int(v.shape[2])
    nb_nodes = int(v.shape[1])

    # Define trainable parameters
    w1 = tf.Variable(tf.truncated_normal([nb_features, nb_kernels], stddev=np.sqrt(2/((nb_nodes*nb_features)+(nb_nodes*nb_kernels)))))
    w2 = tf.Variable(tf.truncated_normal([nb_features, nb_kernels], stddev=np.sqrt(2/((nb_nodes*nb_features)+(nb_nodes*nb_kernels)))))
    w1 = tf.tile(tf.expand_dims(w1, axis=0), [batch_size, 1, 1])
    w2 = tf.tile(tf.expand_dims(w2, axis=0), [batch_size, 1, 1])

    # Normalize adjacency using learned graph kernels
    w1 = tf.split(w1,[1 for i in range(nb_kernels)],axis=-1)
    w2 = tf.split(w2,[1 for i in range(nb_kernels)],axis=-1)
    a_prime = []
    for i, _ in enumerate(list(zip(w1,w2))):
        w1_ = tf.tile(_[0], [1,1,nb_nodes])
        w2_ = tf.tile(_[1], [1,1,nb_nodes])
        e1 = tf.matmul(v, w1_)
        e2 = tf.matmul(v, w2_)
        e2 = tf.transpose(e2, [0,2,1])
        e = tf.nn.softplus(e1+e2, name=namespace+'k_'+str(i))
        if training is not None: tf.layers.batch_normalization(e, training=training)
        x = e * a
        a_ = tf.exp(-(x*x))
        a_prime.append(a_)

    return a_prime

def AngularContribution(v, c_a, a, training=None, namespace='angularcontr_'):
    '''
    Method defines tensorflow layer which learns a linear combination of cosine distance
    tensor c_a and graph kernel normalized euclidean distance tensors a, according to node features in
    tensor v.

    Params:
        v - Rank 3 tensor defining the node features; BATCHxNxF
        a - Rank 3 tensor defining cosine distances between nodes according to tensor c; BATCHxNxN
        a - list(tf.arrays); list of normalized adjacency tensors; BATCHxNxN

    Returns:
        a_prime - list(tf.arrays); list of normalized adjacency tensors; BATCHxNxN

    '''
    # Dimensions
    batch_size = tf.shape(v)[0]
    nb_features = int(v.shape[2])
    nb_nodes = int(v.shape[1])
    nb_kernels = len(a)

    # Define trainable parameters
    w1 = tf.Variable(tf.truncated_normal([nb_features, nb_kernels], stddev=np.sqrt(2/((nb_nodes*nb_features)+(nb_nodes*nb_kernels)))))
    w2 = tf.Variable(tf.truncated_normal([nb_features, nb_kernels], stddev=np.sqrt(2/((nb_nodes*nb_features)+(nb_nodes*nb_kernels)))))
    w1 = tf.tile(tf.expand_dims(w1, axis=0), [batch_size, 1, 1])
    w2 = tf.tile(tf.expand_dims(w2, axis=0), [batch_size, 1, 1])

    # Normalize adjacency using learned graph kernels
    w1 = tf.split(w1,[1 for i in range(nb_kernels)],axis=-1)
    w2 = tf.split(w2,[1 for i in range(nb_kernels)],axis=-1)
    a_prime = []
    for i, _ in enumerate(list(zip(w1,w2))):
        w1_ = tf.tile(_[0], [1,1,nb_nodes])
        w2_ = tf.tile(_[1], [1,1,nb_nodes])
        e1 = tf.matmul(v, w1_)
        e2 = tf.matmul(v, w2_)
        e2 = tf.transpose(e2, [0,2,1])
        z = tf.nn.sigmoid(e1+e2, name=namespace+'z_'+str(i))
        if training is not None: tf.layers.batch_normalization(z, training=training)
        a_ = tf.add((z*c_a), ((1-z)*a[i]))
        a_prime.append(a_)

    return a_prime

def GraphConv(v, a, nb_filters, namespace='graphconv_'):
    '''
    Method defines basic graph convolution operation using inputs V and A. First,
    features between nodes are progated through the graph according to adjacency matrix set
    A. Next, new features are mapped according to fully connected layer to produce
    node feature tensor V'.

    It's important to note that this implemenation of graph convolutions does not
    parameterize each edge in the graph, rather a single weight is shared between all edges
    of any given node. Parameterizing all edges at the moment requires the generation
    of seperate adjacency matricies for each edge independently which is severly inefficient
    due to sparsity. This problem is currently being researched.That being said, single-weight
    graph convolutions still preform reasonably well using euclidean pairwise distance representations.

    Update: Graph Kernel parameterization prior to graph convolution provides substantial improvement over
    single-weight convolution.

    Params:
        v - Rank 3 tensor defining node features.
        a - Rank 3 tensor defining pairwise adjacency matrix of nodes.
        nb_filters - int32; number of features in V'

    Returns:
        v_prime - Rank 3 tensor defining node features for V'

    '''
    # Dimensions
    batch_size = tf.shape(v)[0]
    nb_features = int(v.shape[2])
    nb_nodes = int(v.shape[1])
    support = len(a)

    # Define trainable parameters for feature weights
    w = tf.Variable(tf.truncated_normal([nb_features*support, nb_filters], stddev=np.sqrt(2/((nb_features*support)+(nb_filters)))),name=namespace+'w')
    w = tf.tile(tf.expand_dims(w, axis=0), [batch_size, 1, 1])
    b = tf.Variable(tf.zeros([nb_filters]), name=namespace+'b')

    # Update node features according to graph
    v_ = []
    for _ in a: v_.append(tf.matmul(_, v)/nb_nodes)
    v_ = tf.concat(v_, axis=-1)

    # Apply feature weights
    v_prime = tf.add(tf.matmul(v_, w), b, name=namespace+'vprime')

    return v_prime

def AverageSeqGraphPool(v, c, pool_size, namespace='averseqgraphpool_'):
    '''
    Method preforms sequence based average pooling on graph structure.
    V and C are assumed to be in sequential order.

    Params:
        v - Rank 3 tensor defining the node features; BATCHxNxF
        c - Rank 3 tensor defining coordinates of nodes in n-euclidean space; BATCHxNxC
        pool_size - int32; pool window size along sequence.

    Returns:
        v - Rank 3 tensor defining the node features for pooled V; BATCHx(N/pool_size)xF
        c - Rank 3 tensor defining coordinates of nodes in n-euclidean space for pooled C; BATCHx(N/pool_size)xC
        a - Rank 3 tensor defining L2 distances between nodes according to tensor c; BATCHx(N/pool_size)xN

    '''
    # Average Pool V and C
    v_prime = tf.layers.average_pooling1d(v, pool_size, pool_size, name=namespace+'vprime')
    c_prime = tf.layers.average_pooling1d(c, pool_size, pool_size, name=namespace+'cprime')

    # Generate new A
    a_prime = [L2PDist(c_prime, namespace=namespace+'aprime'), CosinePDist(c_prime)]

    return v_prime, c_prime, a_prime

def NodeAttention(v, nodes, fc_layers=[], fc_dropouts=[], training=None):

    # Dimensions
    batch_size = tf.shape(v)[0]
    nb_features = int(v.shape[2])

    # Define trainable parameters for attention weights
    x_i = tf.contrib.layers.xavier_initializer()
    u = tf.Variable(x_i([nb_features, nodes]))
    u = tf.tile(tf.expand_dims(u, axis=0), [batch_size, 1, 1])

    # Calculate node_pool contribution using attention
    node_pool_atten = tf.nn.softmax(tf.transpose(tf.matmul(v, u), [0,2,1])/np.sqrt(nb_features), axis=-1)

    # Pool features using node_pool_atten
    v_pool = tf.matmul(node_pool_atten, v)
    F= tf.contrib.layers.flatten(v_pool)

    return F
