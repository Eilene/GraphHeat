# -*- coding:UTF-8 -*-
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

from utils import *
from models import GCN, MLP, GCN_WeightShare

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'pubmed', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'wavelet_origin', 'Model string.')  # 'wavelet_origin', 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_bool('weight_share', False, 'Weight share string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')#1000
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 200, 'Tolerance for early stopping (# of epochs).')#200
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_bool('alldata', False, 'All data string.')
flags.DEFINE_float('wavelet_s', 3.0, 'wavelet s .')
flags.DEFINE_float('threshold', 1e-5, 'sparseness threshold .')
flags.DEFINE_bool('mask', False, 'mask string.')
flags.DEFINE_bool('normalize', False, 'normalize string.')
flags.DEFINE_bool('laplacian_normalize', True, 'laplacian normalize string.')
flags.DEFINE_bool('sparse_ness', True, 'wavelet sparse_ness string.')
flags.DEFINE_integer('order', 2, 'neighborhood order .')
flags.DEFINE_bool('weight_normalize', False, 'weight normalize string.')

# Load data
labels, adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset,alldata=FLAGS.alldata)

# Some preprocessing, normalization
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    # print(adj)
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    # print(support)
    num_supports = 1 + FLAGS.max_degree
    if (FLAGS.weight_share):
        model_func = GCN_WeightShare
    else:
        model_func = GCN

elif FLAGS.model == 'wavelet_origin':
    dataset = FLAGS.dataset
    s = FLAGS.wavelet_s
    mask = FLAGS.mask
    normalize = FLAGS.normalize
    threshold = FLAGS.threshold
    laplacian_normalize = FLAGS.laplacian_normalize
    sparse_ness = FLAGS.sparse_ness
    order = FLAGS.order
    support = wavelet_origin(dataset,order,threshold,s,mask,normalize,adj,laplacian_normalize,sparse_ness=sparse_ness)
    num_supports = len(support)
    if (FLAGS.weight_share):
        model_func = GCN_WeightShare
    else:
        model_func = GCN

elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Some preprocessing, normalization
# features = preprocess_features(features)

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
weight_normalize = FLAGS.weight_normalize
node_num = adj.shape[0]
model = model_func(node_num,weight_normalize, placeholders, input_dim=features[2][1], logging=True)


# Initialize session
sess = tf.Session()

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.outputs,model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2],(time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
best_val_acc = 0.0
best_test_acc = 0.0
output_test_acc = 0.0

for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    val_output,cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Test
    test_output, test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    # 记录acc 最大的时候
    if(best_val_acc <= acc):
        best_val_acc = acc
        output_test_acc = test_acc

    if(best_test_acc <= test_acc):
        best_test_acc = test_acc

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "test_loss=", "{:.5f}".format(test_cost),
          "test_acc=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print(epoch,FLAGS.early_stopping)
        print (cost_val[-1],np.mean(cost_val[-(FLAGS.early_stopping+1):-1]))
        print(cost_val[-(FLAGS.early_stopping+1):-1])
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_output,test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
# res_detail,count = TF_detail(test_output)
# print(res_detail)
print("dataset: ",FLAGS.dataset," model: ",FLAGS.model,"order: ",FLAGS.order,",sparse_ness: ",FLAGS.sparse_ness,
      ",laplacian_normalize: ",FLAGS.laplacian_normalize,",threshold",FLAGS.threshold,",wavelet_s:",FLAGS.wavelet_s,",mask:",FLAGS.mask,
      ",normalize:",FLAGS.normalize,",weight_normalize:",FLAGS.weight_normalize," weight_share:",FLAGS.weight_share,
      ",learning_rate:",FLAGS.learning_rate,",hidden1:",FLAGS.hidden1,",dropout:",FLAGS.dropout,",max_degree:",FLAGS.max_degree,",alldata:",FLAGS.alldata)

print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
print("best val acc:", best_val_acc, " test acc: ",output_test_acc," best test acc: ",best_test_acc)

print("********************************************************")