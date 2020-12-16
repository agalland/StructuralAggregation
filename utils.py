import torch
import argparse
import ssl

import networkx as nx
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from ogb.graphproppred import GraphPropPredDataset


class GraphDataset:
    def __init__(self, folder_path=''):
        if folder_path.split("-")[0] != "ogbg":
            g = nx.Graph()
            data_adj = np.loadtxt(folder_path + '_A.txt',
                                  delimiter=',').astype(int)
            data_graph_indicator = np.loadtxt(folder_path + '_graph_indicator.txt',
                                              delimiter=',').astype(int)
            labels = np.loadtxt(folder_path + '_graph_labels.txt',
                                delimiter=',').astype(int)
            # If features aren't available, compute one-hot degree vectors
            try:
                node_labels = np.loadtxt(folder_path + '_node_labels.txt',
                                         delimiter=',').astype(int)
                node_labels -= np.min(node_labels)
                max_feat = np.max(node_labels) + 1
                mat_feat = np.eye(max_feat)
                with_node_features = True
            except:
                with_node_features = False

            data_tuple = list(map(tuple, data_adj))
            g.add_edges_from(data_tuple)
            g.remove_nodes_from(list(nx.isolates(g)))

            le = LabelEncoder()
            self.labels_ = le.fit_transform(labels)
            self.n_classes_ = len(le.classes_)
            self.n_graphs_ = len(self.labels_)

            graph_num = data_graph_indicator.max()
            node_list = np.arange(data_graph_indicator.shape[0]) + 1
            self.graphs_ = []
            self.node_features = []
            max_num_nodes = 0
            self.degree_max = 0
            for i in range(graph_num):
                if i % 500 == 0:
                    print("{}%".format(round((i * 100) / graph_num), 3))

                nodes = node_list[data_graph_indicator == i + 1]
                g_sub = g.subgraph(nodes).copy()

                max_cc = max(nx.connected_components(g_sub), key=len)
                g_sub = g_sub.subgraph(max_cc).copy()

                adj = np.array(nx.adjacency_matrix(g_sub).todense())
                self.degree_max = max(self.degree_max, np.max(np.sum(adj, 0)))
                nodes = range(len(adj))
                g_sub.graph['label'] = self.labels_[i]
                nx.convert_node_labels_to_integers(g_sub)

                tmp = len(nodes)
                self.graphs_.append(g_sub)
                if tmp > max_num_nodes:
                    max_num_nodes = tmp

                if with_node_features:
                    nodes = list(g_sub.nodes()) - np.min(list(g.nodes()))
                    feat_index = node_labels[nodes]
                    node_feat = mat_feat[feat_index]
                    self.node_features.append(node_feat)

            if not with_node_features:
                mat_feat = np.eye(self.degree_max+1)
                for i in range(graph_num):
                    g_sub = self.graphs_[i]
                    deg = np.array(list(dict(nx.degree(g_sub)).values()))
                    node_feat = mat_feat[deg]
                    self.node_features.append(node_feat)
        else:
            dataset = GraphPropPredDataset(name="ogbg-molhiv")
            split_idx = dataset.get_idx_split()
            train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
            self.train_idx = train_idx
            self.test_idx = test_idx
            self.valid_idx = valid_idx
            self.labels_ = dataset.labels.reshape(-1)
            self.graphs_ = []
            self.node_features = []
            max_feat, min_feat = get_boundaries_features(dataset)
            max_num_nodes = 0.
            for k in range(len(dataset.graphs)):
                edge_idx = dataset.graphs[k]["edge_index"]
                a = np.zeros((dataset.graphs[k]["num_nodes"],
                              dataset.graphs[k]["num_nodes"]))
                a[edge_idx[0], edge_idx[1]] = 1.
                g = nx.from_numpy_matrix(a)
                self.graphs_.append(g)
                graph_feat = get_one_hot_feature(max_feat,
                                                 min_feat,
                                                 dataset.graphs[k]["node_feat"])
                self.node_features.append(graph_feat)
                if dataset.graphs[k]["num_nodes"] > max_num_nodes:
                    max_num_nodes = dataset.graphs[k]["num_nodes"]

        self.n_graphs_ = len(self.graphs_)
        self.graphs_ = np.array(self.graphs_)
        self.max_num_nodes = max_num_nodes

        print('Loaded {} graphs,\
              the max number of nodes is {}'.format(self.n_graphs_,
                                                    self.max_num_nodes))

def get_boundaries_features(dataset):
    max_feat = np.zeros(9)
    min_feat = np.zeros(9)
    for k in range(len(dataset.graphs)):
        node_feat = dataset.graphs[k]["node_feat"]
        node_max_feat = np.max(node_feat, 0)
        node_min_feat = np.min(node_feat, 0)
        max_feat = np.concatenate(
            (max_feat.reshape(-1, 1), node_max_feat.reshape(-1, 1)), 1)
        min_feat = np.concatenate(
            (min_feat.reshape(-1, 1), node_min_feat.reshape(-1, 1)), 1)
        max_feat = np.max(max_feat, 1)
        min_feat = np.min(min_feat, 1)
        max_feat = max_feat.reshape(-1)
        min_feat = min_feat.reshape(-1)
    return max_feat, min_feat

def get_one_hot_feature(max_feat, min_feat, node_feat):
    range_feat = max_feat - min_feat
    mat_feat = []
    for k in range(len(range_feat)):
        mat_feat.append(np.eye(int(range_feat[k]) + 1))
    graph_feat_one_hot = None
    for node in range(node_feat.shape[0]):
        node_feat_one_hot = None
        for f in range(node_feat.shape[1]):
            if len(mat_feat[f]) == 0:
                continue
            if node_feat_one_hot is None:
                node_feat_one_hot = mat_feat[f][node_feat[node][f]]
            else:
                node_feat_one_hot = np.concatenate(
                    (node_feat_one_hot, mat_feat[f][node_feat[node][f]]))
        if graph_feat_one_hot is None:
            graph_feat_one_hot = node_feat_one_hot.reshape(1, -1)
        else:
            graph_feat_one_hot = np.concatenate(
                (graph_feat_one_hot, node_feat_one_hot.reshape(1, -1)), 0)

    return graph_feat_one_hot

def normalize(adj):
    deg = np.array(adj.sum(0))
    deg = np.sqrt(deg)
    adj = adj / deg
    adj = adj.transpose()
    adj = adj / deg

    return adj


def normalize_adj(dataset,
                  normalize_bool=True):
    X = []
    y = []
    for i in range(dataset.n_graphs_):
        adj = nx.adjacency_matrix(dataset.graphs_[i])
        if normalize_bool:
            adj = np.array(normalize(adj + 2 * sp.eye(adj.shape[0])))
            adj = sp.csr_matrix(adj)
        nodeFeature = np.array(dataset.node_features[i])
        nodeFeature = sp.csr_matrix(nodeFeature)

        if adj.shape[0] != nodeFeature.shape[0]:
            continue

        X.append((adj, nodeFeature))
        y.append(dataset.labels_[i])

    y = np.array(y)

    return X, y


def cross_val(net,
              X,
              y,
              dataset_name,
              path_results,
              n_feat,
              n_dim,
              g_size,
              bins,
              num_layers,
              out_dim,
              lr,
              num_epochs,
              path_weights,
              device,
              k,
              verbose,
              reg,
              alpha_reg,
              train_idx,
              test_idx,
              val_idx):

    path_save_weights = path_weights + dataset_name + str(bins) + str(n_dim) + str(k) + str(lr) + str(alpha_reg) + str(reg) + str(num_layers)

    print("lr: {}, bins: {}, n_dim: {}, regularize: {}, alpha reg: {}".format(lr, bins, n_dim, reg, alpha_reg))

    n_splits_val = 10
    skf_val = StratifiedKFold(n_splits=n_splits_val,
                              shuffle=True,
                              random_state=0)
    X_vec = np.zeros(len(X))

    accuracy_val = []
    accuracy_test = []
    accuracy_class_val = []
    roc_auc_val = []
    roc_auc_test = []
    accuracyRF_class_val = []

    if train_idx is None:
        for ifold, (train_ind1, val_ind) in enumerate(list(skf_val.split(X_vec, y))):
            accuracies = train_fold(X,
                                    y,
                                    train_ind1,
                                    val_ind,
                                    net,
                                    ifold,
                                    n_feat=n_feat,
                                    n_dim=n_dim,
                                    g_size=g_size,
                                    bins=bins,
                                    num_layers=num_layers,
                                    out_dim=out_dim,
                                    lr=lr,
                                    num_epochs=num_epochs,
                                    path_save_weights=path_save_weights,
                                    device=device,
                                    verbose=verbose,
                                    reg=reg,
                                    alpha_reg=alpha_reg)

            accuracy_val.append(accuracies[0])
            roc_auc_val.append(accuracies[1])
            accuracy_test.append(accuracies[2])
            roc_auc_test.append(accuracies[3])
            accuracy_class_val.append(accuracies[4])
            accuracyRF_class_val.append(accuracies[5])
            print(accuracies[0], accuracies[2], accuracies[4], accuracies[5])
    else:
        accuracies = train_fold(X,
                                y,
                                train_idx,
                                val_idx,
                                net,
                                0,
                                n_feat=n_feat,
                                n_dim=n_dim,
                                g_size=g_size,
                                bins=bins,
                                num_layers=num_layers,
                                out_dim=out_dim,
                                lr=lr,
                                num_epochs=num_epochs,
                                path_save_weights=path_save_weights,
                                device=device,
                                verbose=verbose,
                                reg=reg,
                                alpha_reg=alpha_reg,
                                test_idx=test_idx)

        accuracy_val.append(accuracies[0])
        roc_auc_val.append(accuracies[1])
        accuracy_test.append(accuracies[2])
        roc_auc_test.append(accuracies[3])
        accuracy_class_val.append(accuracies[4])
        accuracyRF_class_val.append(accuracies[5])

    print("mean accuracy on validation over folds: {}".format(np.mean(accuracy_val)))
    print("mean class accuracy on validation over folds: {}".format(np.mean(accuracy_class_val)))
    print("mean class accuracy on validation over folds with RF: {}".format(np.mean(accuracyRF_class_val)))
    print("mean class accuracy on test over folds: {}".format(np.mean(accuracy_test)))

    np.save(path_results + dataset_name + "_acc_" + str(bins) + "_" + str(n_dim) + "_" + str(k) + "_" + str(lr) + "_" + str(alpha_reg) + "_" + str(reg) + str(num_layers) + "_val.npy",
            accuracy_val)
    np.save(path_results + dataset_name + "_auc_" + str(bins) + "_" + str(n_dim) + "_" + str(k) + "_" + str(lr) + "_" + str(alpha_reg) + "_" + str(reg) + str(num_layers) + "_val.npy",
            roc_auc_val)
    np.save(path_results + dataset_name + "_acc_" + str(bins) + "_" + str(n_dim) + "_" + str(k) + "_" + str(lr) + "_" + str(alpha_reg) + "_" + str(reg) + str(num_layers) + "_test.npy",
            accuracy_test)
    np.save(path_results + dataset_name + "_auc_" + str(bins) + "_" + str(n_dim) + "_" + str(k) + "_" + str(lr) + "_" + str(alpha_reg) + "_" + str(reg) + str(num_layers) + "_test.npy",
            roc_auc_test)
    np.save(path_results + dataset_name + "_" + str(bins) + "_" + str(n_dim) + "_" + str(k) + "_" + str(lr) + "_" + str(alpha_reg) + "_" + str(reg) + str(num_layers) + "_class.npy",
            accuracy_class_val)
    np.save(path_results + dataset_name + "_" + str(bins) + "_" + str(n_dim) + "_" + str(k) + "_" + str(lr) + "_" + str(alpha_reg) + "_" + str(reg) + str(num_layers) + "_RFclass.npy",
            accuracyRF_class_val)

    print("mean accuracy test: {}".format(np.mean(accuracy_test)))
    print("mean roc auc score test: {}".format(np.mean(roc_auc_test)))
    print("mean accuracy val: {}".format(np.mean(accuracy_val)))
    print("mean roc auc score val: {}".format(np.mean(roc_auc_val)))


def train_fold(X,
               y,
               train_ind1,
               val_ind,
               net,
               ifold,
               n_feat,
               n_dim,
               g_size,
               bins,
               num_layers,
               out_dim,
               lr,
               num_epochs,
               path_save_weights,
               device,
               verbose,
               reg,
               alpha_reg,
               test_idx=None):

    X_val = [X[k] for k in val_ind]
    y_val = y[val_ind]

    if test_idx is None:
        n_splits = 9
        skf = StratifiedKFold(n_splits=n_splits,
                              shuffle=True,
                              random_state=0)
        X_vec = np.zeros((len(train_ind1), 2))
        y1 = y[train_ind1]

        split = skf.split(X_vec, y1)
        train_ind2, test_ind = list(split)[0]

        train_ind = train_ind1[train_ind2]
        test_ind = train_ind1[test_ind]
    else:
        train_ind = train_ind1
        test_ind = test_idx

    X_train = [X[k] for k in train_ind]
    y_train = y[train_ind]
    X_test = [X[k] for k in test_ind]
    y_test = y[test_ind]
    path_save_weights += "_fold_" + str(ifold)
    train_net = net(n_feat=n_feat,
                    n_dim=n_dim,
                    g_size=g_size,
                    bins=bins,
                    num_layers=num_layers,
                    out_dim=out_dim,
                    lr=lr,
                    num_epochs=num_epochs,
                    path_save_weights=path_save_weights,
                    device=device)
    train_net.fit(X_train,
                  y_train,
                  X_test,
                  y_test,
                  verbose,
                  reg=reg,
                  alpha_reg=alpha_reg)
    train_net.net.load_state_dict(torch.load(path_save_weights))
    acc_val, auc_val = train_net.predict(X_val, y_val)

    # Compute assignment matrices to evaluate histograms of classes
    x_class_list_train = train_net.get_class_select(X_train)
    x_class_list_val = train_net.get_class_select(X_val)

    bins1 = 5
    x_class_train = np.zeros((len(x_class_list_train), bins*bins1))
    x_class_val = np.zeros((len(x_class_list_val), bins*bins1))

    for i, x in enumerate(x_class_list_train):
        for j in range(bins):
            xj = np.histogram(x[j], bins=bins1, range=(0., 1.))[0]
            x_class_train[i, j*bins1:(j+1)*bins1] = xj
    for i, x in enumerate(x_class_list_val):
        for j in range(bins):
            xj = np.histogram(x[j], bins=bins1, range=(0., 1.))[0]
            x_class_val[i, j * bins1:(j + 1) * bins1] = xj

    # SVM to evaluate information in assignment matrices
    clf = SVC(gamma="auto")
    clf.fit(x_class_train, y_train)
    y_pred_class_val = clf.predict(x_class_val)
    acc_class_val = accuracy_score(y_val, y_pred_class_val)

    clfRF = RandomForestClassifier(max_depth=10, n_estimators=1000)
    clfRF.fit(x_class_train, y_train)
    y_pred_class_val = clfRF.predict(x_class_val)
    accRF_class_val = accuracy_score(y_val, y_pred_class_val)

    acc_test, auc_test = train_net.predict(X_test, y_test)

    return acc_val, auc_val, acc_test, auc_test, acc_class_val, accRF_class_val


def parse_args():
    parser = argparse.ArgumentParser(description="structAgg arguments.")
    parser.add_argument("--numLayers", dest="num_layers")
    parser.add_argument("--numEpochs", dest="num_epochs")
    parser.add_argument("--verbose", dest="verbose")
    parser.add_argument("--reg", dest="reg")
    parser.add_argument("--pathResults", dest="path_results")
    parser.add_argument("--pathWeights", dest="path_weights")
    parser.add_argument("--pathData", dest="path_data")
    parser.add_argument("--loadWeights", dest="path_load_weights")
    parser.add_argument("--lr", dest="lr")
    parser.add_argument("--bins", dest="bins")
    parser.add_argument("--ndim", dest="n_dim")
    parser.add_argument("--alphaReg", dest="alpha_reg")
    parser.add_argument("--datasetName", dest="dataset_name")

    parser.set_defaults(num_layers=2,
                        num_epochs=10000,
                        verbose=False,
                        reg=True,
                        lr=0.001,
                        bins=5,
                        n_dim=16,
                        alpha_reg=0.1,
                        path_results="results/",
                        path_weights="weights/",
                        path_data="data/",
                        path_load_weights="weightsEval/")
    args = vars(parser.parse_args())

    return args