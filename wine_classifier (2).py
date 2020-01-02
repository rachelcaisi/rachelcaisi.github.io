#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission.
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""
from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions, _print_for_automaker
#from mpl_toolkits.mplot3d import Axes3D # added this to plot 3D

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']
###################### Danielle Bueno and Rachel Miller ########################
# Last modified: 2019/04/30 14:00 Danielle
#-------------------------------------------------------------------------------
# Displaying things for report functions
'''def plot_2D(train_set, train_labels):
    n_features = train_set.shape[1]
    # ax is matrix of your plot
    class_colours = [CLASS_1_C, CLASS_2_C, CLASS_3_C]
    # Make an array of colors that correspond to train labels
    color_arr = np.zeros_like(train_labels, dtype = np.object)
    color_arr[train_labels == 1] = CLASS_1_C
    color_arr[train_labels == 2] = CLASS_2_C
    color_arr[train_labels == 3] = CLASS_3_C
    # for loops to go through each data set
    fig, ax = plt.subplots(n_features, n_features)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)
    for i in range(0,n_features):
        # Assign each data point different color
        for j in range (0,n_features):
            ax[i,j].scatter(train_set[:,i],train_set[:,j], s= 1, c = color_arr)
            ax[i,j].set_title('Features {} vs {}'. format(i+1, j+1), fontsize = 2)
    fig1, ax1 = plt.subplots()
    ax1.scatter(train_set[:,10], train_set[:,12], s=25, c = color_arr)
    ax1.set_title('Features 11 (Hue) vs 13 (Proline)', fontsize = 15)
    ax1.set_xlabel('Hue')
    ax1.set_ylabel('Proline')
    plt.show() # '''
#--------
'''def plot_3D(train_set, train_label):
    class_colours = [CLASS_1_C, CLASS_2_C, CLASS_3_C]
    # Make an array of colors that correspond to train labels
    color_arr = np.zeros_like(train_labels, dtype = np.object)
    color_arr[train_labels == 1] = CLASS_1_C
    color_arr[train_labels == 2] = CLASS_2_C
    color_arr[train_labels == 3] = CLASS_3_C
    D = train_set
    fig, ax = plt.subplots(subplot_kw={'projection' : '3d'})
    ax.scatter(D[:,6], D[:,10], D[:,12], color=color_arr, s=70)
    ax.set_title('Features 7 (Flavanoidsm) vs 11 (Hue) vs 13 (Proline)', fontsize = 15)
    ax.set_xlabel('Flavanoidsm')
    ax.set_ylabel('Hue')
    ax.set_zlabel('Proline')
    ax.grid(True) # False
    plt.show() # '''
#--------
'''def calculate_accuracy(gt_labels, pred_labels):
    correct = 0
    for i in range(len(gt_labels)):
        if (gt_labels[i] == pred_labels[i]):
            correct = correct + 1
    return correct / len(gt_labels) # '''
#--------
# Takes in a set of training labels and predicted labels and calculates confusion matrix
'''def calculate_confusion_matrix(gt_labels, pred_labels):
    # Save the number of unique classes to an array
    size_matrix = np.unique(train_labels)

    # Create size of confusion matrix
    numerator = np.zeros((len(size_matrix), len(size_matrix)))
    # Calculates the denominator
    denom_matrix = np.zeros(len(size_matrix))

    confusion_final =  np.zeros((len(size_matrix), len(size_matrix)))

    # Check that each row is one
    for k in range(len(gt_labels)):
        for i in range(1, len(size_matrix) + 1):
            # Calculate total number of samples from class i (denominator)
            if (gt_labels[k] == i):
                denom_matrix[i-1] += 1
            for j in range(1, len(size_matrix)+1):
                # If sample was classified as class j but belong to class i, add together
                if (gt_labels[k] == i and pred_labels[k]== j):
                    numerator[i-1][j-1] += 1

    # Divide by denom by each row to get final matrix
    for i in range(0,len(size_matrix)):
        confusion_final[i] = np.around((numerator[i]/denom_matrix[i]),3)
    print (confusion_final)
    return confusion_final

def plot_matrix(matrix, ax=None):
    """
    Displays a given matrix as an image.
    Args:
        - matrix: the matrix to be displayed
        - ax: the matplotlib axis where to overlay the plot.
          If you create the figure with `fig, fig_ax = plt.subplots()` simply pass `ax=fig_ax`.
          If you do not explicitily create a figure, then pass no extra argument.
          In this case the  current axis (i.e. `plt.gca())` will be used
    """
    fig, fig_ax = plt.subplots()
    color_map = plt.get_cmap('summer')
    matrixmap = plt.imshow(matrix, cmap = color_map)
    side_bar = plt.colorbar(matrixmap)
    # Put numbers in
    for i in range(0,3):
        for j in range(0,3):
            plt.text(i,j,s = matrix[i][j])
    if ax is None:
        ax = plt.gca()
    plt.show()
    #return fig, fig_ax #'''
#--------
'''def plot_pca(pca_train, f1, f2):
    class_colours = [CLASS_1_C, CLASS_2_C, CLASS_3_C]
    # Make an array of colors that correspond to train labels
    color_arr = np.zeros_like(train_labels, dtype = np.object)
    color_arr[train_labels == 1] = CLASS_1_C
    color_arr[train_labels == 2] = CLASS_2_C
    color_arr[train_labels == 3] = CLASS_3_C
    fig, ax = plt.subplots()
    ax.scatter(pca_train[:, 0], pca_train[:, 1], c=color_arr)
    ax.set_title('Reduced with my PCA: Features {} vs {}'.format(f1, f2), fontsize = 15)
    ax.set_xlabel('{}'.format(f1))
    ax.set_ylabel('{}'.format(f2))
    plt.show() #'''
#-------------------------------------------------------------------------------
# Additional helper functions
def reduce_data(train_set, test_set, selected_features):
    reduce_train_set = train_set[:, selected_features]
    reduce_test_set = test_set[:, selected_features]
    return reduce_train_set, reduce_test_set

def knn_all(reduce_train_set, reduce_test_set, k):
    dist = lambda x,y: np.sqrt(np.sum((x-y)**2))
    train_set_dist = lambda x : [dist(x, reduce_train_set_1) for reduce_train_set_1 in reduce_train_set]
    distances = np.array([train_set_dist(p) for p in reduce_test_set])
    # Sort distances from min to max
    sorted_dist_idx = np.argsort(distances, axis=1).astype(np.int)
    k_distances = sorted_dist_idx[:,0:k]
    number_test = np.shape(reduce_test_set)[0]
    kn = np.array(np.zeros((number_test,k)))
    for j in range(0,number_test):
        kn[j,:] = train_labels[k_distances[j,0:k]]
    # print('kkkkkk', kn) useful print -------------------------------------------
    n = np.shape(kn)
    counts = np.array(np.zeros((n[0],3))) #an array of 0s to  store counts for each class
    for j in range(0,n[0]): #goes along the rows
        for i in range(0,n[1]): #goes up to column k
            if (kn[j,i]==1):
                counts[j,0] = counts[j,0]+1 #counts for 1 in first column
            elif (kn[j,i]==2):
                counts[j,1] = counts[j,1]+1 #counts for 2 in second column
            else:
                counts[j,2] = counts[j,2]+1 #counts for 3 in third column
    #print('>>>>>>>',counts) #useful print ---------------------
    max_idx = np.argmax(counts, axis=1).reshape((np.shape(kn)[0],1)) #get indices of max count in each row and makes array
    pred_class = max_idx+1 #adds one to the indices so its the class
    for j in range(0,n[0]):
        same_count = np.where(counts[j,:]==counts[j, max_idx[j]])[0]
        #print('nnnnn',same_count) #useful print -----------------------
        if (np.shape(same_count)[0] != 1):
            if (np.shape(same_count)[0] == 2):
                no1 = same_count[0]+1
                no2 = same_count[1]+1
                no1_occur = np.where(kn[j,:]==no1)[0][0]
                no2_occur = np.where(kn[j,:]==no2)[0][0]
                if (no2_occur < no1_occur):
                    pred_class[j] = no2
            else:
                no1 = same_count[0]+1
                no2 = same_count[1]+1
                no3 = same_count[2]+1
                no1_occur = np.where(kn[j,:]==no1)[0][0]
                no2_occur = np.where(kn[j,:]==no2)[0][0]
                no3_occur = np.where(kn[j,:]==no3)[0][0]
                if (no2_occur < no1_occur and no2_occur < no3_occur):
                    pred_class[j] = no2
                elif (no3_occur < no1_occur and no3_occur < no2_occur):
                    pred_class[j] = no3
    return pred_class
#--------
def alt_imp(reduce_train_set, train_labels):
    # same two features 11 and 13
    total = np.shape(reduce_train_set)[0]
    c_prob = np.array(np.zeros((3,1)))
    for j in range(0,3):
        c_prob[j] = np.shape(np.where(train_labels == j+1))[1] / total
    mean_tb = np.array(np.zeros((3,2)))
    for j in range(0,3):
        idx = np.where(train_labels == j+1)[0]
        mean_tb[j,:] = np.mean(reduce_train_set[idx,:], axis = 0)
    std_tb = np.array(np.zeros((3,2)))
    for j in range(0,3):
        idx = np.where(train_labels == j+1)[0]
        std_tb[j,:] = np.std(reduce_train_set[idx,:], axis = 0)
    return c_prob, mean_tb, std_tb

def calcP(x, mean, std):
    expo = np.exp(-(((x-mean)**2)/(2*(std**2))))
    pi = 3.141592653589793
    return (1/(((2*pi)*(std**2))**0.5))*expo

def find_MAP(x, c_prob, mean_tb, std_tb):
    prob_tb = np.array(np.zeros((3,1)))
    for j in range(0,3):
        prob_tb[j] = calcP(x[0], mean_tb[j,0], std_tb[j,0]) * calcP(x[1], mean_tb[j,1], std_tb[j,1]) * c_prob[j]
    return np.argmax(prob_tb)+1
#-------------------------------------------------------------------------------
# Original Functions
def feature_selection(train_set, train_labels, **kwargs):
    '''plot_2D(train_set, train_labels) #'''
    return np.array([10,12])

def knn(train_set, train_labels, test_set, k, **kwargs):
    reduce_train_set, reduce_test_set = reduce_data(train_set, test_set, feature_selection(train_set, train_labels))
    dist = lambda x,y: np.sqrt(np.sum((x-y)**2))
    train_set_dist =lambda x : [dist(x, reduce_train_set_1) for reduce_train_set_1 in reduce_train_set]
    distances = np.array([train_set_dist(p) for p in reduce_test_set])
    pred_class = knn_all(reduce_train_set, reduce_test_set, k)
    '''confusion_ma = calculate_confusion_matrix(test_labels, pred_class)
    fig, fig_ax = plt.subplots()
    plot_matrix(confusion_ma,ax = fig_ax) #'''
    '''print(calculate_accuracy(test_labels, pred_class)) #'''
    #print('>>>>><<<<',pred_class.dtype)
    return pred_class

def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    reduce_train_set, reduce_test_set = reduce_data(train_set, test_set, feature_selection(train_set, train_labels))
    c_prob, mean_tb, std_tb = alt_imp(reduce_train_set, train_labels)
    x = reduce_test_set
    n = np.shape(x)[0]
    pred_class = np.array(np.zeros((n,1)), dtype = 'int64')
    for j in range(0,n):
        pred_class[j] = find_MAP(x[j], c_prob, mean_tb, std_tb) #'''
    '''confusion_ma = calculate_confusion_matrix(test_labels, pred_class)
    fig, fig_ax = plt.subplots()
    plot_matrix(confusion_ma,ax = fig_ax) #'''
    '''print(calculate_accuracy(test_labels, pred_class)) #'''
    #print('>>>>><<<<',pred_class.dtype)
    return pred_class

def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    reduce_train_set, reduce_test_set = reduce_data(train_set, test_set, np.array([9,10,12])) # 10,11,13
    '''plot_3D(train_set,train_labels) '''
    pred_class = knn_all(reduce_train_set, reduce_test_set, k)
    '''confusion_ma = calculate_confusion_matrix(test_labels, pred_class)
    fig, fig_ax = plt.subplots()
    plot_matrix(confusion_ma,ax = fig_ax) #'''
    '''print(calculate_accuracy(test_labels, pred_class)) #'''
    #print('>>>>><<<<',pred_class.dtype)
    return pred_class

def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    cov_mat = np.cov(train_set, rowvar=False)
    e_val, e_vecs = np.linalg.eig(cov_mat)
    # print(e_val)
    order = np.flip(np.argsort(e_val))
    f1 = order[0]+1
    f2 = order[1]+1
    se_val = np.array(np.zeros(n_components))
    se_vecs = np.array(np.zeros((13,n_components)))
    for i in range(0, n_components):
        se_val[i] = e_val[order[i]]
        se_vecs[:,i] = e_vecs[:, order[i]]
    pca_train = np.dot(train_set, se_vecs)
    pca_test = np.dot(test_set, se_vecs)
    pred_class = knn_all(pca_train, pca_test, k)
    '''plot_pca(pca_train, f1, f2) #'''
    '''confusion_ma = calculate_confusion_matrix(test_labels, pred_class)
    fig, fig_ax = plt.subplots()
    plot_matrix(confusion_ma,ax = fig_ax) #'''
    '''print(calculate_accuracy(test_labels, pred_class)) #'''
    #print('>>>>><<<<',pred_class.dtype)
    return pred_class

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')
    args = parser.parse_args()
    mode = args.mode[0]
    return args, mode

if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line
    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path,
                                                                       train_labels_path=args.train_labels_path,
                                                                       test_set_path=args.test_set_path,
                                                                       test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        predictions = knn(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))
