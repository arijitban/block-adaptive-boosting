#####################################################################################################################################################
#
#
#   BLOCK ADAPTIVE BOOSTED EMSEMBLE
#
#   An Adaptive approach to Continual Learning
#   •	Use a block based approach where training data is used in blocks, and one or more weak learners are created per block of training data
#   •	Use the ensemble approach where weak learners from previous blocks are boosted by the learners created from the latest block of data
#   •	Update the “say” (or “weight”) associated with each of the existing weak learners in light of new training data, in an iterative way
#   •	Set up the iteration in a way that makes learners that become less relevant (due to dynamic evolution of the underlying data)
#       converge to smaller “say” values in order to effectively track a dynamically changing environment
#
#   block_adaptive_pipleline.py: This is the main "pipeline" program that calls "block_adaptive_lib" for the algorithm functions
#
#   Authors: Deepak Das, Arijit Banerjee
#
#   Copyright 2020 Deepak Das, Arijit Banerjee
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
#   (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,
#   merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
#   is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
#   WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
#   OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#####################################################################################################################################################

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_hastie_10_2
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import copy
import math
import random
import logging
import sys
import block_adaptive_boost_lib
import argparse



logger = logging.getLogger("block_adaptive_pipeline")
logging.basicConfig(level = logging.INFO)

train_data_pipeline = ['breast_cancer.csv']
train_label_pipeline = ['breast_cancer_label.csv']


basefolder = 'data'


def load_data_from_csv(filename, skiprow=0):
    logger.info(f'READING {filename}')
    filename = basefolder + '/' + filename
    raw_data = open(filename, 'rt')
    # load the CSV file as a numpy matrix
    dataset = np.loadtxt(raw_data, delimiter=",", skiprows=skiprow)
    print(dataset.shape)
    return dataset


if __name__ == '__main__':
    
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--block_size_arg', nargs=1)
    parser.add_argument('-s', '--num_stumps_per_block_arg', nargs=1)
    parser.add_argument('-a', '--num_stumps_in_default_adaboost_arg', nargs=1)
    parser.add_argument('-n', '--num_iterations_arg', nargs=1)

    args = parser.parse_args()

    if not args.block_size_arg:
        raise Exception('No block_size entered')
    else:
        block_size = int(args.block_size_arg[0])

    if not args.num_stumps_per_block_arg:
        raise Exception('No num_stumps_per_block entered')
    else:
        num_stumps_per_block = int(args.num_stumps_per_block_arg[0])

    if not args.num_stumps_in_default_adaboost_arg:
        raise Exception('No  num_stumps_in_default_adaboost entered')
    else:
        num_stumps_in_default_adaboost = int(args.num_stumps_in_default_adaboost_arg[0])

    if not args.num_iterations_arg:
        raise Exception('No of iterations not entered entered')
    else:
        num_iterations = int(args.num_iterations_arg[0])
    
    results_file = open('results.csv','w')
    results_file.write('Iteration,block adaptive accuracy,one off adaboost accuracy,non scaled ensemble accuracy,block_size,num_stumps_per_block,num_stumps_in_adaboost\n')

    pred_block_adaptive_cum_accuracy = 0.0
    pred_one_off_adaboost_cum_accuracy = 0.0
    pred_non_scaled_ensemble_cum_accuracy = 0.0
    total_test_data_length = 0
    total_num_iterations = 0

    logger.info(f'num_iterations : {num_iterations}') 

    for k in range(num_iterations):

        pipeline_depth = len(train_data_pipeline)
        non_scaled_ensemble_agg_stump_list = []
        non_scaled_ensemble_aggr_stump_weight_list = []
        clf = DecisionTreeClassifier(max_depth = 1)
        
        block_adaptive_classfier_obj = block_adaptive_boost_lib.BlockAdpativeBoostClassifier(block_size, num_stumps_per_block, base_classifier='DecisionTree') 

        for i in range(pipeline_depth):
            
            X = load_data_from_csv(train_data_pipeline[i],1)
            Y = load_data_from_csv(train_label_pipeline[i])
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
     
            ##First Iteration
            if i==0:
                ##TRAIN BLOCK ADAPTIVE ADABOOST
                (aggregate_stump_collection_all_blocks, aggregate_stump_amount_of_say_all_blocks) = block_adaptive_classfier_obj.fit(X_train,Y_train, X_test, Y_test)
                block_adaptive_classfier_obj.save()

                ##   TRAIN  NORMAL ADABOOST
                one_off_adaboost_stumps, one_off_adaboost_stump_weights = block_adaptive_classfier_obj.adaboost_clf(Y_train, X_train, Y_test, X_test,num_stumps_in_default_adaboost)
                non_scaled_ensemble_agg_stump_list.append(copy.deepcopy(one_off_adaboost_stumps))
                non_scaled_ensemble_aggr_stump_weight_list.append(copy.deepcopy(one_off_adaboost_stump_weights))

            ## Keep Adapting in subsequent iterations
            else:
                block_adaptive_classfier_obj = block_adaptive_boost_lib.BlockAdpativeBoostClassifier(block_size, num_stumps_per_block, base_classifier='DecisionTree') 
                block_adaptive_classfier_obj.load()
                (aggregate_stump_collection_all_blocks, aggregate_stump_amount_of_say_all_blocks) = block_adaptive_classfier_obj.fit(X_train,Y_train, X_test, Y_test)
                block_adaptive_classfier_obj.save()

                ##   TRAIN  NORMAL ADABOOST
                non_scaled_ensemble_i_stumps, non_scaled_ensemble_i_stump_weights = block_adaptive_classfier_obj.adaboost_clf(Y_train, X_train, Y_test, X_test, num_stumps_in_default_adaboost)
                non_scaled_ensemble_agg_stump_list.append(copy.deepcopy(non_scaled_ensemble_i_stumps))
                non_scaled_ensemble_aggr_stump_weight_list.append(copy.deepcopy(non_scaled_ensemble_i_stump_weights))
            
            
            ##CALC BLOCK ADAPTIVE ACCURACY
            pred_block_adaptive_results = block_adaptive_classfier_obj.predict(X_test)
            pred_block_adaptive_accuracy = accuracy_score(pred_block_adaptive_results,Y_test)*100.0

            pred_block_adaptive_cum_accuracy += pred_block_adaptive_accuracy*len(X_test)

            
            ##CALC ONE OFF ADABOOST ACCURACY
            pred_one_off_adaboost_results = block_adaptive_classfier_obj.predict_aggregate_blocks(X_test, [one_off_adaboost_stumps], [one_off_adaboost_stump_weights])
            pred_one_off_adaboost_accuracy = accuracy_score(pred_one_off_adaboost_results,Y_test)*100.0

            pred_one_off_adaboost_cum_accuracy += pred_one_off_adaboost_accuracy*len(X_test)
            
            ##CALC RETRAINED and CUMULATIVE NON SCALED ENSEMBLE ACCURACY
            pred_non_scaled_ensemble_results = block_adaptive_classfier_obj.predict_aggregate_blocks(X_test, non_scaled_ensemble_agg_stump_list,  non_scaled_ensemble_aggr_stump_weight_list)
            pred_non_scaled_ensemble_accuracy = accuracy_score(pred_non_scaled_ensemble_results,Y_test)*100.0

            pred_non_scaled_ensemble_cum_accuracy += pred_non_scaled_ensemble_accuracy*len(X_test)

            total_test_data_length += len(X_test)

            total_num_iterations +=1
            
            ## TEST ACCURACY ON TEST DATA
            logger.info(f'\n\n*** Testing Accuracy On Test Data - Dataset : {i+1} ')
            logger.info(f'      ML BLOCK ACCURACY: {pred_block_adaptive_accuracy}')
            logger.info(f'      ONE OFF ADABOOST ACCURACY: {pred_one_off_adaboost_accuracy}')
            logger.info(f'      NON SCALED ENSEMBLE ACCURACY: {pred_non_scaled_ensemble_accuracy}')
            logger.info(f'\n*******\n\n')
            outstr = str(i) + ',' + str(pred_block_adaptive_accuracy) + ',' + str(pred_one_off_adaboost_accuracy) + ',' + str(pred_non_scaled_ensemble_accuracy) + ',' + str(block_size) + ',' + str(num_stumps_per_block) + ',' + str(num_stumps_in_default_adaboost) + '\n'
            results_file.write(outstr)

    results_file.close()
    logger.info(f'\n\n*** Testing Accuracy On Test Data - Iterations : {total_num_iterations}') 
    logger.info(f'      ML BLOCK ACCURACY CUMULATIVE: {pred_block_adaptive_cum_accuracy/total_test_data_length}')
    logger.info(f'      ONE OFF ADABOOST ACCURACY_CUMULATIVE: {pred_one_off_adaboost_cum_accuracy/total_test_data_length}')
    logger.info(f'      NON SCALED ENSEMBLE ACCURACY_CUMULATIVE: {pred_non_scaled_ensemble_cum_accuracy/total_test_data_length}')
    logger.info(f'\n*******\n\n') 




