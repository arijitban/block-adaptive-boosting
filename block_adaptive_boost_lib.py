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
#   block_adaptive_lib.py: This is called by block_adaptive__pipeline.py
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
import pickle
#import matplotlib.pyplot as plt

logger = logging.getLogger("adaboost_pipeline")
logging.basicConfig(level = logging.INFO)


class BlockAdpativeBoostClassifier:

    def __init__(self, block_size, num_stumps_per_block, base_classifier='DecisionTree', mu=0.5, ss=1.0):
        self.adaboost_original_accuracy = 0
        self.iteration_count = 0
        self.mu = mu
        self.ss = ss
        ### Varibale to store the stumps from different blocks. 
        ### It a list of list (2D array) where each row corresponds to the results of a block - 
        ### and each column corresponds to a stump in that block. #### 
        self.aggregate_stump_collection_all_blocks = []
        
        ### Varibale to store the amount of say per stump from different blocks. 
        ### It a list of list (2D array) where each row corresponds to the results of a block - 
        ### and each column corresponds to the amount of say of the corresponding stump 
        ### in that block. It has a one to one mapping to the 
        ### aggregate_stump_collection_all_blocks array#### 
        self.aggregate_stump_amount_of_say_all_blocks = []
        
        self.block_size = block_size
        self.num_stumps_per_block = num_stumps_per_block
        self.base_classifier = base_classifier
        self.filename = 'model.dump'

    
    
    def load(self):
        with open(self.filename, 'rb') as f:
            tmp_dict = pickle.load(f)
            self.__dict__.update(tmp_dict) 
    
    def save(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.__dict__, f, 2)
    
    def weighted_choice(self,weights):
        """ returns randomly an element from the sequence of 'objects', 
            the likelihood of the objects is weighted according 
            to the sequence of 'weights', i.e. percentages."""
    
        weights = np.array(weights, dtype=np.float64)
        weights = weights.cumsum()
        weights[-1] = 1.0
        logger.debug('weights' + str(len(weights)))
        logger.debug(weights)
        x = random.random()
        logger.debug('Selected Random : ' + str(x))
        for i in range(len(weights)):
            if x < weights[i]:
                return i
    
    
    """ ADABOOST IMPLEMENTATION ================================================="""
    """ Reference :https://github.com/jaimeps/adaboost-implementation/blob/master/adaboost.py """
    def adaboost_clf(self, Y_train, X_train, Y_test, X_test, M):
        n_train, n_test = len(X_train), len(X_test)
        # Initialize weights
        w = np.ones(n_train) / n_train
        pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
        stumps = []
        stump_weights = []
   
        X_train_prev = X_train
        Y_train_prev = Y_train
        
        for i in range(M):

            X_train_new = []
            Y_train_new = []
            if i>0:
                for index in range(n_train):
                    sample_index = self.weighted_choice(w)
                    logger.debug(sample_index)
                    X_train_new.append(X_train_prev[sample_index])
                    Y_train_new.append(Y_train_prev[sample_index])
                Y_train_prev = Y_train_new
                X_train_prev = X_train_new
            else:
                X_train_new = X_train_prev
                Y_train_new = Y_train_prev

            clf = DecisionTreeClassifier(max_depth=1)
            
            clf.fit(X_train_new, Y_train_new)            

            pred_train_i = clf.predict(X_train_new)
            pred_test_i = clf.predict(X_test)

            # Indicator function
            miss = [int(x) for x in (pred_train_i != Y_train_new)]
            # Equivalent with 1/-1 to update weights
            miss2 = [x if x==1 else -1 for x in miss]
            # Error
            if(sum(w)) != 0:
                err_m = np.dot(w,miss) / sum(w)
                err_m = max(err_m,0.000000001)
                err_m = min(err_m,0.999999999)
                # Alpha : This is the amount of say
                alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))
                # New weights
                w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
                w = w/sum(w)

            if(sum(w)) == 0:
                break
            
            # Add to prediction
            pred_train = [sum(x) for x in zip(pred_train, 
                                              [x * alpha_m for x in pred_train_i])]
            pred_test = [sum(x) for x in zip(pred_test, 
                                             [x * alpha_m for x in pred_test_i])]
            stumps.append(copy.deepcopy(clf))
            stump_weights.append(copy.deepcopy(alpha_m))
        
        pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
        self.adaboost_original_accuracy = accuracy_score(pred_test,Y_test)*100.0
        # Return stumps and amount of say 
        return (stumps, stump_weights)
    
    """ ADABOOST BLOCK IMPLEMENTATION ================================================="""
    def adaboost_clf_block(self, Y_train, X_train, Y_test, X_test, M, sample_weights=None):
    
        n_train, n_test = len(X_train), len(X_test)
        # Initialize weights
        w = np.ones(n_train) / n_train
        pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
        block_stump_list = []
        block_stump_amount_of_say = []

        X_train_prev = X_train
        Y_train_prev = Y_train
        
        for i in range(M):
            X_train_new = []
            Y_train_new = []

            #uncomment this if you want re-distibution across blocks
            #if self.iteration_count>0:

            if i>0:
                for index in range(n_train):
                    sample_index = self.weighted_choice(w)
                    logger.debug(sample_index)
                    X_train_new.append(X_train_prev[sample_index])
                    Y_train_new.append(Y_train_prev[sample_index])
                Y_train_prev = Y_train_new
                X_train_prev = X_train_new
            else:
                X_train_new = X_train_prev
                Y_train_new = Y_train_prev

            self.iteration_count += 1

            #clf = AdaBoostClassifier(n_estimators=M,algorithm='SAMME.R')
            #clf = AdaBoostClassifier(n_estimators=50,algorithm='SAMME.R')
            if self.base_classifier == 'AdaBoost':
                clf = AdaBoostClassifier()
            else:
                clf = DecisionTreeClassifier(max_depth=1)
            clf.fit(X_train_new, Y_train_new)

            pred_train_i = clf.predict(X_train_new)
            pred_test_i = clf.predict(X_test)

            # Indicator function
            miss = [int(x) for x in (pred_train_i != Y_train_new)]
            # Equivalent with 1/-1 to update weights
            miss2 = [x if x==1 else -1 for x in miss]
            # Error
            if(sum(w)) != 0:
                err_m = np.dot(w,miss) / sum(w)
                err_m = max(err_m,0.000000001)
                err_m = min(err_m,0.999999999)
                # Alpha : This is the amount of say
                alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))
                # New weights
                w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
                w = w/sum(w)

            if(sum(w)) == 0:
                break

            block_stump_list.append(copy.deepcopy(clf))
            block_stump_amount_of_say.append(alpha_m)

            # Add to prediction
            pred_train = [sum(x) for x in zip(pred_train, 
                                              [x * alpha_m for x in pred_train_i])]
            pred_test = [sum(x) for x in zip(pred_test, 
                                             [x * alpha_m for x in pred_test_i])]
        
        pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    
        # Return 
        return block_stump_list, block_stump_amount_of_say 
    
    
    ### Function to Calculate the preidictions for a new dataset based on all the trained stumps of a particular block
    def block_model_predict(self, X_train, block_stump_list, block_stump_amount_of_say):
        n_train = len(X_train)
        pred_train = np.zeros(n_train)
        for i in range(len(block_stump_list)):
            clf = block_stump_list[i]
            alpha_m = block_stump_amount_of_say[i]
            pred_train_i = clf.predict(X_train)
            pred_train = [sum(x) for x in zip(pred_train, 
                                              [x * alpha_m for x in pred_train_i])]
        pred_train = np.sign(pred_train)
        return pred_train
    
    ### Calculate overall preidictions for a test dataset based on all the trained stumps of all the blocks
    def predict_aggregate_blocks(self, X_test, aggregate_stump_collection_all_blocks , aggregate_stump_amount_of_say_all_blocks):
        n_test = len(X_test)
        pred_test = np.zeros(n_test)
        for j in range(len(aggregate_stump_collection_all_blocks)):
            block_stump_list = aggregate_stump_collection_all_blocks[j]
            block_stump_amount_of_say = aggregate_stump_amount_of_say_all_blocks[j]
            for i in range(len(block_stump_list)):
                clf = block_stump_list[i]
                alpha_m = block_stump_amount_of_say[i]
                pred_test_i = clf.predict(X_test)
                pred_test = [sum(x) for x in zip(pred_test, 
                                              [x * alpha_m for x in pred_test_i])]
        pred_test = np.sign(pred_test)
        return pred_test
    
    ### Calculate overall preidictions for a test dataset based on all the trained stumps of all the blocks
    def predict(self, X_test):
        n_test = len(X_test)
        pred_test = np.zeros(n_test)
        for j in range(len(self.aggregate_stump_collection_all_blocks)):
            block_stump_list = self.aggregate_stump_collection_all_blocks[j]
            block_stump_amount_of_say = self.aggregate_stump_amount_of_say_all_blocks[j]
            for i in range(len(block_stump_list)):
                clf = block_stump_list[i]
                alpha_m = block_stump_amount_of_say[i]
                pred_test_i = clf.predict(X_test)
                pred_test = [sum(x) for x in zip(pred_test, 
                                              [x * alpha_m for x in pred_test_i])]
        pred_test = np.sign(pred_test)
        return pred_test
    
    
    """ ADABOOST BLOCK Adjustment - Find appropiateness of previous model on the new block ================================================="""
    def compute_sample_weight(self, Y_train, X_train, block_stump_list,block_stump_amount_of_say):
        n_train = len(X_train)
        pred_train = np.zeros(n_train)
        w = np.ones(n_train) / n_train
    
        pred_train_i = self.block_model_predict(X_train, block_stump_list, block_stump_amount_of_say)
        
    
        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x==1 else -1 for x in miss]
        # Error
        err_m = np.dot(w,miss) / sum(w)
        err_m = max(err_m,0.000000001)
        err_m = min(err_m,0.999999999)

        
        # Alpha : This is the amount of say
        alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))

        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
    
        # Return 
        return w, alpha_m 
    
    
    def compute_sample_weight_aggregate(self, Y_train, X_train):
        n_train = len(X_train)
        pred_train = np.zeros(n_train)
        w = np.ones(n_train) / n_train
    
        pred_train_i = self.predict_aggregate_blocks(X_train, self.aggregate_stump_collection_all_blocks , self.aggregate_stump_amount_of_say_all_blocks)
        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x==1 else -1 for x in miss]
        # Error
        err_m = np.dot(w,miss) / sum(w)
        err_m = max(err_m,0.000000001)
        err_m = min(err_m,0.999999999)
        # Alpha : This is the amount of say
        alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))
        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        
        # Return 
        return w, alpha_m 
    
    """ FUNCTION TO FIT TRAIN DATA ON BLOCK ADAPTIVE CLASSIFIER ============="""
    def fit(self,X_train, Y_train, X_test=[], Y_test=[]):
        if X_test == []:
            X_test = X_train
        if Y_test == []:
            Y_test = Y_train
        if len(self.aggregate_stump_collection_all_blocks) == 0: ##First block
            self.adaboost_block_train_wrapper(X_train,Y_train, X_test, Y_test)
        else:
            self.adaboost_adapt_new_block(X_train,Y_train, X_test, Y_test)
        return self.aggregate_stump_collection_all_blocks,self.aggregate_stump_amount_of_say_all_blocks

    
    
    
    """ FUNCTION TO TRAIN BLOCK ADAPTIVE ADABOOST ============================================================="""
    
    def adaboost_block_train_wrapper(self, X_train,Y_train, X_test, Y_test):
    
        adaboost_library_clf = AdaBoostClassifier()
        adaboost_library_clf.fit(X_train, Y_train)
    
        adaboost_library_predict = adaboost_library_clf.predict(X_test)
    
        logger.info('Adaboost SKLEARN Library Accuracy: ' + str(accuracy_score(adaboost_library_predict,Y_test)*100.0))
        
        num_blocks = max( int(math.floor(float(len(X_train))/self.block_size)), 1)
        
        logger.info('block_size: ' + str(self.block_size) + " num_blocks: " + str(num_blocks) + "num_stumps_per_block: " + str(self.num_stumps_per_block))
       
        # Fit a simple decision tree first
        #clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 0, max_leaf_nodes=2)
        clf_tree = DecisionTreeClassifier(max_depth = 1)
        # adaboost_clf(Y_train, X_train, Y_test, X_test, 10, clf_tree)
        
        x_range = range(num_blocks)
        
        
        prev_block_stump_list,prev_block_stump_amount_of_say = (None, None)
        
        ### Loop Through Data Blocks
        for i in x_range:    
            start_index = i * self.block_size
            if start_index < 0:
                start_index = 0
    
            if (start_index + 2*self.block_size) > len(X_train):   
                end_index = start_index + 2*self.block_size
            else:
                end_index = start_index + self.block_size
    
            
            self.ss = min(1, len(X_train[start_index:end_index])/self.block_size)
    
            ## If not first iteration the compute sample weights based on the result of the previous block
            if i != 0:
    
                i_range = range(i)
                for j in i_range:
                    prev_block_stump_list = self.aggregate_stump_collection_all_blocks[j]
                    prev_block_stump_amount_of_say = self.aggregate_stump_amount_of_say_all_blocks[j]
     
                    ### If it is not the first block we have a previous model. Check the appropriateness of the previous model \
                    ### to calculate the sample weights of the new block and the overall amount of say of the previous model \
                    ### when applied to the current training data block
                    sample_weight_i, amount_of_say_prev_model = self.compute_sample_weight(Y_train[start_index:end_index], X_train[start_index:end_index], prev_block_stump_list,prev_block_stump_amount_of_say)
                    logger.info(f'amount_of_say_prev_model: {amount_of_say_prev_model}')
    
                    self.aggregate_stump_amount_of_say_all_blocks[j] = np.multiply(float((1.0 - self.ss*self.mu)),self.aggregate_stump_amount_of_say_all_blocks[j]) + self.ss*self.mu*amount_of_say_prev_model
                    logger.info(f'aggregate_stump_amount_of_say_all_blocks[j] : {self.aggregate_stump_amount_of_say_all_blocks[j] }')
                    
    
                sample_weight_i, amount_of_say_prev_model = self.compute_sample_weight_aggregate(Y_train[start_index:end_index], X_train[start_index:end_index])
                
                ## Each block computation
                block_stump_list,block_stump_amount_of_say = self.adaboost_clf_block(Y_train[start_index:end_index], X_train[start_index:end_index][:], Y_test, X_test, self.num_stumps_per_block, sample_weight_i)
                self.aggregate_stump_collection_all_blocks.append(copy.deepcopy(block_stump_list))
                self.aggregate_stump_amount_of_say_all_blocks.append(copy.deepcopy(block_stump_amount_of_say))
                
            else:
                ## Each block computation
                block_stump_list,block_stump_amount_of_say = self.adaboost_clf_block(Y_train[start_index:end_index], X_train[start_index:end_index][:], Y_test, X_test, self.num_stumps_per_block)
                #logger.info(f'block_stump_list: {block_stump_list}')
                logger.info(f'block_stump_amount_of_say: {block_stump_amount_of_say}')           
    
                self.aggregate_stump_collection_all_blocks.append(copy.deepcopy(block_stump_list))
                self.aggregate_stump_amount_of_say_all_blocks.append(copy.deepcopy(block_stump_amount_of_say))
                #logger.info(f'aggregate_stump_collection_all_blocks: {aggregate_stump_collection_all_blocks}')
    
        
        ### Use all the stumps from all the blocks to make an aggregated prediction on the test data
        prediction_block_agg = self.predict(X_test)
        
        
        return(self.aggregate_stump_collection_all_blocks, self.aggregate_stump_amount_of_say_all_blocks)
    
    
    """ FUNCTION TO ADAPT A NEW BLOCK TO A PRE-TRAINED ADAPTIVE ADABOOST ============================================================="""
    
    def adaboost_adapt_new_block(self, X_train,Y_train, X_test, Y_test):
        
        adaboost_library_clf = AdaBoostClassifier()
        adaboost_library_clf.fit(X_train, Y_train)
    
        adaboost_library_predict = adaboost_library_clf.predict(X_test)
    
        logger.info('Adaboost SKLEARN Library Accuracy: ' + str(accuracy_score(adaboost_library_predict,Y_test)*100.0))
    
        num_blocks = max( int(math.floor(float(len(X_train))/self.block_size)), 1)
        
        logger.info('block_size: ' + str(self.block_size) + " num_blocks: " + str(num_blocks) + "num_stumps_per_block: " + str(self.num_stumps_per_block))
       
        x_range = range(num_blocks)
        
        ### Loop Through Data Blocks
        for i in x_range:    
            start_index = i * self.block_size
            if start_index < 0:
                start_index = 0
    
            if (start_index + 2*self.block_size) > len(X_train):   
                end_index = start_index + 2*self.block_size
            else:
                end_index = start_index + self.block_size
    
            self.ss = min(1,len(X_train[start_index:end_index])/self.block_size)
    
            i_range = range(len(self.aggregate_stump_amount_of_say_all_blocks))
            
            for j in i_range:
                
                prev_block_stump_list = self.aggregate_stump_collection_all_blocks[j]
                prev_block_stump_amount_of_say = self.aggregate_stump_amount_of_say_all_blocks[j]
     
                ### If it is not the first block we have a previous model. Check the appropriateness of the previous model \
                ### to calculate the sample weights of the new block and the overall amount of say of the previous model \
                ### when applied to the current training data block
                sample_weight_i, amount_of_say_prev_model = self.compute_sample_weight(Y_train[start_index:end_index], X_train[start_index:end_index], prev_block_stump_list,prev_block_stump_amount_of_say)
    
                self.aggregate_stump_amount_of_say_all_blocks[j] = np.multiply(float((1.0-self.ss*self.mu)), self.aggregate_stump_amount_of_say_all_blocks[j]) + self.ss*self.mu*amount_of_say_prev_model
                                
            sample_weight_i, amount_of_say_prev_model = self.compute_sample_weight_aggregate(Y_train[start_index:end_index], X_train[start_index:end_index])
            ## Each block computation
            block_stump_list,block_stump_amount_of_say = self.adaboost_clf_block(Y_train[start_index:end_index], X_train[start_index:end_index][:], Y_test, X_test, self.num_stumps_per_block, sample_weight_i)
            ## Add the stumps of the  last trained block
            self.aggregate_stump_collection_all_blocks.append(copy.deepcopy(block_stump_list))
            self.aggregate_stump_amount_of_say_all_blocks.append(copy.deepcopy(block_stump_amount_of_say))
    
        
        ### Use all the stumps from all the blocks to make an aggregated prediction on the test data
        prediction_block_agg = self.predict(X_test)
        
        
        ##Print the accuracy if we did not choose a block adaptive approach and ran Adaboost on the entire training data set
    
        return(self.aggregate_stump_collection_all_blocks, self.aggregate_stump_amount_of_say_all_blocks)
