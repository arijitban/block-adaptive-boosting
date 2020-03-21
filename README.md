# block-adaptive-boosting
On a Block Adaptive Approach to Continual Learning.

Execute the program as follows:
python block_adaptive_pipeline.py -b 100 -s 1 -a 4 -n 10
-b = block size -- size of the blocks when training block adaptive boosted ensemble algorithm
-s = num_stumps_per_block -- number of weak learners to use for each block for the block adaptive boosted ensemble
-a = num_stumps_for_classic_adaboost - number of weak leaners to use for the classic adaboost algorithm
-n = num_iterations - number of iterations the training and testing is run
result.csv will contain the results of each iteration
Also looks for prints like the following in the console:
<the following is printed for each iteration>
*** Testing Accuracy On Test Data - Dataset : 1
INFO:block_adaptive_pipeline:      ML BLOCK ACCURACY: 93.7062937062937
INFO:block_adaptive_pipeline:      ONE OFF ADABOOST ACCURACY: 88.11188811188812
INFO:block_adaptive_pipeline:      NON SCALED ENSEMBLE ACCURACY: 88.11188811188812
INFO:block_adaptive_pipeline:
*******
<the following is printed for averaged over all iterations>
INFO:block_adaptive_pipeline:
*** Testing Accuracy On Test Data - Iterations : 10
INFO:block_adaptive_pipeline:      ML BLOCK ACCURACY CUMULATIVE: 93.7062937062937
INFO:block_adaptive_pipeline:      ONE OFF ADABOOST ACCURACY_CUMULATIVE: 88.11188811188812
INFO:block_adaptive_pipeline:      NON SCALED ENSEMBLE ACCURACY_CUMULATIVE: 88.11188811188812
INFO:block_adaptive_pipeline:
*******
block_adaptive_boost_lib.py : This file has all the functions necessary for the operation of block adaptive and classic adaboost. It is used as a library in the main file.
block_adaptive_pipeline.py : This is the main file that implements the pipeline.
The following lines specifies the pipeline.
train_data_pipeline = [‘data-set1.csv', ‘data-set2.csv', ‘data-set3.csv']
train_label_pipeline = [‘label-data-set1.csv',‘label-data-set1.csv’,‘label-data-set1.csv']
Note that, we can implement any pipeline with any data by modifying the contents of the above.

