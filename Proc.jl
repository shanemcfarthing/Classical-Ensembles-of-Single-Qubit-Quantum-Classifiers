using PyCall
using Random
using StatsBase

# classify a dataset using the given list of parameters and compare to supervised labels
# to determine mean accuracy. Gives a mean accuracy for each set of parameters.
function mean_accuracy_over_training(params, dataset, labels, depth)
    
    # classify the dataset using each parameter set and append the results to predictions.
    predictions = []
    for i in params
        push!(predictions, classify(dataset, "QAUM", depth, i))
    end
    
    # for each parameter set's classification, calculate the mean accuracy and append to 
    # accuracies list.
    accuracies = []
    for p in predictions
        inner_sum=0
        for (c,l) in zip(p,labels)
            inner_sum += (c==l)
        end
        push!(accuracies, inner_sum/length(p)*100)
    end

    return accuracies, predictions[end]

end

# returns randomly sampled datasets of size 500 and their supervised learning labels. Takes a 
# seed value for reproduciblity.
function fetch_datasets_for_python(sample_size::Int64, rand_seed::Int64)

    # read the training and validation datasets from the pulsars.csv file
    train_X, valid_X, train_Y, valid_Y = py"fetch_data_random_seed_val"(100,1)

    # anonymous function to change the structure of the data into a Vector{Vector{Float64}}
    row(m, i) = m[i, :]
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]

    # use the anonymous functions to format the training and validation datasets
    # also append each feature vectors label to the end of the vector, as this is the format used by the classifier
    m = to_rows(train_X)

    validation = to_rows(valid_X)

    # return the final datasets and labels
    return m, validation, train_Y, valid_Y
end

# returns randomly sampled datasets of size 500 and their supervised learning labels. Takes a 
# seed value for reproduciblity.
function fetch_datasets(sample_size::Int64, rand_seed::Int64)

    # read the training and validation datasets from the pulsars.csv file
    train_X, valid_X, train_Y, valid_Y = py"fetch_data_random_seed_val"(100,1)

    # anonymous function to change the structure of the data into a Vector{Vector{Float64}}
    row(m, i) = m[i, :]
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]

    # use the anonymous functions to format the training and validation datasets
    # also append each feature vectors label to the end of the vector, as this is the format used by the classifier
    m = to_rows(train_X)

    for (row, label) in zip(m, train_Y)
        push!(row, label)
    end

    validation = to_rows(valid_X)

    for (row, label) in zip(validation, valid_Y)
        push!(row, label)
    end 

    # return the final datasets and labels
    return m, validation, train_Y, valid_Y
end

py"""
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# returns sample of data from pulsar.csv to use for training and validation
# balances the number of data points from each class, which is not the case usually
def fetch_data_random_seed_val(n_samples, seed):

    # read the dataset file and create dataframe
    dataset = pd.read_csv('pulsar.csv')

    # store all data points belonging to class 0
    data0 = dataset[dataset[dataset.columns[8]] == 0]

    # take a sample of size=n_samples of data points from class 0
    data0 = data0.sample(n=n_samples, random_state=seed)

    # store the feature vectors and labels of these data points
    X0 = data0[data0.columns[0:8]].values
    Y0 = data0[data0.columns[8]].values

    # repeat the process for the data points from class 1
    data1 = dataset[dataset[dataset.columns[8]] == 1]
    data1 = data1.sample(n=n_samples, random_state=seed)
    X1 = data1[data1.columns[0:8]].values
    Y1 = data1[data1.columns[8]].values

    # combine the data points from both classes to create the dataset to be used
    X = np.append(X0, X1, axis=0)
    Y = np.append(Y0, Y1, axis=0)

    # scale the feature values to produce full Fourier series
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, np.pi))
    X = min_max_scaler.fit_transform(X)

    # Separate the test and training datasets
    train_X, validation_X, train_Y, validation_Y = train_test_split(X, Y, test_size=0.5, random_state=seed)

    return train_X, validation_X, train_Y, validation_Y
"""

function partition_datasets(training_data, npartitions,holdout_alloc)
    
    #the size of the holdout dataset calculated as a percentage of training dataset
    holdout_size = round(Int, holdout_alloc * length(training_data))

    #in order to take a random sample for the holdout dataset, we first choose the random indices
    hold_indices = rand(1:length(training_data),holdout_size)

    #build the holdout dataset using the corresponding elements of the training dataset
    holdout = [Any[] for i=1:holdout_size]
    
    count = 1
    for i in hold_indices
        holdout[count] = copy(training_data[i])
        count += 1
    end


    #this will hold npartitions many datasets
    partitioned_datasets = [Any[] for i in 1:npartitions]

    #obtain each dataset in the partition by sampling from the original dataset with replacement
    for i in 1:npartitions
        sampled_dataset = sample(training_data, length(training_data))
        partitioned_datasets[i] = copy(sampled_dataset)
    end

    return partitioned_datasets, holdout
    
end