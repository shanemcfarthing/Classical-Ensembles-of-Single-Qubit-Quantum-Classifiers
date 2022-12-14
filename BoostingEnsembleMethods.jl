using PyCall

include("Classifier.jl")

# train the ensemble by individually training the base learners and using their predictions
# on the holdout dataset to train the metalearner, which combines their predictions.
    # Repeat this process for each learner:
    #   Train the learner on the dataset
    #   Use the learner to classify the dataset
    #   Use losses to recalculate sampling weights with buckets
    #   Resample the dataset

    # use each learner to classify the holdout dataset
    # combine predictions and true labels into training dataset for the metalearner
    # train the metalearner with that dataset
    # return the same things as the other ensemble
function train_boosting_ensemble(dataset, validation, holdout, nlearners, nfevs, depth)

    training_data = dataset
    weights = [Float64(1/length(training_data)) for i in 1:length(training_data)]
    results = []

    for i in 1:nlearners
        println("Learner $i")
        dataset_size = length(training_data)

        # train a learner on the training data and store the results of training
        opt_res = train_learner(training_data, validation, "QAUM", nfevs ÷ nlearners, depth)
        push!(results, opt_res)

        # classify the dataset with the trained learner so that the labels can be used to
        # identify the data points that are misclassified.
        pred_labels = classify(training_data, "QAUM", depth, opt_res[3][end])

        incorrect_prediction::Vector{Int8} = [0 for i in 1:dataset_size]
        # tally the number of misclassifications
        @Threads.threads for data_point in 1:dataset_size
            if training_data[data_point][9] != pred_labels[data_point]
                incorrect_prediction[data_point] = 1
            end
        end

        total_errors = sum(incorrect_prediction)
        if total_errors == 0 || total_errors == dataset_size
            break
        end

        #=
        for instance in 1:length(training_data)
            if training_data[instance][9] == pred_labels[instance]
                push!(incorrect_prediction, 0)
            else
                push!(incorrect_prediction, 1)
                total_errors += 1
            end
        end
        =#

        # reassign weights such that they are increased for incorrectly classified data points 
        # and decreased for correctly classified data points.
        @Threads.threads for j in eachindex(weights)
            if incorrect_prediction[j] == 0
                weights[j] = weights[j] * (1/sqrt((1-(total_errors/dataset_size))/(total_errors/dataset_size)))
            else
                weights[j] = weights[j] * sqrt((1-(total_errors/dataset_size))/(total_errors/dataset_size))
            end
        end

        # Create the buckets for each data point. These buckets will be used to sample the new
        # dataset according to the new weights of the data points.
        bucket::Vector{Float64} = [0.0 for i in 1:dataset_size]
        for j in eachindex(bucket)
            # the buckets are set with the upper range of the bucket for each data point.
            if j == 1
                bucket[j] = weights[j]
            else
                bucket[j] = weights[j] + bucket[j-1]
            end
        end
        
        #normalize the buckets so that buckets cover range from 0 to 1, while preserving proportion
        bucket = bucket ./ bucket[end]

        # get the selection of random numbers to be checked against the buckets for the 
        # selection of the data points for the new dataset.
        selection = rand(Float64, dataset_size)

        # sample the new dataset according to the buckets generated from the new data point
        # weights.
        new_training_data::Vector{Vector} = [[] for j in 1:dataset_size]
        @Threads.threads for (index, j) in collect(enumerate(selection))
            # find the bucket that the random number fits into, thus selecting the corresponding
            # data point
            for b in eachindex(bucket)      #len(bucket) == len(weights) == len(dataset)
                if j < bucket[b]
                    new_training_data[index] = training_data[b]
                    break
                end
            end

            if length(new_training_data[index]) == 0
                println("Selection $index is $j. Buckets are \n$bucket")
                throw("Index $index has an empty datapoint")
            end
        end

        # reassign the training dataset for the next learner
        training_data = new_training_data
        
        @Threads.threads for datap in training_data
            if length(datap) == 0
                println("Empty data point found")
            end
        end
    end

    # process the optimization result to access the final parameters for each base learner.
    trained_weights = [i[3][end] for i in results]

    # use the trained base learners to classify the holdout dataset, and combine their
    # classifications to build the metalearner's training dataset.
    combined = combine_learner_predictions(holdout, trained_weights, depth)
    
    holdout_labels = []
    for i in 1:length(holdout)
        push!(holdout_labels, holdout[i][9])
    end

    # create the metalearner object.
    metalearner = pyimport("sklearn.linear_model").LogisticRegression()

    # train the metalearner on the predictions of the base learners
    metalearner.fit(combined, holdout_labels)

    return metalearner, trained_weights, results
end



# use the ensemble to classify the given dataset. Needs the trained metalearner object, 
# the parameters of the base learners, the dataset to be classified, and the depth of the 
# circuit.
function ensemble_classify(metalearner, trained_weights, dataset, depth)

    # classify the dataset using each of the base learners and use these predictions to create the 
    # metalearner dataset.
    combined_learner_predictions = combine_learner_predictions(dataset, trained_weights, depth)

    # use the metalearner to classify the predictions of the base learners, thereby classifying the
    # original dataset.
    return metalearner.predict(combined_learner_predictions)

end

function get_ensemble_scores(metalearner, trained_weights, dataset, depth)
    # classify the dataset using each of the base learners and use these predictions to create the 
    # metalearner dataset.
    combined_learner_predictions = combine_learner_predictions(dataset, trained_weights, depth)

    # use the metalearner to classify the predictions of the base learners, thereby classifying the
    # original dataset.
    return metalearner.predict_proba(combined_learner_predictions)
end

# returns two vectors, one containing the ensemble accuracy on the training data at each step of the
# training, and the other containing the ensemble accuracy on the validation data at each step.
#
# NOTE: The training dataset should be the holdout dataset used during training.
#
function ensemble_accuracy_over_training(training_data, validation_data, metalearner, parameters, depth)

    # get the ensemble training accuracies at each epoch
    num_epochs = length(parameters[1])
    train_accs::Vector{} = [0.0 for i in 1:num_epochs]
    valid_accs::Vector{} = [0.0 for i in 1:num_epochs]
    final_pred = nothing

    for i in 1:num_epochs
        # use each base learner's parameter for epoch i to create the ensemble parameter set
        params::Vector{Vector{Float64}} = [lp[i] for lp in parameters]

        ensemble_train_preds = ensemble_classify(metalearner, params, training_data, depth)
        ensemble_valid_preds = ensemble_classify(metalearner, params, validation_data, depth)

        if i == num_epochs
            final_pred = ensemble_valid_preds
        end

        train_acc = 0
        valid_acc = 0


        for e in 1:length(training_data)
                
            # increase the accuracy by 1 if the instance was correctly classified
            if ensemble_train_preds[e] == training_data[e][9]
                train_acc += 1                
            end
        end

        for e in 1:length(validation_data)
                
            if ensemble_valid_preds[e] == validation_data[e][9]
                valid_acc += 1                
            end
        end
        

        train_acc = train_acc/length(training_data) * 100
        valid_acc = valid_acc/length(validation_data) * 100

        train_accs[i] = train_acc
        valid_accs[i] = valid_acc

        #push!(train_accs, copy(train_acc))
        #push!(valid_accs, copy(valid_acc))

    end

    return train_accs, valid_accs, final_pred
end
