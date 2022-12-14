using PyCall

include("Classifier.jl")

# train the ensemble by individually training the base learners and using their predictions
# on the holdout dataset to train the metalearner, which combines their predictions.
function train_ensemble(datasets, validation, holdout, nfevs, depth)

    # for each base learner's training dataset, train the base learner and store the 
    # optimization results.
    results::Vector{Tuple{Vector{}, Vector{}, Vector{}}} = [([],[],[]) for i in 1:length(datasets)]
    @Threads.threads for (thread_number, i) in collect(enumerate(datasets))
        opt_res = train_learner(i, validation, "QAUM", nfevs ÷ length(datasets), depth)
        results[thread_number] = opt_res
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


    
        


