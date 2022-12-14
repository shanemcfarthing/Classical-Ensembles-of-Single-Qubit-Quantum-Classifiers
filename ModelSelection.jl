using ScikitLearn.CrossValidation: KFold
using Random
using PyCall

include("SingleLearnerTesting.jl")
#include("BaggingEnsembleTesting.jl")
include("BoostingEnsembleTesting.jl")

function classification_accuracy(classified_labels, dataset)
    accuracy = 0
    for (p, o) in zip(classified_labels, dataset)
        if p == o[9]
            accuracy += 1
        end
    end
    return (accuracy/length(dataset))*100
end

function paired_ttest_5x2_cv(dataset, n_learners, nfevs, depth)
    counter = 1
    ensemble_sc = 0
    single_sc = 0
    function score_diff(X_1, X_2)

        # train the ensemble classifier on the first training dataset
        partitioned_datasets, holdout = partition_datasets(X_1, 1, 1)
        metalearner, trained_weights, results = train_boosting_ensemble(X_1, X_1, holdout,n_learners, nfevs, depth)
        println("   Done training ensemble")

        # train the single learner on the second training dataset
        validation_losses, training_losses, params = train_original_learner(X_2, X_2, "QAUM", nfevs, 2)
        println("   Done training original classifier")

        # classify the alternate dataset using each classifier and record the accuracy
        ensemble_score = classification_accuracy(ensemble_classify(metalearner, trained_weights, X_2, depth), X_2)
        println("Count $counter\n--------------------------------------")
        counter +=1 
        ensemble_sc += ensemble_score
       
        println("Ensemble accuracy         : ",ensemble_score)
        single_score = classification_accuracy(classify(X_1, "QAUM", 2, params[end]), X_1)
        single_sc += single_score
        println("Single Classifier accuracy: ",single_score,"\n")

        return ensemble_score - single_score
    end

    variance_sum = 0
    first_diff = nothing
    
    # conduct the two-fold cross validation five times
    for i in 1:5
        
        # split the original dataset in half, with a random selection
        dataset = Random.shuffle(dataset)
        X_1 = copy(dataset[1:Int64(length(dataset)*0.5)])
        X_2 = copy(dataset[Int64(length(dataset)*0.5)+1:length(dataset)])

        score_diff_1 = score_diff(X_1, X_2)
        score_diff_2 = score_diff(X_2, X_1)
        score_mean = 0.5*(score_diff_1 + score_diff_2)
        score_var = (score_diff_1 - score_mean)^2 + (score_diff_2 - score_mean)^2

        variance_sum += score_var
        if isnothing(first_diff)
            first_diff = score_diff_1
        end
    end

    numerator = first_diff
    denominator = sqrt(1/(5*variance_sum))
    t_stat = numerator / denominator
    println("\nEnsemble Score: $ensemble_sc")
    println("Single score: $single_sc\n")
    pvalue = pyimport("scipy").stats.t.sf(abs(t_stat), 5) * 2
    return Float64(pvalue)
end


# obtain a dataset to use for the hypothesis testing 
X, V, X_labels, V_labels = fetch_datasets(100, 3)
p_value = paired_ttest_5x2_cv(X, 130, 1300, 6)

print("The p-value obtained is ", p_value)