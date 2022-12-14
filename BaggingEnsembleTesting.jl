using DataFrames
using Queryverse
using Plots

include("Proc.jl")
include("datasets.jl")
include("BaggingEnsembleMethods.jl")
include("SingleLearnerTesting.jl")

include("Plotting.jl")


function get_training_params_from_results(results)
    training_params = []
    for i in results
        push!(training_params, i[3])
    end
    return training_params
end

#=
dataset = dataset_map["cancer"]

train_X = dataset.training_samples
train_Y = dataset.training_labels

valid_X = dataset.validation_samples
valid_Y = dataset.validation_labels

for i in eachindex(train_X)
    push!(train_X[i], train_Y[i])
end

for i in eachindex(valid_X)
    push!(valid_X[i], valid_Y[i])
end

=#
#n_learners = 7
#depth = 2
#nfevs = 12000

#train_X, valid_X, train_Y, valid_Y = fetch_datasets(100, 3)

# get the datasets
#partitioned_datasets, holdout = partition_datasets(train_X, n_learners, 1)

#@time metalearner, trained_weights, results = train_ensemble(partitioned_datasets, valid_X, holdout, nfevs, depth)

#println("Done training ensemble of ",n_learners)

# process the results to get the parameters created over training
#training_params = get_training_params_from_results(results)


#train_accs, valid_accs, prediction = ensemble_accuracy_over_training(holdout, valid_X, metalearner, training_params, depth)

#println("Ensemble Training Accuracy: ",train_accs[end])
#println("Ensemble Validation Accuracy: ",valid_accs[end])



#plot(1:nfevs ÷ n_learners, train_accs, title="Training accuracy over training", leg=false,minorticks=4,ylim=(0,100),dpi=200)
#savefig("EnsembleTrainingAccuracy.png")

#plot(1:nfevs ÷ n_learners, valid_accs, title="Validation accuracy over training", leg=false,minorticks=4,ylim=(0,100),dpi=200)
#savefig("EnsembleValidationAccuracy.png")

#println("Ensemble Confusion Matrix:\n")
#println(get_confusion_matrix(valid_Y, prediction))

#sklearn_metrics = pyimport("sklearn.metrics")
#ensemble_scores = get_ensemble_scores(metalearner, trained_weights, valid_X, 2)[:,2]

#roc_curve_ensemble = sklearn_metrics.roc_curve(valid_Y, ensemble_scores)

#plot(roc_curve_ensemble[1], roc_curve_ensemble[2])
#savefig("EnsembleROC.png")

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

    accuracies = Dict("1a"=>[],"1b"=>[],"2a"=>[],"2b"=>[],"3a"=>[],"3b"=>[],"4a"=>[],"4b"=>[],"5a"=>[],"5b"=>[])


    function score_diff(X_1, X_2, iteration)

        println("Count $counter\n--------------------------------------")

        # train the ensemble classifier on the first training dataset
        partitioned_datasets, holdout = partition_datasets(X_1, n_learners, 1)
        @time metalearner, trained_weights, results = train_ensemble(partitioned_datasets, X_1, holdout,nfevs, depth)
        println("   Done training ensemble")

        # train the single learner on the second training dataset
        @time validation_losses, training_losses, params = train_original_learner(X_2, X_2, "QAUM", nfevs, 2)
        println("   Done training original classifier")

        # classify the alternate dataset using each classifier and record the accuracy
        ensemble_score = classification_accuracy(ensemble_classify(metalearner, trained_weights, X_2, depth), X_2)
        counter +=1 
        ensemble_sc += ensemble_score
       
        println("Ensemble accuracy         : ",ensemble_score)
        single_score = classification_accuracy(classify(X_1, "QAUM", 2, params[end]), X_1)
        single_sc += single_score
        println("Single Classifier accuracy: ",single_score,"\n")

        push!(accuracies[string(iteration)*"a"], ensemble_score)
        push!(accuracies[string(iteration)*"b"], single_score)

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

        score_diff_1 = score_diff(X_1, X_2, i)
        score_diff_2 = score_diff(X_2, X_1, i)
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
    return Float64(pvalue), accuracies
end

train_X, valid_X, train_Y, valid_Y = fetch_datasets(200, 1)
p_value, accs = paired_ttest_5x2_cv(train_X, 7, 12000, 2)

print("The p-value obtained is ", p_value)

#create_bar_plot_of_5x2_cv(accs)