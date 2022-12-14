include("OriginalClassifier.jl")
include("Proc.jl")
include("datasets.jl")

using Plots

# print to make previous program output distinguishable from current output
println("------------------------------------------------------------------------")

dataset = dataset_map["digits"]

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

# get the datasets
#train_X, valid_X, train_Y, valid_Y = fetch_datasets(100, 3)


# train the classifier with the training data, and time the result.
#@time validation_losses, training_losses, params = train_original_learner(train_X, valid_X, "QAUM",12000, 2)

# plot the classifier's accuracy over training on the validation dataset

#accuracies, final_prediction = mean_accuracy_over_training(params, valid_X, valid_Y, 2)
#println("Classifier validation accuracy: ",accuracies[end],"%")

#plot(1:length(params), accuracies, title="Validation accuracy over training", leg=false,minorticks=4,ylim=(0,100),dpi=200)
#savefig("OriginalClassifierValidationAccuracy.png")

# plot the classifier's loss over training on the validation dataset
#plot(validation_losses, title="Validation loss over training", leg=false,dpi=200)
#savefig("ValidationLoss.png")

# plot the classifier's accuracy over training on the training dataset
#plot(1:length(params), mean_accuracy_over_training(params, train_X, train_Y, 2), title="Training accuracy over training", leg=false,minorticks=4,ylim=(0,100),dpi=200)
#savefig("OriginalClassifierTrainingAccuracy.png")

# plot the classifier's loss over training on the validation dataset
#plot(validation_losses, title="Training loss over training", leg=false,dpi=200)
#savefig("TrainingLoss.png")