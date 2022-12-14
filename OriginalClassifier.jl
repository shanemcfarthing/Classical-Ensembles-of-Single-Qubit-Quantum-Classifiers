using Yao
using PyCall
using NLopt, LinearAlgebra

include("RandomSeed.jl")

function get_confusion_matrix(actual_labels, predicted_labels)
    
    confusion_matrix = Dict("true_positive" => 0, "false_positive" => 0, "true_negative" => 0, "false_negative" => 0)
    for (pred_label, act_label) in zip(predicted_labels,actual_labels)
        if act_label == 0
            if pred_label == 0
                confusion_matrix["true_negative"] += 1
            else
                confusion_matrix["false_positive"] += 1
            end
        else
            if pred_label == 1
                confusion_matrix["true_positive"] += 1
            else
                confusion_matrix["false_negative"] += 1
            end
        end
    end
    return confusion_matrix

end

#creates the specified variational circuit with the given parameters
function get_var_circuit(config::String; depth=1)
    
    #use config string to determine which circuit to build
    if config == "QAUM"
        
        #functinons for quickly creating the blocks that the circuit is comprised of
        layer(nbit::Int, x::Symbol) = layer(nbit, Val(x))
        layer(nbit::Int, ::Val{:basic}) = chain(nbit, put(i=>chain(H, Rz(0), Rx(0), Ry(0))) for i = 1:nbit)
        layer(nbit::Int, ::Val{:embedding}) = chain(nbit, put(i=>chain(Rz(0), Rz(0), Rx(0), Ry(0))) for i = 1:nbit)

        #create a single qubit circuit, and add the first layer of gates
        circuit = chain(1)
        push!(circuit, layer(1, :basic))

        #use the depth argument to determine how many trainable blocks to create
        for i in 1:depth
            #add one embedding block for every feature in the HTRU 2 pulsar dataset
            for j in 1:8
                push!(circuit, layer(1, :embedding))
            end
        end

    end

    return circuit
end

#takes a list of class probabilities for the data points, and returns the list of class classifications using argmax
function classify_from_probs(class_probs)
    classification = []
    for i in class_probs
        prob_0 = i[1]
        prob_1 = i[2]

        #assign a class label corresponding with the measurement that has the highest probability
        if (prob_0 < prob_1) push!(classification, 1) else push!(classification, 0) end
    end
    return classification
end

# get the probabilities of each datapoint being positive
function get_score(dataset, config, depth, params)
    # create the quantum circuit
    var_circ = get_var_circuit(config; depth=depth)
    nparams = nparameters(var_circ)
    initial_state = zero_state(1)

    # classify every point in the dataset by getting the class probabilities
    # and using those to determine which class an instance belongs to.
    probs = []
    for i in dataset
        # create the circuit parameter set, including the feature encoding
        circ_params = get_circuit_params(i, params, nparams)
        # bind the parameters to the circuit
        dispatch!(var_circ, circ_params)

        # get the class probabilities for the data point
        probabilities = circuit_predict(var_circ, copy(initial_state))
        push!(probs, probabilities)
    end

    # return the classifications
    score = []
    for i in probs
        push!(score, i[2])
    end
    return score
end

#applies the given circuit to the register, and returns the probabilities of measurement
function circuit_predict(circuit, register)
    return probs(register |> circuit)
end

function get_circuit_params(features, weights, n_circ_params)
    params = []
    f_count, w_count = 1, 1

    for i in 1:n_circ_params
        #insert feature values for each embedding gate, which is every fourth parameterised gate
        if i%4 == 0
            
            #if the circuit depth is greater than one then the features must be embedded again, so reset the counter
            if (f_count == 8) f_count = 1 end
            push!(params, features[f_count])
            f_count += 1
        
        else
            push!(params, copy(weights[w_count]))
            w_count += 1
        end
    end
    return params
end

# classify the given dataset using the the specified circuit configuration
function classify(dataset, config, depth, params)

    # create the quantum circuit
    var_circ = get_var_circuit(config; depth=depth)
    nparams = nparameters(var_circ)
    initial_state = zero_state(1)

    # classify every point in the dataset by getting the class probabilities
    # and using those to determine which class an instance belongs to.
    probs = []
    for i in dataset
        # create the circuit parameter set, including the feature encoding
        circ_params = get_circuit_params(i, params, nparams)
        # bind the parameters to the circuit
        dispatch!(var_circ, circ_params)

        # get the class probabilities for the data point
        probabilities = circuit_predict(var_circ, copy(initial_state))
        push!(probs, probabilities)
    end

    # return the classifications
    return classify_from_probs(probs)
end

#returns the average cross entropy loss over the entire dataset given as a parameter
function avg_loss(dataset, config, depth, params)
    loss = 0

    #create parameter set for every data point, bind parameters to circuit, and apply circuit to register of initial state.
    #use the probabilities of measurement outcomes to calculate cross-entropy

    losses::Vector{Float64} = [0.0 for i in 1:length(dataset)]

    # thread this 
    for (index, data_point) in collect(enumerate(dataset))
        
        # build the quantum circuit
        var_circ = get_var_circuit(config; depth=depth)
        nparams = nparameters(var_circ)
        initial_state = zero_state(1)
        
        # bind the parameters to the circuit
        circ_params = get_circuit_params(data_point, params, nparams)
        dispatch!(var_circ, circ_params)

        probabilities = circuit_predict(var_circ, copy(initial_state))
        #use probability associated with datapoint's actual class
        losses[index] = -1*log(probabilities[Int(data_point[9])+1])
    end

    loss = sum(losses)

    loss = loss/length(dataset)
    return loss
end

function train_original_learner(train_X, valid_X, config, nfevs, depth)

    # create an initial guess for the parameters of the circuit's trainable blocks.
    n_params = 24 * depth + 3
    init_params = Random.rand(Float64, n_params)

    # these hold the metrics of interest over the training of the classifier.
    train_v_losses = []
    train_t_losses = []
    train_params::Vector{Vector{Float64}} = []

    epoch_counter = 1

    # the objective function that is used by the optimizer.
    function objective(x)

        # evaluate the loss on the training and validation datasets
        loss = avg_loss(train_X, config, depth, x)

        v_loss = avg_loss(valid_X, config, depth, x)

        # store the results 
        push!(train_v_losses, copy(v_loss))
        push!(train_t_losses, copy(loss))
        push!(train_params, copy(x))

        epoch_counter += 1

        # return the loss on the training dataset for the optimizer to use.
        return loss
    end
    
    # create the optimizer
    opt = Opt(:LN_COBYLA, length(init_params))

    # specify the optimizer's objective function
    opt.min_objective = (x, grad) -> objective(x)

    # the maximum number of function evaluations that the optimizer is allowed to make.
    opt.maxeval = nfevs
    # optimize the classifier
    optimize(opt, init_params)

    return train_v_losses, train_t_losses, train_params
end