using PlotlyJS, CSV, DataFrames

function create_bar_plot_of_5x2_cv(accuracies::Dict{})
    
    accs = Dict("1"=>[[],[]],"2"=>[[],[]],"3"=>[[],[]],"4"=>[[],[]],"5"=>[[],[]])

    for (key, value) in accuracies
        if 'a' in key
            push!(accs[chop(key)][1], value[1])
            push!(accs[chop(key)][2], value[2])
        end
    end

    for (key, value) in accuracies
        if 'b' in key
            push!(accs[chop(key)][2], value[1])
            push!(accs[chop(key)][1], value[2])
        end
    end
    ensemble_acc::Vector{Float64} = []
    og_acc::Vector{Float64} = []
    for (key, value) in accs
        push!(ensemble_acc, value[1][1])
        push!(og_acc, value[1][2])
        push!(ensemble_acc, value[2][1])
        push!(og_acc, value[2][2])

    end
    labels = ["Sample 1","Sample 2","Sample 3","Sample 4","Sample 5","Sample 6","Sample 7","Sample 8","Sample 9","Sample 10",]
    df = DataFrame(A=labels, B=ensemble_acc, C=og_acc)
    rename!(df,:B=>:"Ensemble Accuracy")
    rename!(df,:C=>:"Single QAUM Accuracy")
    println(df)
    labels = 
    trace1 = PlotlyJS.bar(x=df[!,:A],
                  y=df[!,:"Ensemble Accuracy"],
                  name="Bagging Ensemble",text=[string(round(i,digits=1)) for i in df[!,:"Ensemble Accuracy"]],
                  marker_color="FF5800")
    trace2 = PlotlyJS.bar(x=df[!,:A],
                 y=df[!,:"Single QAUM Accuracy"],
                 name="Single QAUM", text=[string(round(i, digits=1)) for i in df[!, :"Single QAUM Accuracy"]],
                 marker_color="8A2BE2")
    data = [trace1, trace2]

    layout = Layout(;barmode="group",title="Validation Accuracy for HTRU 2 Pulsar Dataset",yaxis_title="Validation Accuracy", 
                    xaxis_title="Training and Validation Sample")
    PlotlyJS.plot(data, layout)

end

function create_bar_graph_avg_acc(accuracies, graph_label, x_axis_label, x_values, ensemble_label)

    #ensemble_accs = [round(sum(accuracies[1][1])/10, digits=1), round(sum(accuracies[2][1])/10,digits=1), round(sum(accuracies[3][1])/10,digits=1)]
    #qaum_accs = [round(sum(accuracies[1][2])/10, digits=1), round(sum(accuracies[2][2])/10,digits=1), round(sum(accuracies[3][2])/10,digits=1)]

    ensemble_accs = accuracies[1]
    qaum_accs = accuracies[2]

    datasets = x_values
    df = DataFrame(A=datasets, B=ensemble_accs, C=qaum_accs)
    
    if ensemble_label == "boosting"
        trace1 = PlotlyJS.bar(x=df[!,:A],
                      y=df[!,:B],
                      name="Boosting Ensemble",
                      namefont = "computer modern",
                      text=[string(round(i,digits=1)) for i in df[!,:B]],
                      textpostion="top center",
                      textfont_family="computer modern",
                      textfont_size=15,
                      textfont_color="white",
                      marker_color="FF5800",                    #"689b46", 
                      font_family="computer modern",
                font_color="black",
                title_font_family="computer modern",
                title_font_color="black",
                legend_title_font_color="black")
    
        trace2 = PlotlyJS.bar(x=df[!,:A],
                      y=df[!,:C],
                      name="Single QAUM",
                      text=[string(round(i, digits=1)) for i in df[!, :C]],
                      textpostion="top center",
                      textfont_family="computer modern",
                      textfont_size=15,
                      textfont_color="white",
                      marker_color="8A2BE2",                    #"c11f3d",
                      font_family="computer modern",
            font_color="black",
            title_font_family="computer modern",
            title_font_color="black",
            legend_title_font_color="black")

    elseif ensemble_label == "bagging"
        trace1 = PlotlyJS.bar(x=df[!,:A],
                      y=df[!,:B],
                      name="Bagging Ensemble",
                      namefont = "computer modern",
                      text=[string(round(i,digits=1)) for i in df[!,:B]],
                      textpostion="top center",
                      textfont_family="computer modern",
                      textfont_size=15,
                      textfont_color="white",
                      marker_color="FF5800",
                      font_family="computer modern",
                      font_color="black",
                      title_font_family="computer modern",
                      title_font_color="black",
                      legend_title_font_color="black")
    
        trace2 = PlotlyJS.bar(x=df[!,:A],
                      y=df[!,:C],
                      name="Single QAUM",
                      text=[string(round(i, digits=1)) for i in df[!, :C]],
                      textpostion="top center",
                      textfont_family="computer modern",
                      textfont_size=15,
                      textfont_color="white",
                      marker_color="8A2BE2",                     
                      font_family="computer modern",
            font_color="black",
            title_font_family="computer modern",
            title_font_color="black",
            legend_title_font_color="black")
        
    end

    data = [trace1, trace2]

    layout = Layout(;barmode="group",
                     font_family="computer modern",
                     font_size=18,
                     font_color="black",
                     x_font_family="computer modern",
                     #title=graph_label,
                     titlefont_family = "computer modern",
                     titlefont_color="black",
                     titlexanchor="right",
                     texteposition="top center",
                     titlefont_size=40,
                     yaxis_title="Validation Accuracy", 
                     yaxis_font_family="black",
                     xaxis_font_family="black",
                     yaxis_title_font_family="computer modern", 
                     yaxis_title_font_size=25,
                     xaxis_title_font_size=25,
                     xaxis_title_font_color="black",
                     yaxis_title_font_color="black",
                     xaxis_title_font_family="computer modern",
                     xaxis_title=x_axis_label,
                     legend_font_family="computer modern",
                     legend_font_size=20,
                     legend_font_color="black")

    PlotlyJS.plot(data, layout)

end

#=
boosting_digits = [87.0, 78.0, 78.0, 75.0, 89.0, 79.0, 77.0, 83.0, 92.0, 83.0]
qaum_v_boost_digits = [77.0, 77.0, 76.0, 69.0, 82.0, 69.0, 80.0, 71.0, 86.0, 83.0]

digits = [boosting_digits, qaum_v_boost_digits]

#create_bar_graph_avg_acc(digits, "Validation Accuracy for MNIST Digits Dataset", "Training and Validation Sample", 
#                         ["Sample 1","Sample 2","Sample 3","Sample 4","Sample 5","Sample 6","Sample 7","Sample 8","Sample 9","Sample 10"],
#                         "boosting")

boosting_cancer = [82.7, 92.0, 82.7, 72.0, 82.7, 62.7, 69.3, 85.3, 92.0, 74.7]
qaum_v_boost_cancer = [82.7, 88.0, 85.3, 62.7, 73.3, 85.3, 64.0, 76.0, 81.3, 66.7]

cancer = [boosting_cancer, qaum_v_boost_cancer]

#create_bar_graph_avg_acc(cancer, "Validation Accuracy for Wisconsin Breast Cancer Dataset", "Training and Validation Sample", 
#                         ["Sample 1","Sample 2","Sample 3","Sample 4","Sample 5","Sample 6","Sample 7","Sample 8","Sample 9","Sample 10"],
#                         "boosting")



boosting_pulsar = [84.0, 86.0, 82.0, 96.0, 78.0, 90.0, 84.0, 90.0, 84.0, 84.0]
qaum_v_boost_pulsar = [86.0, 86.0, 78.0, 88.0, 84.0, 92.0, 88.0, 86.0, 78.0, 72.0]

pulsar = [boosting_pulsar, qaum_v_boost_pulsar]

#create_bar_graph_avg_acc(pulsar, "Validation Accuracy for HTRU 2 Pulsar Dataset", "Training and Validation Sample", 
#                         ["Sample 1","Sample 2","Sample 3","Sample 4","Sample 5","Sample 6","Sample 7","Sample 8","Sample 9","Sample 10"],
#                         "boosting")



#create_bar_graph_avg_acc([digits, cancer, pulsar], "Mean Validation Accuracy across Samples", "Dataset", ["MNIST Digits", "Wisconsin Breast Cancer", "HTRU 2 Pulsar"], "boosting")

bagging_digits = [95.0, 96.0, 96.0, 95.0, 93.0, 96.0, 92.0, 98.0, 90.0, 98.0]
qaum_v_bagging_digits = [88.0, 94.0, 91.0, 93.0, 92.0, 90.0, 94.0, 95.0, 90.0, 91.0]

digits = [bagging_digits, qaum_v_bagging_digits]

#create_bar_graph_avg_acc(digits, "Validation Accuracy for MNIST Digits Dataset", "Training and Validation Sample", 
#                         ["Sample 1","Sample 2","Sample 3","Sample 4","Sample 5","Sample 6","Sample 7","Sample 8","Sample 9","Sample 10"],
#                         "bagging")

bagging_cancer = [97.3, 92.0, 94.7, 94.7, 89.3, 93.3, 93.3, 97.3, 92.0, 96.0]
qaum_v_bagging_cancer = [96.0, 92.0, 93.3, 89.3, 93.3, 97.3, 94.7, 97.3, 88.0, 93.3]

cancer = [bagging_cancer, qaum_v_bagging_cancer]

#create_bar_graph_avg_acc(cancer, "Validation Accuracy for Wisconsin Breast Cancer Dataset", "Training and Validation Sample", 
#                         ["Sample 1","Sample 2","Sample 3","Sample 4","Sample 5","Sample 6","Sample 7","Sample 8","Sample 9","Sample 10"],
#                        "bagging")


bagging_pulsar = [88.0, 100.0, 88.0, 94.0, 92.0, 88.0, 94.0, 88.0, 92.0, 88.0]
qaum_v_bagging_pulsar = [80.0, 98.0, 88.0, 90.0, 94.0, 86.0, 100.0, 86.0, 86.0, 88.0]

pulsar = [bagging_pulsar, qaum_v_bagging_pulsar]

#create_bar_graph_avg_acc(pulsar, "Validation Accuracy for HTRU 2 Pulsar Dataset", "Training and Validation Sample", 
#                         ["Sample 1","Sample 2","Sample 3","Sample 4","Sample 5","Sample 6","Sample 7","Sample 8","Sample 9","Sample 10"],
#                         "bagging")

create_bar_graph_avg_acc([digits, cancer, pulsar], "Mean Validation Accuracy across Samples", "Dataset", ["MNIST Digits", "Wisconsin Breast Cancer", "HTRU 2 Pulsar"], "bagging")
=#