# Classical-Ensembles-of-Single-Qubit-Quantum-Classifiers
The code in this repository is written in Julia and uses the Yao package to simulate classical bagging and boosting ensembles of single-qubit QAUM classifiers.

The code has been tested with Julia 1.7.3, however it should work with any version from 1.7.2 and newer.

The following Julia packages are required:
- PyCall
- DataFrames
- Queryverse
- Plots
- Yao
- Pycall
- NLopt
- LinearAlgebra
- JLD2
- CSV
- ScikitLearn
- Random
- PlotlyJS
- StatsBase

The following Python libraries are required:
- numpy
- pandas
- scikit-learn

The following files contain the code for performing training and testing of the relevant classifiers:
- BaggingEnsembleTesting.jl
- BoostingEnsembleTesting.jl
- SingleLearnerTesting.jl

The following files contain the functions with the training and classification algorithms for the ensembles:
- BaggingEnsembleMethods.jl
- BoostingEnsembleMethods.jl
- Classifier.jl

The following file contains the functions with the training and classification algorithms for the single-qubit QAUM classifier:
- OriginalClassifier.jl
