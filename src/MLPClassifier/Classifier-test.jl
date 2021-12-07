include("MLPClassifier.jl")
include("../datasets/Drug-Classification/dataset1.jl")
include("../datasets/Income-Classification/dataset2.jl")
include("../datasets/Wine-Classification/dataset3.jl")
include("../datasets/Drug-Consumism-Classification/dataset4.jl")
include("../datasets/Heart-Disease-Classification/dataset5.jl")


net = MLPClassifier(dataset1_nfeatures, dataset1_labels, [15,5,10])
#net = MLPClassifier(dataset1_nfeatures, dataset1_labels, [5])
x_train, y_train, x_test, y_test = get_dataset1("../datasets/Drug-Classification")
println("Accuracy before training: ", accuracy(zip(x_test, y_test), net))
println("Error rate before training: ", error_rate(zip(x_test, y_test), net))
#fit!(net, x_train, y_train; max_epochs = 1000)
fit!(net, x_train, y_train; max_epochs = 10000)
println("Accuracy after training: ", accuracy(zip(x_test, y_test), net))
println("Error rate after training: ", error_rate(zip(x_test, y_test), net))
#=
-> 5 classes
Accuracy before training: 0.05
Error rate before training: 0.95

Accuracy after training: 0.5
Error rate after training: 0.5
=#


#=
net = MLPClassifier(dataset2_nfeatures, dataset2_labels, [26,13,6])
x_train, y_train, x_test, y_test = get_dataset2("../datasets/Income-Classification")
println("Accuracy before training: ", accuracy(zip(x_test, y_test), net))
println("Error rate before training: ", error_rate(zip(x_test, y_test), net))
#fit!(net, x_train, y_train; max_epochs = 5000)
fit!(net, x_train, y_train; max_epochs = 10000)
println("Accuracy after training: ", accuracy(zip(x_test, y_test), net))
println("Error rate after training: ", error_rate(zip(x_test, y_test), net))
=#
#=
-> 2 classes
Accuracy before training: 0.24888684170121295
Error rate before training: 0.751113158298787

Accuracy after training: 0.7543374788883771
Error rate after training: 0.2456625211116229
=#

#=
net = MLPClassifier(dataset3_nfeatures, dataset3_labels, [22,10,15,11])
x_train, y_train, x_test, y_test = get_dataset3("../datasets/Wine-Classification")
println("Accuracy before training: ", accuracy(zip(x_test, y_test), net))
println("Error rate before training: ", error_rate(zip(x_test, y_test), net))
#fit!(net, x_train, y_train; max_epochs = 15000)
fit!(net, x_train, y_train; max_epochs = 10000)
println("Accuracy after training: ", accuracy(zip(x_test, y_test), net))
println("Error rate after training: ", error_rate(zip(x_test, y_test), net))
=#
#=
-> 11 classes
Accuracy before training: 0.007692307692307693
Error rate before training: 0.9923076923076923

Accuracy after training: 0.5030769230769231
Error rate after training: 0.4969230769230769
=#

#=
#net = MLPClassifier(dataset4_nfeatures, dataset4_labels, [24,6,12,7])
net = MLPClassifier(dataset4_nfeatures, dataset4_labels, [24,6,12,4,7])
x_train, y_train, x_test, y_test = get_dataset4("../datasets/Drug-Consumism-Classification")
println("Accuracy before training: ", accuracy(zip(x_test, y_test), net))
println("Error rate before training: ", error_rate(zip(x_test, y_test), net))
#fit!(net, x_train, y_train; max_epochs = 20000)
fit!(net, x_train, y_train; max_epochs = 10000)
println("Accuracy after training: ", accuracy(zip(x_test, y_test), net))
println("Error rate after training: ", error_rate(zip(x_test, y_test), net))
=#
#=
-> 7 classes
Accuracy before training: 0.17506631299734748
Error rate before training: 0.8249336870026526

Accuracy after training: 0.3952254641909814
Error rate after training: 0.6047745358090185
=#

#=
net = MLPClassifier(dataset5_nfeatures, dataset5_labels, [26, 13, 7, 5])
x_train, y_train, x_test, y_test = get_dataset5("../datasets/Heart-Disease-Classification")
println("Accuracy before training: ", accuracy(zip(x_test, y_test), net))
println("Error rate before training: ", error_rate(zip(x_test, y_test), net))
fit!(net, x_train, y_train; max_epochs = 10000)
println("Accuracy after training: ", accuracy(zip(x_test, y_test), net))
println("Error rate after training: ", error_rate(zip(x_test, y_test), net))
=#
#=
-> 5 classes
Accuracy before training: 0.03333333333333333
Error rate before training: 0.9666666666666667

Accuracy after training: 0.48333333333333334
Error rate after training: 0.5166666666666667
=#
