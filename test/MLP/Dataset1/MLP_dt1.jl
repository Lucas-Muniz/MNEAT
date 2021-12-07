using Dates
const SRC_PATH = "../../../src"

include(joinpath(SRC_PATH, "MLPClassifier/MLPClassifier.jl"))
include(joinpath(SRC_PATH, "datasets/dataset_reader.jl"))
include(joinpath(SRC_PATH, "datasets/Drug-Classification/dataset1.jl"))
include(joinpath(SRC_PATH, "FileInterface/fileInterface.jl"))
include("../MLP_test_functions.jl")
include(joinpath(SRC_PATH, "MutualInformation/information-plane.jl"))
#include(joinpath(SRC_PATH, "MutualInformation/information-plane.jl"))

function show_results(dt, net)
    println("** Test set")
    println("Accuracy: ", accuracy(zip(dt.input_test, dt.output_test), net))
    println("Error rate: ", error_rate(zip(dt.input_test, dt.output_test), net))
    println("** Training set")
    println("Accuracy: ", accuracy(zip(dt.input_train, dt.output_train), net))
    println("Error rate: ", error_rate(zip(dt.input_train, dt.output_train), net))
end

t0 = time()
#topology = [15,5,10]
topology = [5]
#topology = [10,5]
#topology = [5,5]
Random.seed!(1234)
net1 = MLPClassifier(dataset1_nfeatures, dataset1_labels, topology)
#Random.seed!(1234)
#net2 = MLPClassifier(dataset1_nfeatures, dataset1_labels, topology)
net2 = deepcopy(net1)
#net = MLPClassifier(dataset1_nfeatures, dataset1_labels, [5])
path = joinpath(SRC_PATH, "datasets/Drug-Classification")
dt = get_dataset1_setup(path, normalise=true, separate=true)
dt_b = get_dataset1_setup(path, normalise=true, separate=true, balance=true)

#epochs = 10000
epochs = 5000

println("Topology: ", topology)
println("Epochs: ", epochs)

#Random.seed!(1234)
println("-> Not balanced")
println()
println("--> Results before training")
show_results(dt, net1)
fit!(net1, dt.input_train, dt.output_train; max_epochs = epochs)
println()
println("--> Results after training")
show_results(dt, net1)
println()
t1 = time()
#Random.seed!(1234)
println("-> Balanced")
println()
println("--> Results before training")
show_results(dt_b, net2)
fit!(net2, dt_b.input_train, dt_b.output_train; max_epochs = epochs)
println()
println("--> Results after training")
show_results(dt_b, net2)

println()

t2 = time()
println("Training time (not balanced):", t1 - t0," seconds.")
println("Training time (balanced):", t2 - t1," seconds.")

current_date = Dates.now()
time_str = Dates.format(current_date, "HH:MM:SS")
date_str = Dates.format(current_date, "dd-mm-yyyy")

#filename = "MLP-nets-dt1.jld2"
filename = string("MLP-nets-dt1-", date_str, "-", time_str, ".jld2")
addToFile(filename, "Net-notbalanced", net1)
addToFile(filename, "Net-balanced", net2)


#=
data = zip(dt.input_train, dt.output_train)
data_test = zip(dt.input_test, dt.output_test)
epochs = 10000
number_nets = 20
step_len = 1
nbins = 5
plot_IP_average(dt.n_features, dt.labels, topology, number_nets, data, epochs, step_len, "IP-dt1-MLP.png")
plot_MLP_error_rate(dt.n_features, dt.labels, topology, number_nets, data, data_test, epochs, step_len, "MLP-error_rate.png")
points_IP = calculate_IP_average_points(dt.n_features, dt.labels, topology, number_nets, data, epochs, step_len, nbins)
generate_plot_IP_average(points_IP, number_nets, epochs, step_len, "IP-dt1-MLP-$(nbins)bins.png")
=#
# Criar uma função para salvar as redes em um arquivo
# file = load("Resultados_30:09:2021/MLP-dt1-IP-points.jld2")
