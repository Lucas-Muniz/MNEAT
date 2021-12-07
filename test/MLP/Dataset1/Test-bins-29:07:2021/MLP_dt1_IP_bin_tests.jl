using Dates
const SRC_PATH = "../../../src"

#include(joinpath(SRC_PATH, "MLPClassifier/MLPClassifier.jl"))
include(joinpath(SRC_PATH, "datasets/dataset_reader.jl"))
include(joinpath(SRC_PATH, "MutualInformation/information-plane.jl"))
#include(joinpath(SRC_PATH, "datasets/Drug-Classification/dataset1.jl"))
include(joinpath(SRC_PATH, "FileInterface/fileInterface.jl"))

#t0 = time()
#topology = [15,5,10]
#topology2 = [10,5]
topology = [5]
Random.seed!(1234)
path = joinpath(SRC_PATH, "datasets/Drug-Classification")
#dt = get_dataset1_setup(path, normalise=true, separate=true)
dt_b = get_dataset1_setup(path, normalise=true, separate=true, balance=true)

net_filename = "MLP-nets-dt1-29-07-2021-06:09:56.jld2"
net_balanced = loadFromFile(net_filename, "Net-balanced")

data = zip(dt_b.input_train, dt_b.output_train)

epochs = 10000
number_nets = 20
step_len = 10
nbins = 5

points_filename = "MLP-nets-dt1-IP-bins-test.jld2"

#=
epochs = 10
number_nets = 2
step_len = 2
nbins = 5
=#

#points_IP = calculate_IP_average_points(dt_b.n_features, dt_b.labels, topology, number_nets, data, epochs, step_len, nbins)
#generate_plot_IP_average(points_IP, number_nets, epochs, step_len, "IP-dt1-MLP-$(nbins)bins.png")

current_date = Dates.now()
time_str = Dates.format(current_date, "HH:MM:SS")
date_str = Dates.format(current_date, "dd-mm-yyyy")

addToFile(points_filename, "Date", date_str)
addToFile(points_filename, "Time", time_str)
addToFile(points_filename, "Topology", topology)
for bins = 2:1:30
    points_IP_test = calculate_IP_average_points(dt_b.n_features, dt_b.labels, topology, number_nets, data, epochs, step_len, bins)
    generate_plot_IP_average(points_IP_test, number_nets, epochs, step_len, "IP-dt1-MLP-$(bins)bins.png")
    result = (bins, points_IP_test)
    addToFile(points_filename, "$(bins)-bins", result)
end
