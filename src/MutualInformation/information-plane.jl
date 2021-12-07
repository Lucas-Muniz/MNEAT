include("../MLPClassifier/MLPClassifier.jl")
include("mutual-information.jl")
include("../datasets/dataset_reader.jl")
include("../datasets/Drug-Classification/dataset1.jl")
include("../datasets/Income-Classification/dataset2.jl")
include("../datasets/Wine-Classification/dataset3.jl")
include("../datasets/Drug-Consumism-Classification/dataset4.jl")
include("../datasets/Heart-Disease-Classification/dataset5.jl")
include("../datasets/Demo/datasetDemo.jl")
include("../datasets/Demo-overlap/datasetDemoOverlap.jl")
using Plots, Statistics, LaTeXStrings
using Flux: onehot

struct LinearizedLayer
    vector::Vector
    nbins::Int
    function LinearizedLayer(vector, nbins::Int)
        new(vector, nbins)
    end
    function LinearizedLayer(vector, nfeatures::Int, neuron_nbins::Int, output_nbins::Int)
        hists = Vector{Vector{Int}}(undef, nfeatures)
        columns = separate_columns(vector)
        for feature in 1:nfeatures
            feature_values = columns[feature]
            min = minimum(feature_values)
            max = maximum(feature_values)
            hists[feature] = create_bin_vector(feature_values, neuron_nbins, min, max)
        end
        bounds = repeat([neuron_nbins], nfeatures)
        indices = separate_columns(hists)
        combination_values = get_bin_from_indices.(indices, [bounds])
        max_value = foldr(*, bounds)
        linearized = create_bin_vector(combination_values, output_nbins, 1, max_value)
        new(linearized, output_nbins)
    end
    function LinearizedLayer(vector, neuron_nbins::Int, output_nbins::Int, lower_lim, upper_lim)
        columns = separate_columns(vector)
        hists = create_bin_histogram.(columns, [neuron_nbins], [lower_lim], [upper_lim])
        bounds = (get_weights_length.(hists)...,)
        indices = [getindex.(get_bin.(hists, v),1) for v in vector]
        indices = [correct_upper_bounds(ind, bounds) for ind in indices]
        combination_values = get_bin_from_indices.(indices, [bounds])
        #max_value = max(prod(bounds), typemax(Int64)) # foldr(*, bounds) == prod(bounds)
        max_value = map(n -> UInt128(n), bounds) |> prod
        max_value = max_value > typemax(Int64) ? typemax(Int64) : max_value
        linearized = create_bin_vector(combination_values, output_nbins, 1, max_value)
        new(linearized, output_nbins)
    end
end

# Check if the number of nodes explodes
# (map(n -> UInt128(n), nbins*ones(Int, num_nodes)) |> prod) > typemax(Int64)


# lims::::Tuple{T,T} where T <: Number
function layer_information_plane_position_MarkovChain_t(model::MLPClassifier, input, layer::Int, lims, X_flatten, Y_flatten, I_XY, nbins::Int=10)
    if length(model.net) >= layer && layer > 0
        subnet = model.net[1:layer]
        local subnet_output
        if length(model.net) == layer
            subnet_output = get_label_position.(predict.([model], input), [model.labels])
            nbins_subnet_output = length(model.labels)
            subnet_output = LinearizedLayer(subnet_output, nbins_subnet_output)
        else
            subnet_output = subnet.(input)
            nbins_subnet_output = nbins*length(subnet_output[1])#=nbins^length(subnet_output[1])=#
            subnet_output = LinearizedLayer(subnet_output, nbins, nbins_subnet_output, lims[1], lims[2])
        end
        I_XT = mutual_information_unit(X_flatten.vector, subnet_output.vector, (0, X_flatten.nbins), (0, subnet_output.nbins))
        I_TY = mutual_information(calculate_markov_chain_probability(Y_flatten.vector, X_flatten.vector, subnet_output.vector, (X_flatten.nbins, Y_flatten.nbins, subnet_output.nbins))...)
        (I_XT, I_TY/I_XY)
    else
        error("Invalid layer number.")
    end
end


get_label_position(value, label_vector) = findfirst(x -> x == value, label_vector)

#=
function calculate_joint_probability(X::Vector{<:Number}, Y::Vector{<:Number}, lim_X, lim_Y)
    min_X, max_X = lim_X
    min_Y, max_Y = lim_Y
    range_X = min_X:max_X
    range_Y = min_Y:max_Y
    p_x = fit(Histogram, (X), range_X, closed=:right)
    p_y = fit(Histogram, (Y), range_Y, closed=:right)
    p_xy = fit(Histogram, (X, Y), (range_X, range_Y), closed=:right)
    p_X = p_x.weights/sum(p_x.weights)
    p_Y = p_y.weights/sum(p_y.weights)
    p_XY = p_xy.weights/sum(p_xy.weights)
    return p_X, p_Y, p_XY
end
=#

function calculate_markov_chain_probability(Y::Vector{<:Number}, X::Vector{<:Number}, T::Vector{<:Number})
    p_t = fit(Histogram, (T))
    p_y = fit(Histogram, (Y))
    p_x = fit(Histogram, (X))
    p_xy = fit(Histogram, (X, Y))
    p_tx = fit(Histogram, (T, X))
    p_ty = zeros(Float64, (length(p_t.weights), length(p_y.weights)))
    for t = 1:length(p_t.weights)
        for y = 1:length(p_y.weights)
            for x = 1:length(p_x.weights)
                p1 = get_bin_probability(p_xy, (x,y)...)
                p2 = get_bin_probability(p_tx, (t,x)...)
                p3 = get_bin_probability(p_x, (x)...)
                if p2 == 0 && p3 == 0
                    p_ty[t,y] += 0
                else
                    p_ty[t,y] += p1*(p2/p3)
                end
            end
        end
    end
    p_T = p_t.weights/sum(p_t.weights)
    p_Y = p_y.weights/sum(p_y.weights)
    return p_T, p_Y, p_ty
end

function calculate_markov_chain_probability(Y::Vector{<:Number}, X::Vector{<:Number}, T::Vector{<:Number}, nbins)
    p_t = fit(Histogram, (T), nbins=nbins[3])
    p_y = fit(Histogram, (Y), nbins=nbins[2])
    p_x = fit(Histogram, (X), nbins=nbins[1])
    p_xy = fit(Histogram, (X, Y), nbins=(nbins[1:2]...,))
    p_tx = fit(Histogram, (T, X), nbins=(nbins[3], nbins[1]))
    p_ty = zeros(Float64, (length(p_t.weights), length(p_y.weights)))
    for t = 1:length(p_t.weights)
        for y = 1:length(p_y.weights)
            for x = 1:length(p_x.weights)
                p1 = get_bin_probability(p_xy, (x,y)...)
                p2 = get_bin_probability(p_tx, (t,x)...)
                p3 = get_bin_probability(p_x, (x)...)
                if p2 == 0 && p3 == 0
                    p_ty[t,y] += 0
                else
                    p_ty[t,y] += p1*(p2/p3)
                end
            end
        end
    end
    p_T = p_t.weights/sum(p_t.weights)
    p_Y = p_y.weights/sum(p_y.weights)
    return p_T, p_Y, p_ty
end

function get_corrected_bin(h::Histogram, val)
    bin = searchsortedfirst.(h.edges, val)
    bounds = size(h.weights)
    for i = 1:length(bin)
        if bin[i] >= bound[i]
            bin[i] = bound[i]
        end
    end
    bin
end


function evaluate_information_plane(model::MLPClassifier, input, output, nbins::Int=10)
    layers_number = length(model.net)
    X_flatten = LinearizedLayer(input, model.nfeatures, nbins, nbins*model.nfeatures#=nbins^model.nfeatures=#)
    Y_flatten = LinearizedLayer(get_label_position.(getindex.(output,1), [model.labels]), length(model.labels))
    I_XY = mutual_information_unit(X_flatten.vector, Y_flatten.vector, (0, X_flatten.nbins), (0, Y_flatten.nbins))
    activation_lims = (-1, 1) # Limits of activation function tanh
    output_lims = (0, 1) # Limits of activation function softmax
    IP_points = layer_information_plane_position_MarkovChain_t.([model], [input], 1:(layers_number-2), [activation_lims], [X_flatten], [Y_flatten], [I_XY], [nbins])
    push!(IP_points, layer_information_plane_position_MarkovChain_t(model, input, layers_number, (minimum(Y_flatten.vector), maximum(Y_flatten.vector)), X_flatten, Y_flatten, I_XY, nbins))
    IP_points
end


function information_bottleneck(β::Float64, model::MLPClassifier, input, output;  nbins::Int=10)
    net_output = flat_vector(model(input), nbins = nbins)
    X_flatten = flat_vector(input, nbins = nbins)
    Y_flatten = flat_vector(output, nbins = nbins)
    I_XZ = mutual_information([X_flatten], [net_output])
    I_ZY = mutual_information([net_output], [Y_flatten])
    I_XZ - β*I_ZY
end

function flat_vector1(vector)
    columns = separate_columns(vector)
    h = fit(Histogram, (columns...,))
    bounds = size(h.weights)
    indexes = [searchsortedfirst.(h.edges, v) for v in vector]
    correct_upper_bounds!.(indexes, [bounds])
    [LinearIndices(h.weights)[ind...] for ind in indexes]
end


#=function flat_vector(vector)
    columns = separate_columns(vector)
    hists = fit.(Histogram, columns)
    bounds = (get_weights_length.(hists)...,)
    indices = [getindex.(get_bin.(hists, v),1) for v in vector]
    indices = [correct_upper_bounds(ind, bounds) for ind in indices]
    #linear_indexes = LinearIndices(bounds)
    get_bin_from_indices.(indices, [bounds])
end=#

include("information_plane_plot.jl")


#=
# Test
model = MLPClassifier(dataset1_nfeatures, dataset1_labels, [15,5,10,5])
x_train, y_train, x_test, y_test = get_dataset1("../datasets/Drug-Classification")
x_train, y_train, x_test, y_test = get_dataset2("../datasets/Income-Classification")
net = MLPClassifier(dataset2_nfeatures, dataset2_labels, [26,13])

plot_IP_average(dataset2_nfeatures, dataset2_labels, [26,13], 3, get_train_dataset2, "../datasets/Income-Classification", 100, 2, "IP-avg-10n-100ep-v2.png")
plot_IP_average(dataset2_nfeatures, dataset2_labels, [26,13], 3, data, 100, 2, "IP-avg-10n-100ep-v2.png")


#fit!(model, x_train, y_train ; max_epochs=20)
plot_IP(model, x_train, y_train)
println("-> Error: ", error_rate(zip(x_test, y_test), model))
fit!(model, x_train, y_train ; max_epochs=1000)
plot_IP(model, x_train, y_train, imagename="IP2.png")
println("-> Error: ", error_rate(zip(x_test, y_test), model))
fit!(model, x_train, y_train ; max_epochs=2000)
plot_IP(model, x_train, y_train, imagename="IP3.png")
println("-> Error: ", error_rate(zip(x_test, y_test), model))
=#

#=
x_train, y_train, x_test, y_test = get_datasetDemo("../datasets/Demo")
net = MLPClassifier(datasetDemo_nfeatures, datasetDemo_labels, [3,2])
fit!(net, x_train, y_train ; max_epochs=10)
println("-> Error: ", error_rate(zip(x_test, y_test), net))
data = zip(x_train, y_train)
plot_IP_average(datasetDemo_nfeatures, datasetDemo_labels, topology, number_nets, data, epochs, step_len, "IP-dtDemo-MLP-fixed.png")
=#

#=
topology = [3,2]
number_nets = 10
epochs = 100
step_len = 10
x_train, y_train, x_test, y_test = get_datasetDemoOverlap("../datasets/Demo-overlap")
data = zip(x_train, y_train)
net = MLPClassifier(datasetOverlap_nfeatures, datasetOverlap_labels, topology)
plot_IP_average(datasetOverlap_nfeatures, datasetOverlap_labels, topology, number_nets, data, epochs, step_len, "IP-dtOverlap-MLP-fixed.png")
plot_IP_average(datasetOverlap_nfeatures, datasetOverlap_labels, topology, number_nets, data, epochs, step_len, "IP-dtOverlap-MLP-fixed2.png")
=#
