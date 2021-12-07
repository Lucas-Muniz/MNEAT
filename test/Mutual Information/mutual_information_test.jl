include("../../src/MutualInformation/mutual-information.jl")
include("reader_MI_tests.jl")

function get_layer_bins_vector(vector; neuron_nbins = 5, nbins = 5, lower_lim=-1, upper_lim=1)
    columns = separate_columns(vector)
    hists = create_bin_histogram.(columns, [neuron_nbins], [lower_lim], [upper_lim])
    bounds = (get_weights_length.(hists)...,)
    indices = [getindex.(get_bin.(hists, v),1) for v in vector]
    indices = [correct_upper_bounds(ind, bounds) for ind in indices]
    combination_values = get_bin_from_indices.(indices, [bounds])
    max_value = foldr(*, bounds)
    create_bin_vector(combination_values, nbins, 1, max_value)
end

function calculate_joint_probability_MI(X, Y, lim_X, lim_Y, nbins)
    number_neurons_X = length(X[1])
    bins_X = nbins^(number_neurons_X)
    if number_neurons_X > 1
        linearized_X = get_layer_bins_vector(X, neuron_nbins=bins_X, nbins=nbins, lower_lim=lim_X[1], upper_lim=lim_X[2])
        limits_X = (minimum(linearized_X), maximum(linearized_X))
    else
        linearized_X = getindex.(X, 1)
        limits_X = lim_X
    end

    number_neurons_Y = length(Y[1])
    bins_Y = nbins^(number_neurons_Y)
    if number_neurons_Y > 1
        linearized_Y = get_layer_bins_vector(Y, neuron_nbins=nbins, nbins=bins_Y, lower_lim=lim_Y[1], upper_lim=lim_Y[2])
        limits_Y = (minimum(linearized_Y), maximum(linearized_Y))
    else
        linearized_Y = getindex.(Y, 1)
        limits_Y = lim_Y
    end

    calculate_joint_probability(linearized_X, linearized_Y, limits_X, limits_Y, (bins_X+1, round(Int, (bins_Y))))
end

function get_layer_probabilities(vector; neuron_nbins = 5, nbins = 5, lower_lim=-1, upper_lim=1)
    columns = separate_columns(vector)
    hists = create_bin_histogram.(columns, [neuron_nbins], [lower_lim], [upper_lim])
    probabilities = map(h -> h.weights/sum(h.weights), hists)
    probability = fit(Histogram, (colunms...), nbins = neuron_nbins)
end

#mutual_information(calculate_joint_probability()...)

# dt2 = get_numerical_dataset_test2()
#p2 = calculate_joint_probability_MI(dt2[1], dt2[2], (-2, 1.5), (-1, 1), 3)
#MI = mutual_information(p2...)

#=
Cálculo da informação mútua (caso 1):
-> Valor teórico calculado: I(X;T) = 0.047056227
-> Valor obtido pela aproximação: I(X;T) = 0.04700065437122643 (3 bins)
-> Valor obtido pela aproximação: I(X;T) = 0.04510842519666598 (2 bins)


Cálculo da informação mútua utilizando os valores teóricos (caso 2):
-> Valor teórico calculado: I(X;T) = 0.1685846799340601
-> Valor calculado pela aproximação: I(X;T) = 0.16843457465541672 (9 bins)
-> Valor calculado pela aproximação: I(X;T) = 0.05633855652784133 (4 bins)
=#
