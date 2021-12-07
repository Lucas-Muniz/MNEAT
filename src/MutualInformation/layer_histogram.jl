function separate_columns(data::Vector{Vector{T}}) where T <: Number
    if length(data) > 0
        columns_number = length(data[1])
        columns = Vector{Vector{Float64}}(undef, columns_number)
        for i = 1:columns_number
            columns[i] = getindex.(data, i)
        end
        return columns
    else
        error("Null data vector.")
    end
end

function separate_columns(data::Vector{Vector{T}}) where T <: Any
    if length(data) > 0
        columns_number = length(data[1])
        type = typeof(data[1])
        columns = Vector{type}(undef, columns_number)
        for i = 1:columns_number
            columns[i] = getindex.(data, i)
        end
        return columns
    else
        error("Null data vector.")
    end
end

function correct_upper_bounds!(vector, bounds)
    if length(vector) == length(bounds)
        for i = 1:length(vector)
            if vector[i] > bounds[i]
                vector[i] = bounds[i]
            end
        end
    else
        error("Bounds were not set to all elements.")
    end
end

function correct_upper_bounds(vector, bounds)
    if length(vector) == length(bounds)
        new_vector = typeof(vector)(undef, length(vector))
        for i = 1:length(vector)
            if vector[i] > bounds[i]
                new_vector[i] = bounds[i]
            else
                new_vector[i] = vector[i]
            end
        end
        return new_vector
    else
        error("Bounds were not set to all elements.")
    end
end

function get_bin_from_indices(vector, bounds)
    if length(vector) == length(bounds)
        index = 0
        len = length(vector)
        for i = 1:(length(vector)-1)
            index += (vector[i]-1)*prod(bounds[(i+1):end])
        end
        if len >= 1 index += vector[end] end
        return index
    else
        error("Vector and bound vector have different lengths.")
    end
end

get_weights_length(h::Histogram) = length(h.weights)

function flat_vector(vector; nbins = 5, lower_lim=-1, upper_lim=1)
    step = (upper_lim-lower_lim)/nbins
    columns = separate_columns(vector)
    hists = fit.(Histogram, columns, [lower_lim:step:upper_lim])
    bounds = (get_weights_length.(hists)...,)
    indices = [getindex.(get_bin.(hists, v),1) for v in vector]
    indices = [correct_upper_bounds(ind, bounds) for ind in indices]
    get_bin_from_indices.(indices, [bounds])
end

#=
function get_layer_bins_vector(vector; nbins = 5, lower_lim=-1, upper_lim=1)
    columns = separate_columns(vector)
    hists = create_bin_histogram.(columns, [nbins], [lower_lim], [upper_lim])
    bounds = (get_weights_length.(hists)...,)
    indices = [getindex.(get_bin.(hists, v),1) for v in vector]
    indices = [correct_upper_bounds(ind, bounds) for ind in indices]
    combination_values = get_bin_from_indices.(indices, [bounds])
    max_value = foldr(*, bounds)
    create_bin_vector(combination_values, nbins, 1, max_value)
end
=#

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

#=
function fit_into_histogram(vector, nbins::Int, lower_lim, upper_lim; ϵ = 0.01)
    hist_step = (upper_lim-lower_lim)/nbins
    ϵ = 0.01
    edges = collect(lower_lim:hist_step:upper_lim)
    edges[1] -= ϵ; edges[end] += ϵ
    hist = fit(Histogram, vector, edges)
    [getindex.(get_bin(hist, v),1) for v in vector]
end
=#

function create_bin_histogram(vector::Vector{T}, nbins::Int, lower_lim, upper_lim; ϵ = 0.01) where T <: Number
    hist_step = abs(upper_lim-lower_lim)/nbins
    edges = collect(lower_lim:hist_step:upper_lim)
    edges[1] -= ϵ; edges[end] += ϵ
    hist = fit(Histogram, vector, edges)
end

function create_bin_vector(vector::Vector{T}, nbins::Int, lower_lim, upper_lim; ϵ = 0.01) where T <: Number
    hist = create_bin_histogram(vector, nbins, lower_lim, upper_lim, ϵ = ϵ)
    [getindex.(get_bin(hist, v),1) for v in vector]
end

function get_input_bin_vector(vector, nfeatures; nbins = 5, return_bins::Int=0)
    hists = Vector{Vector{Int}}(undef, nfeatures)
    columns = separate_columns(vector)
    for feature in 1:nfeatures
        feature_values = columns[feature]
        min = minimum(feature_values)
        max = maximum(feature_values)
        hists[feature] = create_bin_vector(feature_values, nbins, min, max)
    end
    bounds = repeat([nbins], nfeatures)
    indices = separate_columns(hists)
    combination_values = get_bin_from_indices.(indices, [bounds])
    if return_bins > 0
        max_value = foldr(*, bounds)
        return create_bin_vector(combination_values, return_bins, 1, max_value)
    else
        return floor.(Int, combination_values)
    end
end
