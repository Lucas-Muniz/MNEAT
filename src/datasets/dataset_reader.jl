using Random
using Flux: normalise
using StatsBase: countmap

function get_data(filename::String; split=true, dlm=",", delete_first_line=true)
    file = open(filename, "r")
    number_lines = countlines(file)
    close(file)
    lines = Vector{String}(undef, number_lines)
    i = 1
    for l in eachline(filename)
        lines[i] = l
        i += 1
    end
    if delete_first_line lines = lines[2:end] end
    if split lines = split_columns(lines; dlm=dlm) end
    lines
end

function split_columns(data::Vector{String}; dlm=",")::Vector{Vector{String}}
    data_length = length(data)
    columns_data = Vector{Vector{String}}(undef, data_length)
    for line = 1:data_length
        columns_data[line] = dlm == " " ? split(data[line]) : split(data[line], dlm)
    end
    columns_data
end

function separate_data(data::Vector{T} where T <: Any, test_percentage::Float64; shuffle=false)
    if shuffle shuffle!(data) end
    bound = Int(floor((1-test_percentage)*length(data)))
    train_set = data[1:bound]
    test_set = data[(bound+1):end]
    train_set, test_set
end


function convert_values(data::Vector{Vector{T}}, column::Int, conversion_table::Dict{String, <: Number}) where T <: Any
    data_length = length(data)
    if data_length >= 1 && length(data[1]) < column
        error("Column $column is out of data column bound.")
    end
    converted_data = Vector{Vector{Number}}(undef, data_length)
    num_columns  = length(data[1])
    for l = 1:data_length
        conv_dt = Vector{Number}(undef, num_columns)
        for c = 1:num_columns
            conv_dt[c] = c == column ? conversion_table[data[l][c]] : data[l][c]
        end
        converted_data[l] = conv_dt
    end
    converted_data
end

function convert_values(data::Vector{Vector{T}}, column::Int, conversion::Type{<: Number}) where T <: Any
    data_length = length(data)
    if data_length >= 1 && length(data[1]) < column
        error("Column $column is out of data column bound.")
    end
    converted_data = Vector{Vector{Any}}(undef, data_length)
    num_columns  = length(data[1])
    for l = 1:data_length
        conv_dt = Vector{Any}(undef, num_columns)
        for c = 1:num_columns
            conv_dt[c] = c == column ? parse(conversion, data[l][c]) : data[l][c]
        end
        converted_data[l] = conv_dt
    end
    converted_data
end

# Vector{Union{Type{<: Number}, Dict{String, <: Number}}}
function convert_values(data::Vector{Vector{String}}, conversion_vector::Vector{S} where S <: Any)
    data_length = length(data)
    converted_data = Vector{Vector{Float64}}(undef, data_length)
    num_columns  = length(data[1])
    if num_columns != length(conversion_vector)
        error("There is no conversion function to all columns.")
    end
    for l = 1:data_length
        conv_dt = Vector{Any}(undef, num_columns)
        for c = 1:num_columns
            conv_dt[c] = conversion(data[l][c], conversion_vector[c])
        end
        converted_data[l] = conv_dt
    end
    converted_data
end

conversion(data::String, table::Dict{String, <: Number}) = table[data]
conversion(data::String, type::Type{T}) where T <: Number = parse(type, data)


function separate_input_output(data::Vector{Vector{T}};output_columns::Int=1, normalised::Bool=false) where T <: Any
    data_length = length(data)
    num_columns = length(data[1])
    if output_columns >= data_length
        error("The quantity of output columns is equal or greater than the number of data columns.")
    end
    separator = num_columns-output_columns+1
    input_data = Vector{Vector{Float64}}(undef, data_length)
    output_data = Vector{Vector{Float64}}(undef, data_length)
    for i = 1:data_length
        input_data[i] = data[i][1:(separator-1)]
        output_data[i] = data[i][separator:end]
    end
    if normalised input_data = normalise.(input_data, dims=1) end
    input_data, output_data
end

#=
function separate_dataset(data::Vector{Vector{T}} where T <: Number; output_columns::Int=1,
                          test_percentage::Float64=0.2, shuffle::Bool=false,
                          normalise::Bool=false, separate_test_data::Bool=true,
                          balance_classes::Bool=false)
    #if normalise normalise_inputs!(data, output_columns) end
    if separate_test_data
        train_data, test_data = separate_data(data, test_percentage; shuffle=shuffle)
        if balance_classes balance_classes!(train_data) end
        input_train_data, output_train_data = separate_input_output(train_data, normalised=normalise)
        input_test_data, output_test_data = separate_input_output(test_data, normalised=normalise)
        return input_train_data, output_train_data, input_test_data, output_test_data
    else
        if balance_classes balance_classes!(data) end
        input_data, output_data = separate_input_output(data, normalised=normalise)
    end
end
=#


function separate_dataset(data::Vector{Vector{T}} where T <: Number; output_columns::Int=1,
                          test_percentage::Float64=0.2, shuffle::Bool=false,
                          normalise::Bool=false, separate_test_data::Bool=true,
                          balance_classes::Bool=false)
    if normalise normalise_inputs!(data, output_columns) end
    if separate_test_data
        train_data, test_data = separate_data(data, test_percentage; shuffle=shuffle)
        if balance_classes balance_classes!(train_data) end
        input_train_data, output_train_data = separate_input_output(train_data, normalised=false)
        input_test_data, output_test_data = separate_input_output(test_data, normalised=false)
        return input_train_data, output_train_data, input_test_data, output_test_data
    else
        if balance_classes balance_classes!(data) end
        input_data, output_data = separate_input_output(data, normalised=false)
    end
end


#!!!
function normalise_inputs!(data::Vector{Vector{T}} where T <: Number, output_columns::Int=1)
    number_features = length(data[1])
    normalised_features = Vector{Vector{Float64}}(undef, number_features)

    for feature in 1:number_features
        if feature <= (number_features - output_columns)
            normalised_features[feature] = Flux.normalise(getindex.(data, feature))
        else
            normalised_features[feature] = getindex.(data, feature)
        end
    end

    for i in 1:length(data)
        data[i] = getindex.(normalised_features, i)
    end
end

function create_numerical_map(data::Vector{T})::Dict{T, <: Any}  where {T <: Any}
    set_values = sort(collect(Set(data)))
    indexes = 1.0:length(set_values)
    Dict(collect(zip(set_values, indexes)))
end

function balance_classes!(data::Vector{Vector{T}} where T <: Number)
    # The class (output) is the last element in the vector
    outputs = map(v -> v[end], data)
    distribution = countmap(outputs)
    biggest_class = maximum(values(distribution))
    if any(c -> c[2] != biggest_class, distribution)
        for class in keys(distribution)
            if distribution[class] !=  biggest_class
                diff = biggest_class - distribution[class]
                class_inputs = filter(v -> v[end] == class, data)
                class_len = length(class_inputs)
                number_copies = floor(Int, diff/class_len)
                if number_copies != 0
                    append!(data, repeat(class_inputs, number_copies))
                end
                fill = diff - number_copies*class_len
                if fill > 0
                    append!(data, class_inputs[1:fill])
                end
            end
        end
    end
    data
end

abstract type DatasetSetup end

struct Dataset <: DatasetSetup
    input_train::Vector{Vector{Float64}}
    output_train::Vector{Vector{Float64}}
    labels::Vector{Float64}
    n_features::Int
    dataset_id::Int
end

struct SeparatedDataset <: DatasetSetup
    input_train::Vector{Vector{Float64}}
    output_train::Vector{Vector{Float64}}
    input_test::Vector{Vector{Float64}}
    output_test::Vector{Vector{Float64}}
    labels::Vector{Float64}
    n_features::Int
    dataset_id::Int
end

function Base.show(io::IO, dt::DatasetSetup)
    print(io, "Dataset -> id:$(dt.dataset_id), features:$(dt.n_features), labels:$(dt.labels)")
end

include("dataset_plot.jl")
