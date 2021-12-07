#include("../dataset_reader.jl")
const dataset4_labels = Float64[1, 2, 3, 4, 5, 6, 7]
const dataset4_nfeatures = 12
const dataset4_filename = "drug_consumption.data"

function get_numerical_dataset4(path::String="")
    data = get_data(joinpath(path, dataset4_filename); delete_first_line=false)

    class_dict = create_numerical_map(getindex.(data, 14))

    columns_number = 13
    for line = 1:length(data) data[line] = data[line][2:columns_number+1] end

    columns_number = length(data[1])
    conversion_pattern = Vector{Union{DataType, Dict{String, <: Number}}}(undef, columns_number)
    for i = 1:12 conversion_pattern[i] = Float64 end
    for i = 13:columns_number conversion_pattern[i] = class_dict end

    convert_values(data, conversion_pattern)
end

function get_dataset4(path::String=""; shuffle::Bool=false, test_perc::Float64=0.2,
                      normalise::Bool=false, separate_test_data::Bool=true, balance_classes::Bool=false)
    data = get_numerical_dataset4(path)
    separate_dataset(data, test_percentage=test_perc, shuffle=shuffle, normalise=normalise,
                     separate_test_data=separate_test_data, balance_classes=balance_classes)
end

function get_dataset4_setup(path::String=""; normalise::Bool=false, separate::Bool=false,
                            balance::Bool=false)
    dataset = get_dataset4(path, normalise=normalise, separate_test_data=separate, balance_classes=balance)
    if separate
        SeparatedDataset(dataset..., dataset4_labels, dataset4_nfeatures, 3)
    else
        Dataset(dataset..., dataset4_labels, dataset4_nfeatures, 3)
    end
end
