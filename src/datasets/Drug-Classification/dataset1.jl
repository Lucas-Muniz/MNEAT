#include("../dataset_reader.jl")
const dataset1_labels = Float64[1, 2, 3, 4, 5]
const dataset1_nfeatures = 5
const dataset1_filename = "drug200.csv"

function get_numerical_dataset1(path::String="")
    data = get_data(joinpath(path, dataset1_filename))

    sex_dict = create_numerical_map(getindex.(data, 2))
    bp_dict = create_numerical_map(getindex.(data, 3))
    cholesterol_dict = create_numerical_map(getindex.(data, 4))
    drug_dict = create_numerical_map(getindex.(data, 6))

    conversion_pattern = [Float64, sex_dict, bp_dict, cholesterol_dict, Float64, drug_dict]
    convert_values(data, conversion_pattern)
end

function get_dataset1(path::String=""; shuffle::Bool=false, test_perc::Float64=0.2,
                      normalise::Bool=false, separate_test_data::Bool=true, balance_classes::Bool=false)
    data = get_numerical_dataset1(path)
    separate_dataset(data, test_percentage=test_perc, shuffle=shuffle, normalise=normalise,
                     separate_test_data=separate_test_data, balance_classes=balance_classes)
end

function get_dataset1_setup(path::String=""; normalise::Bool=false, separate::Bool=false,
                            balance::Bool=false)
    dataset = get_dataset1(path, normalise=normalise, separate_test_data=separate, balance_classes=balance)
    if separate
        SeparatedDataset(dataset..., dataset1_labels, dataset1_nfeatures, 2)
    else
        Dataset(dataset..., dataset1_labels, dataset1_nfeatures, 2)
    end
end
