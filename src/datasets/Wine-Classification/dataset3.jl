#include("../dataset_reader.jl")
const dataset3_labels = Float64[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
const dataset3_nfeatures = 11
const dataset3_filename1 = "winequality-red.csv"
const dataset3_filename2 = "winequality-white.csv"

function get_numerical_dataset3(path::String="")
    data_red = get_data(joinpath(path, dataset3_filename1); dlm=";")
    data_white = get_data(joinpath(path, dataset3_filename2); dlm=";")

    data = vcat(data_red, data_white)

    conversion_pattern = [Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64]
    convert_values(data, conversion_pattern)
end

function get_dataset3(path::String=""; shuffle::Bool=false, test_perc::Float64=0.2,
                      normalise::Bool=false, separate_test_data::Bool=true, balance_classes::Bool=false)
    data = get_numerical_dataset3(path)
    separate_dataset(data, test_percentage=test_perc, shuffle=shuffle, normalise=normalise,
                     separate_test_data=separate_test_data, balance_classes=balance_classes)
end

function get_dataset3_setup(path::String=""; normalise::Bool=false, separate::Bool=false,
                            balance::Bool=false)
    dataset = get_dataset3(path, normalise=normalise, separate_test_data=separate, balance_classes=balance)
    if separate
        SeparatedDataset(dataset..., dataset3_labels, dataset3_nfeatures, 4)
    else
        Dataset(dataset..., dataset3_labels, dataset3_nfeatures, 4)
    end
end
