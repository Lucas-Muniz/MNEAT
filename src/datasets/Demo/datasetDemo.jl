#include("../dataset_reader.jl")
const datasetDemo_labels = Float64[-1, 1]
const datasetDemo_nfeatures = 3
const datasetDemo_filename = "dados1.txt"

function get_numerical_datasetDemo(path::String="")
    data = get_data(joinpath(path, datasetDemo_filename), dlm=" ", delete_first_line=false)

    conversion_pattern = [Float64, Float64, Float64, Float64]
    convert_values(data, conversion_pattern)
end

function get_datasetDemo(path::String=""; shuffle::Bool=false, test_perc::Float64=0.2,
                         normalise::Bool=false, separate_test_data::Bool=true, balance_classes::Bool=false)
    data = get_numerical_datasetDemo(path)
    separate_dataset(data, test_percentage=test_perc, shuffle=shuffle, normalise=normalise,
                     separate_test_data=separate_test_data, balance_classes=balance_classes)
end

function get_dataset0_setup(path::String=""; normalise::Bool=false, separate::Bool=false,
                            balance::Bool=false)
    dataset = get_datasetDemo(path, normalise=normalise, separate_test_data=separate, balance_classes=balance)
    if separate
        SeparatedDataset(dataset..., datasetDemo_labels, datasetDemo_nfeatures, 0)
    else
        Dataset(dataset..., datasetDemo_labels, datasetDemo_nfeatures, 0)
    end
end
