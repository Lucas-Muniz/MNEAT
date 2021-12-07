#include("../dataset_reader.jl")
const datasetOverlap_labels = Float64[1, 2]
const datasetOverlap_nfeatures = 3
const datasetOverlap_filename = "overlapping_data.txt"

function get_numerical_datasetDemoOverlap(path::String="")
    data = get_data(joinpath(path, datasetOverlap_filename), dlm=" ", delete_first_line=false)

    conversion_pattern = [Float64, Float64, Float64, Float64]
    convert_values(data, conversion_pattern)
end

function get_datasetDemoOverlap(path::String=""; shuffle::Bool=false, test_perc::Float64=0.2,
                                normalise::Bool=false, separate_test_data::Bool=true, balance_classes::Bool=false)
    data = get_numerical_datasetDemoOverlap(path)
    separate_dataset(data, test_percentage=test_perc, shuffle=shuffle, normalise=normalise,
                     separate_test_data=separate_test_data, balance_classes=balance_classes)
end

function get_dataset0_overlap_setup(path::String=""; normalise::Bool=false, separate::Bool=false,
                                    balance::Bool=false)
    dataset = get_datasetDemoOverlap(path, normalise=normalise, separate_test_data=separate, balance_classes=balance)
    if separate
        SeparatedDataset(dataset..., datasetOverlap_labels, datasetOverlap_nfeatures, 1)
    else
        Dataset(dataset..., datasetOverlap_labels, datasetOverlap_nfeatures, 1)
    end
end
