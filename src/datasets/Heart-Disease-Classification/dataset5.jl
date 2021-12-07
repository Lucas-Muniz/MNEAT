#include("../dataset_reader.jl")
const dataset5_labels = Float64[0, 1, 2, 3, 4]
const dataset5_nfeatures = 13
const dataset5_filename = "original/processed/processed.cleveland.data"

function remove_null_data(data::Vector{Vector{String}})
    regex_number = r"\S*((\+|-)?([0-9]+(\.[0-9]*)?|\.[0-9]+))\S*"
    new_data = Vector{String}[]
    for line in data
        if all(occursin.(regex_number, line))
            push!(new_data, line)
        end
    end
    new_data
end

function get_numerical_dataset5(path::String="")
    data = get_data(joinpath(path, dataset5_filename); dlm=",", delete_first_line=false)
    data = remove_null_data(data)

    conversion_pattern = [Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64]
    convert_values(data, conversion_pattern)
end

function get_dataset5(path::String=""; shuffle::Bool=false, test_perc::Float64=0.2,
                      normalise::Bool=false, separate_test_data::Bool=true,
                      balance_classes::Bool=false)
    data = get_numerical_dataset5(path)
    separate_dataset(data, test_percentage=test_perc, shuffle=shuffle, normalise=normalise,
                     separate_test_data=separate_test_data, balance_classes=balance_classes)
end

function get_dataset5_setup(path::String=""; normalise::Bool=false, separate::Bool=false,
                            balance::Bool=false)
    dataset = get_dataset5(path, normalise=normalise, separate_test_data=separate, balance_classes=balance)
    if separate
        SeparatedDataset(dataset..., dataset5_labels, dataset5_nfeatures, 6)
    else
        Dataset(dataset..., dataset5_labels, dataset5_nfeatures, 6)
    end
end

#=
Data attributes:
#1 - Numerical (integer)
#2 - Categorical (0, 1)
#3 - Categorical (1, 2, 3, 4)
#4 - Numerical (fractional)
#5 - Numerical (fractional)
#6 - Categorical (0, 1)
#7 - Categorical (0, 1, 2)
#8 - Numerical (fractional)
#9 - Categorical (0, 1)
#10 - Numerical (fractional)
#11 - Categorical (1, 2, 3)
#12 - Categorical (0, 1, 2, 3)
#13 - Categorical (3, 6, 7)
#14 - Categorical (0, 1, 2, 3, 4)
=#
