#include("../dataset_reader.jl")
const dataset2_labels = [0, 1]
const dataset2_nfeatures = 13
const dataset2_filename = "income.csv"

function get_numerical_dataset2(path::String="")
    data = get_data(joinpath(path, dataset2_filename))

    #data_percentage = 0.308 # aprox.: 10000
    #data_percentage = 0.153558 # aprox.: 5000
    data_percentage = 0.092134 # aprox.: 3000
    #data_percentage = 1.0
    data = data[1:round(Int, data_percentage*length(data))]


    workclass_dict = create_numerical_map(getindex.(data, 2))
    education_dict = create_numerical_map(getindex.(data, 3))
    marital_dict = create_numerical_map(getindex.(data, 5))
    occupation_dict = create_numerical_map(getindex.(data, 6))
    relationship_dict = create_numerical_map(getindex.(data, 7))
    race_dict = create_numerical_map(getindex.(data, 8))
    sex_dict = create_numerical_map(getindex.(data, 9))
    country_dict = create_numerical_map(getindex.(data, 13))

    conversion_pattern = [Float64, workclass_dict, education_dict, Int, marital_dict, occupation_dict, relationship_dict, race_dict, sex_dict, Int, Int, Int, country_dict, Float64]
    convert_values(data, conversion_pattern)
end

function get_dataset2(path::String=""; shuffle::Bool=false, test_perc::Float64=0.2,
                      normalise::Bool=false, separate_test_data::Bool=true, balance_classes::Bool=false)
    data = get_numerical_dataset2(path)
    separate_dataset(data, test_percentage=test_perc, shuffle=shuffle, normalise=normalise,
                     separate_test_data=separate_test_data, balance_classes=balance_classes)
end

function get_train_dataset2(path::String=""; shuffle::Bool=false, test_perc::Float64=0.2)
    data = get_numerical_dataset2(path)
    train_data, test_data = separate_data(data, test_perc; shuffle=shuffle)
    input_train_data, output_train_data = separate_input_output(train_data)
    input_train_data, output_train_data
end

function get_dataset2_setup(path::String=""; normalise::Bool=false, separate::Bool=false,
                            balance::Bool=false)
    dataset = get_dataset2(path, normalise=normalise, separate_test_data=separate, balance_classes=balance)
    if separate
        SeparatedDataset(dataset..., dataset2_labels, dataset2_nfeatures, 3)
    else
        Dataset(dataset..., dataset2_labels, dataset2_nfeatures, 3)
    end
end



#=
workclass_dict = Dict(
  "Private" => 1,
  "State-gov" => 2,
  "Without-pay" => 3,
  "Self-emp-inc" => 4,
  "Never-worked" => 5,
  "Local-gov" => 6,
  "Self-emp-not-inc" => 7,
  "Federal-gov" => 8,
  "" => 9
  ) # column 2

education_dict = Dict(
  "Preschool" => 1,
  "1st-4th" => 2,
  "5th-6th" => 3,
  "7th-8th" => 4,
  "9th" => 5,
  "10th" => 6,
  "11th" => 7,
  "12th" => 8,
  "Some-college" => 9,
  "Bachelors" => 10,
  "HS-grad" => 11,
  "Assoc-acdm" => 12,
  "Prof-school" => 13,
  "Assoc-voc" => 14,
  "Masters" => 15,
  "Doctorate" => 16
  ) # column 3

marital_dict = Dict(
  "Never-married" => 1,
  "Married-civ-spouse" => 2,
  "Married-AF-spouse" => 3,
  "Married-spouse-absent" => 4,
  "Separated" => 5,
  "Divorced" => 6,
  "Widowed" => 7
  ) # column 5

occupation_dict = Dict("" => 1,
  "Prof-specialty" => 2,
  "Farming-fishing" => 3,
  "Handlers-cleaners" => 4,
  "Transport-moving" => 5,
  "Adm-clerical" => 6,
  "Machine-op-inspct" => 7,
  "Exec-managerial" => 8,
  "Craft-repair" => 9,
  "Priv-house-serv" => 10,
  "Armed-Forces" => 11,
  "Tech-support" => 12,
  "Sales" => 13,
  "Protective-serv" => 14,
  "Other-service" => 15) # column 6

relationship_dict = Dict(
  "Wife" => 1,
  "Husband" => 2,
  "Own-child" => 3,
  "Other-relative" => 4,
  "Not-in-family" => 5,
  "Unmarried" => 6
  ) # column 7

race_dict = Dict( "Black" => 1,
 "Asian-Pac-Islander" => 2,
 "Amer-Indian-Eskimo" => 3,
 "White" => 4,
 "Other" => 5
 ) # column 8

sex_dict = Dict("Female" => 1, "Male" => 2) # column 9
=#
