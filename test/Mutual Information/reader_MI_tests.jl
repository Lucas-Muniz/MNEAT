include("../../src/datasets/dataset_reader.jl")

const dataset_test1_filename = "amostras.txt"
const dataset_test2_filename = "amostras_2T.txt"

function get_numerical_dataset_test1(path::String="")
    data = get_data(joinpath(path, "Teste_IM", dataset_test1_filename), dlm=" ", delete_first_line=false)

    conversion_pattern = [Float64, Float64]
    data = convert_values(data, conversion_pattern)
    separate_input_output(data,  output_columns=1)
end

function get_numerical_dataset_test2(path::String="")
    data = get_data(joinpath(path, "Teste_IM", dataset_test2_filename), dlm=" ", delete_first_line=false)

    conversion_pattern = [Float64, Float64, Float64]
    data = convert_values(data, conversion_pattern)
    separate_input_output(data,  output_columns=2)
end
