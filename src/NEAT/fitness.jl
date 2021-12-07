#include("../datasets/dataset_reader.jl")
#include("../datasets/Demo/datasetDemo.jl")
#include("../datasets/Demo-overlap/datasetDemoOverlap.jl")
#include("../datasets/Drug-Classification/dataset1.jl")
#include("../datasets/Income-Classification/dataset2.jl")
#include("../datasets/Wine-Classification/dataset3.jl")
#include("../datasets/Drug-Consumism-Classification/dataset4.jl")
#include("../datasets/Heart-Disease-Classification/dataset5.jl")

const MAX_DATASET_ID = 7
const DEFAULT_DATASET_PATH = "../datasets"

function f1!(ch::Chromosome, dt_setup::DatasetSetup)::Float64
    ch.fitness = 1.0
end

function f2!(ch::Chromosome, dt_setup::DatasetSetup)::Float64
    mlp = convert_to_FluxNet(ch, dt_setup.labels)
    error = error_rate(zip(dt_setup.input_train, dt_setup.output_train), mlp)
    ch.fitness = (2.0/(error+1.0)) - 1.0
end

fitness_functions = Function[f1!, f2!]
dataset_functions = Function[get_dataset0_setup, get_dataset0_overlap_setup, get_dataset1_setup,
                             get_dataset2_setup, get_dataset3_setup, get_dataset4_setup,
                             get_dataset5_setup]
dataset_paths = String["Demo", "Demo-overlap", "Drug-Classification", "Income-Classification",
                       "Wine-Classification", "Drug-Consumism-Classification",
                       "Heart-Disease-Classification"]


function get_dataset(dataset_id::Int=0; normalise::Bool=false, balance::Bool=false)
    @assert length(dataset_functions) == MAX_DATASET_ID
    @assert length(dataset_paths) == MAX_DATASET_ID
    if dataset_id > MAX_DATASET_ID || dataset_id <= 0
        get_setup = dataset_functions[1]
        path = joinpath(DEFAULT_DATASET_PATH, dataset_paths[1])
    else
        get_setup = dataset_functions[dataset_id+1]
        path = joinpath(DEFAULT_DATASET_PATH, dataset_paths[dataset_id+1])
    end
    setup = get_setup(path, normalise=normalise, balance=balance, separate=true)
    return setup
end


struct Evaluation <: Function
    dataset_setup::DatasetSetup
    fitness_function::Function
    function Evaluation(dataset_id::Int=0, function_id::Int=1; normalise::Bool=false,
                        balance::Bool=false)
        dataset_setup = get_dataset(dataset_id, normalise=normalise, balance=balance)
        if function_id < 1 || function_id > length(fitness_functions)
            fitness = fitness_functions[1]
        else
            fitness = fitness_functions[function_id]
        end
        new(dataset_setup, fitness)
    end
end

(e::Evaluation)(ch::Chromosome) = e.fitness_function(ch, e.dataset_setup)
