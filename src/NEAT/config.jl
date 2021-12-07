
mutable struct Config

    # phenotype config
    input_nodes::Int64
    output_nodes::Int64
    initial_hidden_nodes::Int64
    max_weight::Float64
    min_weight::Float64
    nn_activation::Symbol
    #weight_stdev::Float64

    # GA config
    pop_size::Int64
    max_fitness_threshold::Float64
    prob_addnode::Float64
    prob_removenode::Float64
    prob_addlayer::Float64
    prob_mutatebias::Float64
    bias_mutation_power::Float64
    prob_mutate_weight::Float64 # dynamic mutation rate (future release)
    weight_mutation_power::Float64
    prob_structural_mutation::Float64
    prob_enable_mutation::Float64
    elitism::Bool

    # genotype compatibility
    compatibility_threshold::Float64
    compatibility_change::Float64
    excess_coeficient::Float64
    disjoint_coeficient::Float64
    weight_coeficient::Float64

    # species
    species_size::Int64
    survival_threshold::Float64 # only the best 20% for each species is allowed to mate
    old_threshold::Int64
    youth_threshold::Int64
    old_penalty::Float64    # always in (0,1)
    youth_boost::Float64    # always in (1,2)
    max_stagnation::Int64
    best_max_stagnation::Int64
    limit_species::Int64

    # evaluation
    dataset_id::Int64
    fitness_function_id::Int64
    normalise_input::Bool
    balance_dataset::Bool

    # checkpoint
    checkpoint_filename::String

    function Config(params::Dict{String,String})

        new(
            # phenotype
            parse(Int64, params["input_nodes"]),
            parse(Int64, params["output_nodes"]),
            parse(Int64, params["initial_hidden_nodes"]),
            parse(Float64, params["max_weight"]),
            parse(Float64, params["min_weight"]),
            Meta.parse(params["nn_activation"]),
            #float(params["weight_stdev"]),

            # GA
            parse(Int64, params["pop_size"]),
            parse(Float64, params["max_fitness_threshold"]),
            parse(Float64, params["prob_addnode"]),
            parse(Float64, params["prob_removenode"]),
            parse(Float64, params["prob_addlayer"]),
            parse(Float64, params["prob_mutatebias"]),
            parse(Float64, params["bias_mutation_power"]),
            parse(Float64, params["prob_mutate_weight"]),
            parse(Float64, params["weight_mutation_power"]),
            parse(Float64, params["prob_structural_mutation"]),
            parse(Float64, params["prob_enable_mutation"]),
            parse(Bool, params["elitism"]),

            # genotype compatibility
            parse(Float64, params["compatibility_threshold"]),
            parse(Float64, params["compatibility_change"]),
            parse(Float64, params["excess_coeficient"]),
            parse(Float64, params["disjoint_coeficient"]),
            parse(Float64, params["weight_coeficient"]),

            # species
            parse(Int64, params["species_size"]),
            parse(Float64, params["survival_threshold"]),
            parse(Int64, params["old_threshold"]),
            parse(Int64, params["youth_threshold"]),
            parse(Float64, params["old_penalty"]),
            parse(Float64, params["youth_boost"]),
            parse(Int64, params["max_stagnation"]),
            parse(Int64, params["best_max_stagnation"]),
            parse(Int64, params["limit_species"]),

            # evaluation
            parse(Int64, params["dataset_id"]),
            parse(Int64, params["fitness_function_id"]),
            parse(Bool, params["normalise_input"]),
            parse(Bool, params["balance_dataset"]),

            # checkpoint
            params["checkpoint_filename"]
         )
    end
end

function loadConfig(file::String)
    str = open(f->read(f, String), file)

    str = replace(str, r"\r(\n)?" => '\n')
    ls = split(str, "\n")

    ls = filter(l->length(l)>0 && l[1] != '#', ls)
    lsMap = map(x->split(x,'='),ls)
    params = Dict{String,String}()
    for i = 1:length(lsMap)
#         println((i,lsMap[i][1],lsMap[i][2]))
        params[rstrip(lsMap[i][1])] = lstrip(lsMap[i][2])
    end

    return params
end
