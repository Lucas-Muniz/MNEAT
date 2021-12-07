mutable struct CheckpointPopulation
    # Manages all the species
    population::Vector{MLPChromosome}
    popsize::Int64
    species::Vector{Species}
    species_history::Vector{Vector{Tuple{Int, Int}}}#Vector{<:Union{Array{Int64,1}, Tuple{Int, Int}}}
    generation::Int64
    avg_fitness::Vector{Float64}
    best_fitness::Vector{MLPChromosome}
    number_species::Int
    simulation_time::Float64
    #setup::Evaluation
    #evaluate::Function # Evaluates population. Override this method in your experiments
    function CheckpointPopulation(p::Population)
        checkpoint_population =
            new(p.population,
                p.popsize,
                p.species,
                p.species_history,
                p.generation,
                p.avg_fitness,
                p.best_fitness,
                p.number_species,
                p.simulation_time
            )
        return checkpoint_population
    end
end

function generate_population(p::CheckpointPopulation, g::Global)
    Population(g,
               p.population,
               p.popsize,
               p.species,
               p.species_history,
               p.generation,
               p.avg_fitness,
               p.best_fitness,
               p.number_species,
               p.simulation_time
    )
end

function generate_population(p::Population, g::Global)
    p
end

function get_population(filename::String)
    data = jldopen(filename, "r")["simulation"]
    generate_population(data["Population"], data["Global"])
end

function get_simulation_state(filename::String)
    data = jldopen(filename, "r")["simulation"]
    SimulationState(data["Global"], data["Population"])
end


function resume_checkpoint(checkpoint_filename::String)
    # Resumes the simulation from a previous saved point.
    checkpoint = string(checkpoint_filename, ".jld2")
    if isfile(checkpoint)
        #return loadFromFile(checkpoint, "Population")
        return get_population(checkpoint)
    else
        error("There is no checkpoint file '$checkpoint'")
    end
end

#Dates.format(now(), "HH:MM:SS-E:U:yyyy")
using Dates

function create_checkpoint(g::Global, p::Population, report::Bool)
        filename = string(g.cf.checkpoint_filename, ".jld2")
        # Saves global variables.
        addToFile(filename, "Global", g)
        # Saves the current simulation state.
        #population = CheckpointPopulation(p)
        population = p
        addToFile(filename, "Population", population)
        # Saves current time and date
        current_date = Dates.now()
        time = Dates.format(current_date, "HH:MM:SS")
        date = Dates.format(current_date, "dd/mm/yyyy")
        addToFile(filename, "Time", time)
        addToFile(filename, "Date", date)

        if report println("Creating checkpoint file at generation: ", p.generation)  end
end
