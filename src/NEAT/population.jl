mutable struct Population
    # Manages all the species
    population::Vector{MLPChromosome}
    popsize::Int64
    species::Vector{Species}
    species_history::Vector{Vector{Tuple{Int, Int}}}#Vector{<:Union{Array{Int64,1}, Tuple{Int, Int}}}
    generation::Int64
    avg_fitness::Vector{Float64}
    best_fitness::Vector{MLPChromosome}
    best_no_improvement_age::Int64
    number_species::Int
    simulation_time::Float64
    #setup::Evaluation
    #evaluate::Function # Evaluates population. Override this method in your experiments
    function Population(g::Global, checkpoint::String="")

        if isfile(string(checkpoint, ".jld2"))
            resume_checkpoint(g.cf.checkpoint_filename)
        else
            popsize = g.cf.pop_size # total population size
            population = Vector{MLPChromosome}(undef, popsize)
            #minimal_chromosome = create_minimal_chromossome(g)
            for i in 1:popsize
                g.innov_number = 0
                population[i] = create_minimal_chromossome(g)
            end
            #=setup = Evaluation(g.cf.dataset_id, g.cf.fitness_function_id, normalise=g.cf.normalise_input,
                               balance=g.cf.balance_dataset)=#

            p = new(population, popsize,
                    Species[],Array{Int64,1}[], # currently living species and species history
                    -1, # generation
                    Float64[], # avg_fitness
                    MLPChromosome[], # best_fitness
                    0,
                    0,
                    0.0#,
                    #setup
                    );
            #p.evaluate = (f::Function) -> f.(p.population); # Evaluates population. Override this method in your experiments
            p
        end
    end
    function Population(g::Global, population::Vector{MLPChromosome}, popsize::Int64,
                        species::Vector{Species}, species_history::Vector{Vector{Tuple{Int, Int}}},
                        generation::Int64, avg_fitness::Vector{Float64}, best_fitness::Vector{MLPChromosome},
                        number_species::Int, simulation_time::Float64)
        setup = Evaluation(g.cf.dataset_id, g.cf.fitness_function_id, normalise=g.cf.normalise_input)
        evaluate_function = (f::Function) -> f.(p.population)
        new(population, popsize, species, species_history, generation, avg_fitness,
            best_fitness, number_species, simulation_time, setup, evaluate_function)
    end
end

function Base.show(io::IO, p::Population)
    @printf(io,"Population size: %3d   Total species: %3d", p.popsize, length(p.species))
end

function remove(p::Population, ch::MLPChromosome)
    # Removes a chromosome from the population
    deleteat!(p.population,findfirst(p.population,ch))
    return
end

function remove_specie!(p::Population, id::Int64)
    index = findfirst((s) -> s.id == id, p.species)
    if index != nothing
        deleteat!(p.species, index)
        p.population = filter((i) -> i.species_id != id, p.population)
    else
        println("Specie with id $id not found within population.")
    end
end

function limit_species!(g::Global, p::Population)
    # If the value of limit_species is less than 1, species limitation won't be enabled
    if g.cf.limit_species > 0
        sort!(p.species, by = s -> s.age, rev = true)
        i = 1
        while length(p.species) > g.cf.limit_species && i <= length(p.species)
            sp = p.species[i]
            if !sp.hasBest
                remove_specie!(p, sp.id)
                i -= 1
            end
            i += 1
        end
    end
end

function speciate(g::Global, p::Population, report::Bool)
    # Group chromosomes into species by similarity
        # Speciate the population
    for individual in p.population
        found = false
        for s in p.species
            if distance(individual, s.representant, g.cf) < g.cf.compatibility_threshold
                add(s, individual)
                found = true
                break
            end
        end

        if !found
            push!(p.species, Species(g, individual))
            p.number_species += 1
        end
    end

    # eliminate empty species
    keep = map((s)->length(s)==0 ? false : true, p.species)
    if report
        for i = 1:length(keep)
            if !keep[i] println("Removing species $(p.species[i].id) for being empty") end
        end
    end
    p.species = p.species[keep]
    #set_compatibility_threshold(g, p)
end

function set_compatibility_threshold(g::Global, p::Population)
    # controls compatibility threshold
    if length(p.species) > g.cf.species_size
        g.cf.compatibility_threshold += g.cf.compatibility_change
    elseif length(p.species) < g.cf.species_size
        if g.cf.compatibility_threshold > g.cf.compatibility_change
            g.cf.compatibility_threshold -= g.cf.compatibility_change
        else
            println("Compatibility threshold cannot be changed (minimum value has been reached)")
        end
    end
end

# Returns the average raw fitness of population
average_fitness(p::Population) = mean(map((c) -> c.fitness::Float64, p.population))

stdeviation(p::Population) = std(map((c) -> c.fitness::Float64, p.population))

function compute_spawn_levels(g::Global, p::Population)
    #  Compute each species' spawn amount (Stanley, p. 40)

    # 1. Boost if young and penalize if old
    # TODO: does it really increase the overall performance?
    species_stats = zeros(length(p.species))
    for i = 1:length(p.species)
        s = p.species[i]
        #species_stats[i] = s.age < g.cf.youth_threshold ? average_fitness(s) * g.cf.youth_boost :
        #    s.age > g.cf.old_threshold ? average_fitness(s) * g.cf.youth_boost : average_fitness(s)
        if s.age < g.cf.youth_threshold
            species_stats[i] = average_fitness(s) * g.cf.youth_boost
        elseif s.age > g.cf.old_threshold
           species_stats[i] = average_fitness(s) * g.cf.old_penalty
        else
            species_stats[i] = average_fitness(s)
        end
    end

    # 2. Share fitness (only usefull for computing spawn amounts)
    # Sharing the fitness is only meaningful here
    # we don't really have to change each individual's raw fitness
    total_average = sum(species_stats)
    #push!(p.avg_fitness, mean(total_average))

     # 3. Compute spawn
    for i= 1:length(p.species)
        s = p.species[i]
        s.spawn_amount = Int(round((species_stats[i] * p.popsize / total_average)))
    end
end

function tournamentSelection(p::Population, k=2)
    # Tournament selection with size k (default k=2).
    # randomly select k competitors
    chs = p.population[randperm(length(p.population))[1:k]]
    best = chs[1]
    for ch in chs # choose best among randomly selected
        best = ch.fitness > best.fitness ? ch : best
    end
    return best
end

function simple_log_species(p::Population)
    # Logging species data for visualizing speciation
    specById = sort(p.species, by=s->s.id)
    spec_size = zeros(Int64,specById[end].id+1)
    map(s->spec_size[s.id]=length(s) , specById)
    push!(p.species_history,spec_size)
end

function log_species(p::Population)
    # Logging species data for visualizing speciation
    specById = sort(p.species, by = s->s.id)
    spec_size = map((s) -> (s.id, length(s)) , specById)
    push!(p.species_history, spec_size)
end

function population_diversity(p::Population)
    # Calculates the diversity of population: total average weights,
    # number of connections, nodes

    num_nodes = 0
    avg_weights = 0.0

    for ch in p.population
        num_nodes += number_nodes(ch)
        sum_weights = (n) -> sum(n.weights)
        avg_weights = map((l) -> sum(sum_weights.(l.nodes)), ch.layers) |> sum
    end

    total = length(p.population)
    return num_nodes/total, avg_weights/total
end
