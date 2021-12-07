using Statistics

#g::Global, p::Population,
function epoch(state::SimulationState, n::Int, report::Bool=true, save_best::Bool=false,
               checkpoint_interval::Int=15, checkpoint_generation=0)
    #= Runs NEAT's genetic algorithm for n epochs.
        Keyword arguments:
        report -- show stats at each epoch (default true)
        save_best -- save the best chromosome from each epoch (default False)
        checkpoint_interval -- time in minutes between saving checkpoints (default 15 minutes)
        checkpoint_generation -- time in generations between saving checkpoints
            (default 0 -- option disabled)
    =#
    Random.seed!(123456)
    t0 = time()
    tc = time() # for saving checkpoints
    #fitness = Evaluation(g.cf.dataset_id, g.cf.fitness_function_id)
    #p.evaluate()

    # Evaluate individuals
    #p.evaluate(p.setup)
    state.evaluate!(state.setup)
    g = state.g
    p = state.population

    initial_generation = p.generation == 0 ? 0 : 1

    for gen in initial_generation:n
        p.generation += 1

        print("\n ****** Running generation $(p.generation) ******")

        if report println("\n ****** Running generation $(p.generation) ******") end

        # Speciates the population
        speciate(g, p, report)

        # Current population's average fitness
        push!(p.avg_fitness, average_fitness(p))

        # Current generation's best chromosome
        bestfit, bestidx = findmax(map(ch-> ch.fitness, p.population))
        best = p.population[bestidx]
        previous_best_fitness = p.generation == 0 ? 0 : p.best_fitness[end].fitness
        push!(p.best_fitness, best)

        if previous_best_fitness == best.fitness
            p.best_no_improvement_age += 1
        else
            p.best_no_improvement_age = 0
        end

        # Which species has the best chromosome?
        for s in p.species
            s.hasBest = false
            if best.species_id == s.id
                s.hasBest = true
            end
        end

        limit_species!(g, p)

        #-----------------------------------------
        # Prints chromosome's parents id:  {dad_id, mon_id} -> child_id
#         map(ch-> @printf("{%3d; %3d} -> %3d   Nodes %3d   Connections %3d\n",
#                          ch.parent1_id, ch.parent2_id, ch.id, size(ch)[1], size(ch)[2]), p.population)
        #-----------------------------------------

        remove_stagnated_species!(p, g, report)

        # Compute spawn levels for each remaining species
        compute_spawn_levels(g, p)

        remove_unspawned_species!(p, report)

        # Logging speciation stats
        log_species(p)

        print_debugging_information(report, p, best)

        # Stops the simulation
        if best.fitness >= g.cf.max_fitness_threshold
            @printf("Best individual found in epoch %s - complexity: %s\n", p.generation, number_nodes(best))
            break
        end

        # -------------------------- Producing new offspring -------------------------- #
        new_population = Tuple{MLPChromosome, Bool}[] # next generation's population

        # Spawning new population
        for s in p.species new_population = vcat(new_population, reproduce(g, s, state.setup)) end

        update_population!(p, g, state.setup, new_population, report)

        if time() > tc + 60 * checkpoint_interval
            create_checkpoint(g, p, report)
            tc = time() # updates the counter
        elseif  checkpoint_generation != 0 && p.generation % checkpoint_generation == 0
            create_checkpoint(g, p, report)
        end
    end
    t1 = time()
    Δt = t1 - t0
    p.simulation_time += Δt
    t2 = time()
    create_checkpoint(g, p, report)
    total_time = Δt + (time() - t2)
    println("Time: $total_time seconds.")
end

function remove_stagnated_species!(p::Population, g::Global, report)
    # Remove stagnated species and its members (except if it has the best chromosome)
    number_species = length(p.species)
    speciesToKeep = trues(number_species)
    deletedSpeciesIds = Int[]

    if number_species > 2 && p.best_no_improvement_age > g.cf.best_max_stagnation
        sort!(p.species, by = s-> s.last_avg_fitness, rev = true)
        speciesToKeep = map((s) -> s[2].hasBest || s[1] == 1 || s[1] == 2, enumerate(p.species))
        #speciesToKeep[3:end] .= false
        p.best_no_improvement_age = 0
        append!(deletedSpeciesIds, map((s) -> s.id, p.species[.!speciesToKeep]))
    else
        number_kept_species = number_species
        for i = 1:length(p.species)
            if p.species[i].no_improvement_age > g.cf.max_stagnation
                if (!p.species[i].hasBest && p.species[i].no_improvement_age > 2 * g.cf.max_stagnation) && number_kept_species > 2
                    if report @printf("\n   Species %2d age %2s (with %2d individuals) is stagnated: removing it",
                                      p.species[i].id, p.species[i].age, length(p.species[i])) end
                    speciesToKeep[i] = false
                    number_kept_species -= 1
                    push!(deletedSpeciesIds, p.species[i].id)
                end
            end
        end
    end

    #if length(p.species) > 1 p.species = p.species[speciesToKeep] end
    p.species = p.species[speciesToKeep] # prune unwanted species

    # remove species' chromosomes from population
    # prune unwanted chromosomes
    p.population = filter((i) -> i.species_id ∉ deletedSpeciesIds, p.population)
end

function remove_unspawned_species!(p::Population, report)
    # Removing species with spawn amount = 0
    speciesToKeep = trues(length(p.species))
    deletedSpeciesIds = Int64[]
    for i = 1:length(p.species)

        # This rarely happens
        if p.species[i].spawn_amount == 0
            if report @printf("\n   Species %2d age %2s removed: produced no offspring",p.species[i].id, p.species[i].age) end
            speciesToKeep[i] = false
            push!(deletedSpeciesIds,p.species[i].id)
        end
    end
    p.species = p.species[speciesToKeep] # prune unwanted species

    # remove species' chromosomes from population
    p.population = filter((i) -> i.species_id ∉ deletedSpeciesIds, p.population)  # prune unwanted chromosomes
end

function print_debugging_information(report::Bool, p::Population, best::MLPChromosome)
    if report
        @printf("\nPopulation's average fitness: %3.5f stdev: %3.5f", p.avg_fitness[end], stdeviation(p))
        @printf("\nBest fitness: %2.12s - size: %s - species %s - id %s", best.fitness, number_nodes(best), best.species_id, best.id)

        # print some "debugging" information
        @printf("\nSpecies length: %d totalizing %d individuals", length(p.species), sum([length(s) for s in p.species]))
        @printf("\nSpecies ID       : %s",   [s.id for s in p.species])
        @printf("\nEach species size: %s",   [length(s) for s in p.species])
        @printf("\nAmount to spawn  : %s",   [s.spawn_amount for s in p.species])
        @printf("\nSpecies age      : %s",   [s.age for s in p.species])
        @printf("\nSpecies no improv: %s\n", [s.no_improvement_age for s in p.species]) # species no improvement age

        for s in p.species println(s) end
    end
end

function update_population!(p::Population, g::Global, fitness::Function, new_population, report)
    # ----------------------------------------------#
    # Controls target population under or overflow  #
    # ----------------------------------------------#
    fill = p.popsize - length(new_population)
    if fill < 0 # overflow
        if report println("\n   Removing $(abs(fill)) excess individual(s) from the new population") end
        # TODO: This is dangerous? I can't remove a species' representant!
        #new_population = new_population[1:end+fill] # Removing the last added members
        while fill < 0
            pos = rand((1:length(new_population)))
            # checks flag variable which indicates the individuals that must be kept
            if !new_population[pos][2]
                deleteat!(new_population, pos)
                fill += 1
            end
        end
    end

    new_population = getindex.(new_population, 1) # Delete flag variables

    if fill > 0 # underflow
        if report println("\n   Producing $fill more individual(s) to fill up the new population") end

        # TODO:
        # what about producing new individuals instead of reproducing?
        # increasing diversity from time to time might help
        while fill > 0
            # Selects a random chromosome from population
            parent1 = p.population[rand(1:length(p.population))]
            # Search for a mate within the same species
            found = false
            for parent2 in p.population
                if parent2.species_id == parent1.species_id && parent2.id != parent1.id
                    child = mutate!(crossover(g, parent1, parent2),g)
                    fitness(child)
                    push!(new_population, child)
                    found = true
                    break
                end
            end
            if !found
                child = mutate!(deepcopy(parent1),g)
                fitness(child)
                push!(new_population, child) # will irreversibly mutate parent. ok?
            end # If no mate was found, just mutate it

            fill -= 1
        end
    end

    @assert p.popsize == length(new_population) # Different population sizes!

    # Updates current population
    p.population = new_population
end
