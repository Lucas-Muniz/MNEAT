mutable struct SimulationState
    g::Global
    population::Population
    setup::Evaluation
    evaluate!::Function
    function SimulationState(g::Global)
        p = Population(g, g.cf.checkpoint_filename)
        setup = Evaluation(g.cf.dataset_id, g.cf.fitness_function_id, normalise=g.cf.normalise_input,
                           balance=g.cf.balance_dataset)
        fitness_function = (f::Function) -> f.(p.population)
        new(g, p, setup, fitness_function)
    end
    function SimulationState(g::Global, p::Population)
        setup = Evaluation(g.cf.dataset_id, g.cf.fitness_function_id, normalise=g.cf.normalise_input,
                           balance=g.cf.balance_dataset)
        fitness_function = (f::Function) -> f.(p.population)
        new(g, p, setup, fitness_function)
    end
end

function Base.show(io::IO, state::SimulationState)
    print(io, "State: \n")
    println(io, state.population)
    print(io, state.setup.dataset_setup)
end
