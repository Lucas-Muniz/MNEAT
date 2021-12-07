using Printf

const SRC_PATH = "../../src"

include(joinpath(SRC_PATH, "NEAT/config.jl"))
include(joinpath(SRC_PATH, "NEAT/global.jl"))
include(joinpath(SRC_PATH, "NEAT/genome.jl"))
include(joinpath(SRC_PATH, "NEAT/chromosome.jl"))
include(joinpath(SRC_PATH, "MutualInformation/information-plane.jl"))
include(joinpath(SRC_PATH, "NEAT/fitness.jl"))
include(joinpath(SRC_PATH, "NEAT/species.jl"))
include(joinpath(SRC_PATH, "NEAT/population.jl"))
include(joinpath(SRC_PATH, "NEAT/simulation_state.jl"))
include(joinpath(SRC_PATH, "NEAT/simulation-plot.jl"))
include(joinpath(SRC_PATH, "NEAT/checkpoint.jl"))
include(joinpath(SRC_PATH, "NEAT/neat.jl"))
#include(joinpath(SRC_PATH, "MLPClassifier/MLPClassifier.jl"))
include(joinpath(SRC_PATH, "FileInterface/fileInterface.jl"))

const DEFAULT_DATASET_PATH = joinpath("..", SRC_PATH, "datasets")

function run_NEAT_epochs(config_file::String, generations::Int=1, report::Bool=false)
    params =  loadConfig(config_file)
    cf = Config(params)
    print("checkpoint filename: ", cf.checkpoint_filename)
    filename = string(cf.checkpoint_filename, ".jld2")
    if isfile(filename)
        g = loadFromFile(filename, "Global")
    else
        g = Global(cf)
    end
    state = SimulationState(g)
    epoch(state, generations, report)
    return state
end

function print_best_solution_perfomance(state::SimulationState)
    best = state.population.best_fitness[end]
    dt_setup = state.setup.dataset_setup
    println("Best solution's fitness: ", best.fitness)
    mlp = convert_to_FluxNet(best, dt_setup.labels)
    error = error_rate(zip(dt_setup.input_train, dt_setup.output_train), mlp)
    acc = accuracy(zip(dt_setup.input_train, dt_setup.output_train), mlp)
    println("Accuracy after NEAT: ", acc)
    println("Error rate after NEAT: ", error)
end

#=
params0 =  loadConfig("test_demo_config.txt")
cf0 = Config(params0)
g0 = Global(cf0)
=#
#ch = create_minimal_chromossome(g)
#mutate_add_layer!(ch, g)
#mutate_add_node!(ch, g)
#mutate_remove_node!(ch, g)
#mutate_weights!(ch, g)
#mutate_enable_bit!(ch)
#evaluation = Evaluation(0,2)
#evaluation(ch)

#population0 = Population(g0)
#epoch(g0, population0, 10)
#time: 0.2 secs

#=
println("--> Dataset demo 2 (overlap)")
p1 = run_NEAT_epochs("test_overlap_config.txt", 1000, false)
#plot_neat_fitness(p1, imagename="Figures/fitness-demo2-t.png")
#plot_neat_IP(p1, imagename="Figures/IP-demo2-t.png")
#plot_neat_species(p1, imagename="Figures/species-demo2-t.png")
print_best_solution_perfomance(p1)
=#
#=
Time: 86.14915895462036 seconds.
Best solution's fitness: 0.42653352353780294
Accuracy after NEAT: 0.598
Error rate after NEAT: 0.402
=#

#=
println("--> Dataset 1")
p2 = run_NEAT_epochs("test_dataset1_config.txt", 1000, false)
#plot_neat_fitness(p2, imagename="Figures/fitness-dt1.png")
#plot_neat_IP(p2, imagename="Figures/IP-dt1.png")
#plot_neat_species(p2, imagename="Figures/species-dt1.png")
print_best_solution_perfomance(p2)
#=
Time: 1550.5301270484924 seconds.
Best solution's fitness: 0.4184397163120568
Accuracy after NEAT: 0.59
Error rate after NEAT: 0.41
=#


println("--> Dataset 2")
p3 = run_NEAT_epochs("test_dataset2_config.txt", 1000, false)
plot_neat_fitness(p3, imagename="Figures/fitness-dt2.png")
plot_neat_IP(p3, imagename="Figures/IP-dt2.png")
plot_neat_species(p3, imagename="Figures/species-dt2.png")
print_best_solution_perfomance(p3)


println("--> Dataset 3")
p4 = run_NEAT_epochs("test_dataset3_config.txt", 1000, false)
plot_neat_fitness(p4, imagename="Figures/fitness-dt3.png")
plot_neat_IP(p4, imagename="Figures/IP-dt3.png")
plot_neat_species(p4, imagename="Figures/species-dt3.png")
print_best_solution_perfomance(p4)


println("--> Dataset 4")
p5 = run_NEAT_epochs("test_dataset4_config.txt", 1000, false)
plot_neat_fitness(p5, imagename="Figures/fitness-dt4.png")
plot_neat_IP(p5, imagename="Figures/IP-dt4.png")
plot_neat_species(p5, imagename="Figures/species-dt4.png")
print_best_solution_perfomance(p5)
=#

#=
println("--> Dataset 5")
p6 = run_NEAT_epochs("test_dataset5_config.txt", 1000, false)
plot_neat_fitness(p6, imagename="Figures/fitness-dt5.png")
plot_neat_IP(p6, imagename="Figures/IP-dt5.png")
plot_neat_species(p6, imagename="Figures/species-dt5.png")
print_best_solution_perfomance(p6)
=#
