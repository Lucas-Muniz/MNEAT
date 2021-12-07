include("../neat_test.jl")

#const DEFAULT_DATASET_PATH = "../../../datasets"

println("--> Dataset 1")
#config = "test_dataset1_config.txt"
config = ARGS[1]
generations = parse(Int, ARGS[2])
st1 = run_NEAT_epochs(config, generations, false)
print_best_solution_perfomance(st1)
print("* Calculating information plane points")
calculate_information_plane_points(st1)

#=
# st = get_simulation_state("checkpoint-dt1-limited-10000g.jld2")
# st = get_simulation_state("Resultados_23:07:2021/not-limited/checkpoint-dt1-not-limited-10000g.jld2")
# st = get_simulation_state("Resultados_23:07:2021/limited/checkpoint-dt1-limited-10000g.jld2")
plot_neat_fitness(st1, imagename="fitness-dt1.png")
plot_neat_IP(st1, imagename="IP-dt1.png", nbins=10)
plot_neat_species(st1, imagename="species-dt1.png")
plot_output_distribution(st1, imagename="distribution-dt1.png")
plot_complexity(st1, imagename="solution-complexity-dt1.png")
plot_accuracy(st1, imagename="accuracy-dt1.png")
plot_total_accuracy(st1, imagename="total-accuracy-dt1.png")
plot_output_distribution_test(st1, imagename="output-test-distribution-best-NEAT.png")
plot_confusion_matrix(st1; imagename="confusion-matrix-testset-dt1.png", set=:test)
=#
#pop = getPopulation("checkpoint-dt1.jld2")
