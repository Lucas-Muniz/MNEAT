include("../../neat_test.jl")

const DEFAULT_DATASET_PATH = "../../../../src/datasets"

println("--> Dataset 4")
config = "test_dataset4_config.txt"
#config = ARGS[1]
#generations = parse(Int, ARGS[2])
generations = 5000
st4 = run_NEAT_epochs(config, generations, false)
print_best_solution_perfomance(st4)
#print("* Calculating information plane points")
#calculate_information_plane_points(st4)
println()
print(st4.population.best_fitness[end])

plot_neat_fitness(st4, imagename="fitness-dt4.png")
plot_neat_IP(st4, imagename="IP-dt4.png", nbins=10)
plot_output_distribution(st4, imagename="distribution-dt4.png")
plot_complexity(st4, imagename="solution-complexity-dt4.png")
plot_total_accuracy(st4, imagename="total-accuracy-dt4.png")
plot_output_distribution_test(st4, imagename="output-test-distribution-best-NEAT-dt4.png")

#=
plot_neat_fitness(st4, imagename="fitness-dt4.png")
plot_neat_IP(st4, imagename="IP-dt4.png", nbins=10)
plot_neat_species(st4, imagename="species-dt4.png")
plot_output_distribution(st4, imagename="distribution-dt4.png")
plot_complexity(st4, imagename="solution-complexity-dt4.png")
plot_accuracy(st4, imagename="accuracy-dt4.png")
plot_total_accuracy(st4, imagename="total-accuracy-dt4.png")
plot_output_distribution_test(st4, imagename="output-test-distribution-best-NEAT-dt4.png")
plot_confusion_matrix(st4; imagename="confusion-matrix-testset-dt4.png", set=:test)
=#

# nohup julia neat_test_dt4.jl &> nohup-NEAT-dt4-relatorio.out &
