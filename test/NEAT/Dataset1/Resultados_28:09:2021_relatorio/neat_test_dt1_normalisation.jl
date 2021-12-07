include("../../neat_test.jl")

const DEFAULT_DATASET_PATH = "../../../../src/datasets"

println("--> Dataset 1")
config = "test_dataset1_config.txt"
generations = 5000
st1 = run_NEAT_epochs(config, generations, false)
print_best_solution_perfomance(st1)

plot_neat_fitness(st1, imagename="fitness-dt1.png")
plot_neat_IP(st1, imagename="IP-dt1.png", nbins=10)
plot_output_distribution(st1, imagename="distribution-dt1.png")
plot_complexity(st1, imagename="solution-complexity-dt1.png")
plot_total_accuracy(st1, imagename="total-accuracy-dt1.png")
plot_output_distribution_test(st1, imagename="output-test-distribution-dt1.png")

#=
plot_neat_fitness(st1, imagename="fitness-dt1-pt.png", language=:pt)
plot_neat_IP(st1, imagename="IP-dt1-pt.png", nbins=10, language=:pt)
plot_output_distribution(st1, imagename="distribution-dt1-pt.png", language=:pt)
plot_complexity(st1, imagename="solution-complexity-dt1-pt.png", language=:pt)
plot_total_accuracy(st1, imagename="total-accuracy-dt1-pt.png", language=:pt)
plot_output_distribution_test(st1, imagename="output-test-distribution-dt1-pt.png", language=:pt)
=#

#=
st1 = get_simulation_state("checkpoint-dt1-limited-normalised-5000g-t2.jld2")
plot_neat_fitness(st1, imagename="fitness-dt1-crossover.png")
plot_neat_IP(st1, imagename="IP-dt1-crossover.png", nbins=10)
plot_neat_species(st1, imagename="species-dt1-crossover.png")
plot_output_distribution(st1, imagename="distribution-dt1-crossover.png")
plot_complexity(st1, imagename="solution-complexity-dt1-crossover.png")
plot_accuracy(st1, imagename="accuracy-dt1-crossover.png")
plot_total_accuracy(st1, imagename="total-accuracy-dt1-crossover.png")
plot_output_distribution_test(st1, imagename="output-test-distribution-best-NEAT-crossover.png")
plot_confusion_matrix(st1; imagename="confusion-matrix-testset-dt1-crossover.png", set=:test)
=#
