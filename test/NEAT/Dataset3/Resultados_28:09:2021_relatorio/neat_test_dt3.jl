include("../../neat_test.jl")

const DEFAULT_DATASET_PATH = "../../../../src/datasets"

println("--> Dataset 3")
config = "test_dataset3_config.txt"
#config = ARGS[1]
#generations = parse(Int, ARGS[2])
generations = 5000
st3 = run_NEAT_epochs(config, generations, false)
print_best_solution_perfomance(st3)
#print("* Calculating information plane points")
#calculate_information_plane_points(st3)

plot_neat_fitness(st3, imagename="fitness-dt3.png")
#plot_neat_IP(st3, imagename="IP-dt3.png", nbins=10)
plot_output_distribution(st3, imagename="distribution-dt3.png")
plot_complexity(st3, imagename="solution-complexity-dt3.png")
plot_total_accuracy(st3, imagename="total-accuracy-dt3.png")
plot_output_distribution_test(st3, imagename="output-test-distribution-best-NEAT-dt3.png")

#=
plot_neat_fitness(st3, imagename="fitness-dt3.png")
plot_neat_IP(st3, imagename="IP-dt3.png", nbins=10)
plot_neat_species(st3, imagename="species-dt3.png")
plot_output_distribution(st3, imagename="distribution-dt3.png")
plot_complexity(st3, imagename="solution-complexity-dt3.png")
plot_accuracy(st3, imagename="accuracy-dt3.png")
plot_total_accuracy(st3, imagename="total-accuracy-dt3.png")
plot_output_distribution_test(st3, imagename="output-test-distribution-best-NEAT-dt3.png")
plot_confusion_matrix(st3; imagename="confusion-matrix-testset-dt3.png", set=:test)
=#

# nohup julia neat_test_dt3.jl &> nohup-NEAT-dt3-relatorio.out &
