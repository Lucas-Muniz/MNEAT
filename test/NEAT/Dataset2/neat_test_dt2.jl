include("../neat_test.jl")


println("--> Dataset 2")
config = "test_dataset2_config.txt"
#config = ARGS[1]
#generations = parse(Int, ARGS[2])
generations = 5000
st2 = run_NEAT_epochs(config, generations, false)
print_best_solution_perfomance(st2)
#print("* Calculating information plane points")
#calculate_information_plane_points(st1)

#=
plot_neat_fitness(st2, imagename="fitness-dt2.png")
plot_neat_IP(st2, imagename="IP-dt2.png", nbins=10)
plot_neat_species(st2, imagename="species-dt2.png")
plot_output_distribution(st2, imagename="distribution-dt2.png")
plot_complexity(st2, imagename="solution-complexity-dt2.png")
plot_accuracy(st2, imagename="accuracy-dt2.png")
plot_total_accuracy(st2, imagename="total-accuracy-dt2.png")
plot_output_distribution_test(st2, imagename="output-test-distribution-best-NEAT-dt2.png")
plot_confusion_matrix(st2; imagename="confusion-matrix-testset-dt2.png", set=:test)
=#
