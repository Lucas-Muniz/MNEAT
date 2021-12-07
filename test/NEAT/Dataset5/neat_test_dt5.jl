include("../neat_test.jl")


println("--> Dataset 5")
config = "test_dataset5_config.txt"
p5 = run_NEAT_epochs(config, 2000, false)
print_best_solution_perfomance(p5)

#=
plot_neat_fitness(p5, imagename="fitness-dt5.png")
plot_neat_IP(p5, imagename="IP-dt5.png")
plot_neat_species(p5, imagename="species-dt5.png")
plot_output_distribution(p5, imagename="distribution-dt5.png")
plot_complexity(p5, imagename="solution-complexity-dt5.png")
plot_hit_rate(p5, imagename="hit-rate-dt5.png")
=#
#pop = getPopulation("checkpoint-dt5.jld2")
