include("../neat_test.jl")


println("--> Dataset 4")
config = "test_dataset4_config.txt"
p4 = run_NEAT_epochs(config, 2000, false)
print_best_solution_perfomance(p4)

#=
plot_neat_fitness(p4, imagename="fitness-dt4.png")
plot_neat_IP(p4, imagename="IP-dt4.png")
plot_neat_species(p4, imagename="species-dt4.png")
plot_output_distribution(p4, imagename="distribution-dt4.png")
plot_complexity(p4, imagename="solution-complexity-dt4.png")
plot_hit_rate(p4, imagename="hit-rate-dt4.png")
=#
#pop = getPopulation("checkpoint-dt4.jld2")
