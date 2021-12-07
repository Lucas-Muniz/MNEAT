include("../neat_test.jl")

#const DEFAULT_DATASET_PATH = "../../../src/datasets"

println("--> Dataset demo 2 (overlap)")
config = "test_overlap_config.txt"
#p1 = run_NEAT_epochs(config, 5000, false)
p0 = run_NEAT_epochs(config, 3000, false)
print_best_solution_perfomance(p0)


#plot_neat_fitness(p0, imagename="fitness-demo2-t.png")
#plot_neat_IP(p0, imagename="IP-demo2-t.png")
#plot_neat_species(p0, imagename="species-demo2-t.png")
#plot_output_distribution(p0, imagename="distribution-demo2-t.png")
#plot_complexity(p0, imagename="solution-complexity-demo2-t.png")
#plot_hit_rate(p0, imagename="hit-rate-demo2-t.png")

#pop = getPopulation("checkpoint-dt-demo-overlap.jld2")
