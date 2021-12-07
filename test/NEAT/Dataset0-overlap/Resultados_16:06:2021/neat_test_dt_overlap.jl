include("../neat_test.jl")

const DEFAULT_DATASET_PATH = "../../../datasets"

println("--> Dataset demo 2 (overlap)")
config = "test_overlap_config.txt"
p1 = run_NEAT_epochs(config, 5000, false)
print_best_solution_perfomance(p1)


plot_neat_fitness(p1, imagename="fitness-demo2-t1.png")
plot_neat_IP(p1, imagename="IP-demo2-t1.png")
plot_neat_species(p1, imagename="species-demo2-t1.png")
plot_output_distribution(p1, imagename="distribution-demo2-t1.png")
plot_complexity(p1, imagename="solution-complexity-demo2-t1.png")
plot_hit_rate(p1, imagename="hit-rate-demo2-t1.png")

#pop = getPopulation("checkpoint-dt-demo-overlap-t.jld2")
