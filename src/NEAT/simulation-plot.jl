using Colors, FixedPointNumbers, LaTeXStrings, StatsBase#, ColorSchemes

#include("../MutualInformation/information-plane.jl")
include("plots_legends.jl")

function calculate_information_plane_point(ch::MLPChromosome, dt_setup::DatasetSetup, nbins::Int)
    mlp = convert_to_FluxNet(ch, dt_setup.labels)
    evaluate_information_plane(mlp, dt_setup.input_train, dt_setup.output_train, nbins)
end

function calculate_information_plane_points(state::SimulationState, nbins::Int)
    key = joinpath("simulation", "IP-points")
    #filename = string(state.g.cf.checkpoint_filename, ".jld2")
    filename = string(state.g.cf.checkpoint_filename, "-IP-points", ".jld2")
    if isfile(filename)
        local points
        file = load(filename)
        if haskey(file, key)
            points = file[key]
        else
            points = calculate_information_plane_point.(state.population.best_fitness, [state.setup.dataset_setup], [nbins])
            file[key] = points
    		#save(filename, file)
            save(filename, file)
            #addToFile(filename, key, points)
        end
        return points
    else
        file = Dict()
        points = calculate_information_plane_point.(state.population.best_fitness, [state.setup.dataset_setup], [nbins])
        #addToFile(filename, key, points)
        file[key] = points
        save(filename, file)
        return points
    end
end

function plot_neat_IP(state::SimulationState; nbins::Int=10, imagename::String = "IP-MNEAT.png", language::Symbol=:en, xlim=nothing, ylim=nothing)
    points = calculate_information_plane_points(state, nbins)
    legends = language == :pt ? IP_legends_pt : IP_legends_en
    #points = calculate_information_plane_point.(state.population.best_fitness, [state.setup.dataset_setup], [nbins])
    generations = length(points)
    z_colors = [cgrad(:inferno, [0.01, 0.99])[z] for z ∈ range(0.0, 1.0, length = generations+1)]
    local graph
    local p
    #p = plot(bar = true, legend=false, cbartitle = legends["cbartitle"],
         #xlabel = L"I(X,Z)", ylabel = L"I(Z,Y)", title = legends["title"])

    for gen = 1:generations
        #color_index = gen
        x = getindex.(points[gen], 1)
        y = getindex.(points[gen], 2)
        markers = get_marker_vector(length(x))
        if gen == 1
            p = plot(x, y, line_z = gen, cbar = true, legend=false, cbartitle = legends["cbartitle"],
                 xlabel = L"I(X,Z)", ylabel = L"I(Z,Y)", title = legends["title"])
            #plot!(p, x, y, line_z = gen, bar = true, legend=false, cbartitle = legends["cbartitle"])
            if xlim != nothing plot!(p, xlim=xlim) end
            if ylim != nothing plot!(p, ylim=ylim) end
            Plots.scatter!(p, x, y, color = z_colors[gen], m = markers, markersize = 7)

        else
            plot!(p, x, y, line_z = gen)
            Plots.scatter!(p, x, y, color = z_colors[gen], m = markers,  markersize = 7)
        end
    end
    savefig(p, imagename)
end

function plot_neat_IP_gif(state::SimulationState; imagename::String = "IP-MNEAT.gif", fps::Int = 15)
    points = calculate_information_plane_point.(state.population.best_fitness, [state.setup.dataset_setup])
    generations = length(points)-1
    z_colors = [cgrad(:inferno, [0.01, 0.99])[z] for z ∈ range(0.0, 1.0, length = generations+1)]
    anim = Animation()
    for gen = 0:generations
        color_index = gen+1
        x = getindex.(points[gen+1], 1)
        y = getindex.(points[gen+1], 2)
        if gen == 0
            p = plot(x, y, line_z = gen, cbar = true, legend=false, cbartitle = "Generation",
                 xlabel = L"I(X,T)", ylabel = L"I(T,Y)", title = "Information Plane (MNEAT)")
            Plots.scatter!(x, y, color = z_colors[color_index], markersize = 7)

        else
            plot!(x, y, line_z = gen)
            Plots.scatter!(x, y, color = z_colors[color_index],  markersize = 7)
        end
        frame(anim)
    end
    gif(anim, imagename, fps=fps)
end

function plot_neat_fitness(state::SimulationState; imagename::String = "fitness-MNEAT.png", language::Symbol=:en)
    legends = language == :pt ? fitness_legends_pt : fitness_legends_en
    p = state.population
    generations = 0:p.generation
    best_fit = map((ch) -> ch.fitness, p.best_fitness)
    avg_fit = p.avg_fitness
    graph = plot(xlabel = legends["xlabel"], ylabel = legends["ylabel"], title = legends["title"], legend=:bottomright)
    plot!(graph, generations, best_fit, label=legends["label1"], color=parse(RGB{N0f8}, "green4"), line = 2)
    plot!(graph, generations, avg_fit, label=legends["label2"], color=parse(RGB{N0f8}, "dodgerblue1"), line = 2)
    savefig(graph, imagename)
end

function plot_neat_species(state::SimulationState; imagename::String = "species-MNEAT.png")
    species = get_species_evolution(state.population)
    colors = distinguishable_colors(length(species), [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    graph = plot(xlabel = "Generation", ylabel = "Size", title = "Evolution of species", legend=false) # legend=:outertopright
    for sp = 1:length(species)
        plot!(graph, species[sp], label="Specie $sp", color=colors[sp], marker=true)
    end
    savefig(graph, imagename)
end

function get_species_evolution(p::Population)
    total_species = p.number_species
    species = [Tuple{Int, Int}[] for i in 1:total_species]
    for gen = 0:p.generation-1
        for (sp_id, len) in p.species_history[gen+1]
            point = (gen, len)
            push!(species[sp_id], point)
        end
    end
    species
end

function plot_output_distribution(state::SimulationState; imagename::String = "output-distribution-best-MNEAT.png", language::Symbol=:en)
    legends = language == :pt ? output_dist_legends_pt : output_dist_legends_en
    p = state.population
    labels = state.setup.dataset_setup.labels
    best = p.best_fitness[end]
    input = state.setup.dataset_setup.input_train
    expected_output = getindex.(state.setup.dataset_setup.output_train, 1)
    mlp = convert_to_FluxNet(best,  labels)
    output = predict.([mlp], input)
    best_fitness_distribution = countmap(output)
    expected_distribution = countmap(expected_output)
    best_fitness_frequencies = Vector{Int}(undef, length(labels))
    for i = 1:length(labels)
        if haskey(best_fitness_distribution, labels[i])
            best_fitness_frequencies[i] = best_fitness_distribution[labels[i]]
        else
            best_fitness_frequencies[i] = 0
        end
    end
    expected_frequencies = map(l -> haskey(expected_distribution, l) ? expected_distribution[l] : 0, labels) #getindex.([expected_distribution], labels)
    achieved_prob = best_fitness_frequencies ./ sum(best_fitness_frequencies)
    expected_prob = expected_frequencies ./ sum(expected_frequencies)
    colors = distinguishable_colors(2, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    p1 = plot(labels, achieved_prob, xlabel = legends["xlabel"], xticks = 0:1:length(labels), linetype=[:bar],
              ylabel = legends["ylabel"], title = legends["title1"], legend=false, color=:blue)
    p2 = plot(labels, expected_prob, xlabel = legends["xlabel"], xticks = 0:1:length(labels), linetype=[:bar],
              ylabel = legends["ylabel"], title = legends["title2"], legend=false, color=:green)
    graph = plot(p1, p2) # legend=:outertopright
    savefig(graph, imagename)
end

function plot_output_distribution_test(state::SimulationState; imagename::String = "output-test-distribution-best-MNEAT.png", language::Symbol=:en)
    legends = language == :pt ? output_dist_legends_pt : output_dist_legends_en
    p = state.population
    labels = state.setup.dataset_setup.labels
    best = p.best_fitness[end]
    input = state.setup.dataset_setup.input_test
    expected_output = getindex.(state.setup.dataset_setup.output_test, 1)
    mlp = convert_to_FluxNet(best,  labels)
    output = predict.([mlp], input)
    best_fitness_distribution = countmap(output)
    expected_distribution = countmap(expected_output)
    best_fitness_frequencies = Vector{Int}(undef, length(labels))
    for i = 1:length(labels)
        if haskey(best_fitness_distribution, labels[i])
            best_fitness_frequencies[i] = best_fitness_distribution[labels[i]]
        else
            best_fitness_frequencies[i] = 0
        end
    end
    expected_frequencies = map(l -> haskey(expected_distribution, l) ? expected_distribution[l] : 0, labels)
    achieved_prob = best_fitness_frequencies ./ sum(best_fitness_frequencies)
    expected_prob = expected_frequencies ./ sum(expected_frequencies)
    colors = distinguishable_colors(2, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    p1 = plot(labels, achieved_prob, xlabel = legends["xlabel"], xticks = 0:1:length(labels), linetype=[:bar],
              ylabel = legends["ylabel"], title = legends["title1"], legend=false, color=:blue)
    p2 = plot(labels, expected_prob, xlabel = legends["xlabel"], xticks = 0:1:length(labels), linetype=[:bar],
              ylabel = legends["ylabel"], title = legends["title2"], legend=false, color=:green)
    graph = plot(p1, p2) # legend=:outertopright
    savefig(graph, imagename)
end

function plot_complexity(state::SimulationState; imagename::String = "complexity-MNEAT.png", language::Symbol=:en)
    legends = language == :pt ? complexity_legends_pt : complexity_legends_en
    p = state.population
    generations = 0:p.generation
    complexities = chromosome_complexity.(p.best_fitness)
    graph = plot(xlabel = legends["xlabel"], ylabel = legends["ylabel"], title = legends["title"], legend=false)
    plot!(graph, generations, complexities, color=:red, line = 2)
    savefig(graph, imagename)
end

function plot_accuracy(state::SimulationState; imagename::String = "accuracy-MNEAT.png", language::Symbol=:en)
    legends = language == :pt ? accuracy_legends_pt : accuracy_legends_en
    p = state.population
    generations = 0:p.generation
    labels = state.setup.dataset_setup.labels
    mlps = convert_to_FluxNet.(p.best_fitness,  [labels])
    errors = error_rate.([zip(state.setup.dataset_setup.input_train, state.setup.dataset_setup.output_train)], mlps)
    hits = 1 .- errors
    graph = plot(xlabel = legends["xlabel"], ylabel = legends["ylabel"], title = legends["title"],
                legend=false, ylims = (0,1))
    plot!(graph, generations, hits, color=:purple, line = 2)
    savefig(graph, imagename)
end

function plot_total_accuracy(state::SimulationState; imagename::String = "total-accuracy-MNEAT.png", language::Symbol=:en)
    legends = language == :pt ? accuracy_legends_pt : accuracy_legends_en
    p = state.population
    generations = 0:p.generation
    labels = state.setup.dataset_setup.labels
    mlps = convert_to_FluxNet.(p.best_fitness,  [labels])
    errors_training = error_rate.([zip(state.setup.dataset_setup.input_train, state.setup.dataset_setup.output_train)], mlps)
    errors_test = error_rate.([zip(state.setup.dataset_setup.input_test, state.setup.dataset_setup.output_test)], mlps)
    hits_training = 1 .- errors_training
    hits_test = 1 .- errors_test
    graph = plot(xlabel = legends["xlabel"], ylabel = legends["ylabel"], title = legends["title"],
                 ylims = (0,1), legend=:bottomright)
    plot!(graph, generations, hits_training, color=:blue, line = 2, label=legends["label1"])
    plot!(graph, generations, hits_test, color=:green, line = 2, label=legends["label2"])
    savefig(graph, imagename)
end


function calculate_confusion_matrix(state::SimulationState, set::Symbol = :test)
    p = state.population
    labels = state.setup.dataset_setup.labels
    num_labels = length(labels)
    mlp = convert_to_FluxNet(p.best_fitness[end],  labels)
    confusion_matrix = zeros(Float64, (num_labels, num_labels))
    label_map = map(t -> (t[2], t[1]), enumerate(labels)) |> Dict
    if set == :test
        input = state.setup.dataset_setup.input_test
        output = predict.([mlp], input)
        true_labels = getindex.(state.setup.dataset_setup.output_test, 1)
    elseif set == :training
        input = state.setup.dataset_setup.input_train
        output = predict.([mlp], input)
        true_labels = getindex.(state.setup.dataset_setup.output_train, 1)
    end
    set_length = length(output)
    mapping = map((l) -> (label_map[l[1]], label_map[l[2]]), zip(true_labels, output))
    map(m -> confusion_matrix[m...] += 1, mapping)
    confusion_matrix ./ set_length
end

function plot_confusion_matrix(state::SimulationState; imagename::String = "confusion-matrix-MNEAT.png", set::Symbol = :test)
    set_text = set == :test ? "test" : "training"
    confusion_matrix = calculate_confusion_matrix(state, set)
    println("Confusion matrix sum", sum(confusion_matrix))
    h = heatmap(xlabel="Predicted label", ylabel="True label", cbar = true,
                title="Confusion Matrix ($set_text set)", cbartitle = "Probability", yflip=true)
    heatmap!(h, confusion_matrix, color = :bluesreds)
    savefig(h, imagename)
end
