using Plots

function plot_dataset_distribution(dataset_setup::Dataset; id::String="",
                                   imagename::String = "dataset-distribution.png")
    plot_distribution(dataset_setup.output_train, dataset_setup.labels, id = id, imagename = imagename)
end

function plot_dataset_distribution(dataset_setup::SeparatedDataset; set::Symbol=:training, id::String="",
                                   imagename::String = "dataset-distribution.png")
    if set == :training
        plot_distribution(dataset_setup.output_train, dataset_setup.labels, id = string(id, " (training)"), imagename = imagename)
    elseif set == :test
        plot_distribution(dataset_setup.output_test, dataset_setup.labels, id = string(id, " (test)"), imagename = imagename)
    end
end


function plot_distribution(outputs::Vector{Vector{T}}, labels::Vector{T}; id::String="", 
                          imagename::String = "dataset-distribution.png") where T <: Number
    distribution = countmap(getindex.(outputs,1))
    frequencies = getindex.([distribution], labels)
    title = id == "" ? "Dataset distribution" : "Dataset $id distribution"
    dist = plot(labels, frequencies, xlabel = "Class", xticks = 0:1:length(labels), linetype=[:bar],
              ylabel = "Number of samples", title = title, legend=false, color=:purple)
    savefig(dist, imagename)
end
