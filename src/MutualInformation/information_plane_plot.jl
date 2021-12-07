IP_back_legends_en = Dict("title" => "Information Plane (Backpropagation)", "cbartitle" => "Epochs")
IP_back_legends_pt = Dict("title" => "Plano de informação (Backpropagation)", "cbartitle" => "Épocas")

IP_legends_en = Dict("title" => "Information Plane", "cbartitle" => "Epochs")
IP_legends_pt = Dict("title" => "Plano de informação", "cbartitle" => "Épocas")

function plot_IP(model::MLPClassifier, input, output; imagename::String = "IP.png", nbins::Int = 10,  language::Symbol=:en)
    legends = language == :pt ? IP_legends_pt : IP_legends_en
    gr()
    positions = evaluate_information_plane(model, input, output, nbins)
    p = plot(positions, color=:purple, marker = (:circle, 6, 0.8, Plots.stroke(1, :gray)), legend=false,
             xlabel = L"I(X,Z)", ylabel = L"I(Z,Y)", title = legends["title"])
    savefig(p, imagename)
end

function get_net_sets(nfeatures::Int, labels::Vector{<:Union{Int, Float64}},
        hidden_layers::Vector{Int}, nets_number::Int, get_train_data, path)
    net_sets = map(1:nets_number) do i
        net = MLPClassifier(nfeatures, labels, hidden_layers)
        input_train, output_train = get_train_data(path, shuffle=true)
        net, input_train, output_train
    end
    net_sets
end

function plot_IP_average(nfeatures::Int, labels::Vector{<:Union{Int, Float64}},
        hidden_layers::Vector{Int}, nets_number::Int, get_train_data, path, epochs::Int,
        step_len::Int, imagename::String = "IP-average.png", nbins::Int=10; language::Symbol=:en)
    net_sets = get_net_sets(nfeatures, labels, hidden_layers, nets_number, get_train_data, path)
    generate_IP_average_plot(net_sets, nets_number, epochs, step_len, nbins, imagename, language=language)
end

function get_net_sets(nfeatures::Int, labels::Vector{<:Union{Int, Float64}},
        hidden_layers::Vector{Int}, nets_number::Int, data)
    net_sets = map(1:nets_number) do i
        net = MLPClassifier(nfeatures, labels, hidden_layers)
        train_data = shuffle(collect(data))
        (input_train, output_train) = getindex.(train_data, 1), getindex.(train_data, 2)
        net, input_train, output_train
    end
    net_sets
end

function plot_IP_average(nfeatures::Int, labels::Vector{<:Union{Int, Float64}},
        hidden_layers::Vector{Int}, nets_number::Int, data, epochs::Int,
        step_len::Int, imagename::String = "IP-average.png", nbins::Int=10; language::Symbol=:en)
    net_sets = get_net_sets(nfeatures, labels, hidden_layers, nets_number, data)
    generate_IP_average_plot(net_sets, nets_number, epochs, step_len, nbins, imagename, language=language)
end

function get_marker_vector(size::Int)
    if size == 0
        return []
    elseif size == 1
        return [:star6]
    elseif size >= 2
        markers = [:circle for i in 1:size]
        markers[end] = :star6
        return markers
    end
end

function generate_IP_average_plot(net_sets, nets_number::Int, epochs::Int, step_len::Int,
                                  nbins::Int=10, imagename::String = "IP-average.png";
                                   language::Symbol=:en)
    legends = language == :pt ? IP_back_legends_pt : IP_back_legends_en
    nets = getindex.(net_sets, 1)
    inputs = getindex.(net_sets, 2)
    outputs = getindex.(net_sets, 3)
    gr()
    #step_len = 10
    for step = 0:step_len:epochs
        map((n, in, out) -> fit!(n, in, out; max_epochs=step_len), nets, inputs, outputs)
        IP_points = map((n, in, out) -> evaluate_information_plane(n, in, out, nbins), nets, inputs, outputs)
        layer_points = collect.(hcat(IP_points...))'
        average_points = ((1/nets_number)*ones(Float64, (1,nets_number)))*layer_points |> adjoint
        x, y = separate_columns(average_points[:,1])
        z_colors = [cgrad(:inferno, [0.01, 0.99])[z] for z ∈ range(0.0, 1.0, length = Int((epochs/step_len)+1))]
        color_index = Int((step/step_len)+1)
        markers = get_marker_vector(length(x))
        if step == 0 #step_len
            plot(x, y, line_z = step, #=marker = (:circle, 6, 0.8, :inferno, Plots.stroke(1, :gray)),=# cbar = true, legend=false,
                 xlabel = L"I(X,Z)", ylabel = L"I(Z,Y)", title = legends["title"], cbartitle = legends["cbartitle"])
            Plots.scatter!(x, y, color = z_colors[color_index], m = markers, markersize = 7)
        else
            plot!(x, y, line_z = step)
            Plots.scatter!(x, y, color = z_colors[color_index], m = markers, markersize = 7)
            #Plots.scatter!(x[1:end-1], y[1:end-1], color = z_colors[color_index], markersize = 7)
            #Plots.scatter!([x[end]], [y[end]], color = z_colors[color_index], m = :hexagon, markersize = 7)
        end
    end
    savefig(imagename)
end

function calculate_IP_average_points(nfeatures::Int, labels::Vector{<:Union{Int, Float64}},
        hidden_layers::Vector{Int}, nets_number::Int, get_train_data, path, epochs::Int,
        step_len::Int, nbins::Int=10)
    net_sets = get_net_sets(nfeatures, labels, hidden_layers, nets_number, get_train_data, path)
    IP_average_points(net_sets, nets_number, epochs, step_len, nbins)
end

function calculate_IP_average_points(nfeatures::Int, labels::Vector{<:Union{Int, Float64}},
        hidden_layers::Vector{Int}, nets_number::Int, data, epochs::Int,
        step_len::Int, nbins::Int=10)
    net_sets = get_net_sets(nfeatures, labels, hidden_layers, nets_number, data)
    IP_average_points(net_sets, nets_number, epochs, step_len, nbins)
end

function IP_average_points(net_sets, nets_number::Int, epochs::Int, step_len::Int, nbins::Int=10)
    nets = getindex.(net_sets, 1)
    inputs = getindex.(net_sets, 2)
    outputs = getindex.(net_sets, 3)
    points = []
    #step_len = 10
    for step = 0:step_len:epochs
        println("Step: ", step)
        map((n, in, out) -> fit!(n, in, out; max_epochs=step_len), nets, inputs, outputs)
        IP_points = map((n, in, out) -> evaluate_information_plane(n, in, out, nbins), nets, inputs, outputs)
        layer_points = collect.(hcat(IP_points...))'
        average_points = ((1/nets_number)*ones(Float64, (1,nets_number)))*layer_points |> adjoint
        push!(points, average_points)
    end
    points
end

function generate_plot_IP_average(points, nets_number::Int, epochs::Int, step_len::Int,
                                  imagename::String = "IP-average.png"; language::Symbol=:en,
                                  xlim=nothing, ylim=nothing)
    legends = language == :pt ? IP_back_legends_pt : IP_back_legends_en
    gr()
    local p
    z_colors = [cgrad(:inferno, [0.01, 0.99])[z] for z ∈ range(0.0, 1.0, length = Int((epochs/step_len)+1))]
    for step = 0:step_len:epochs
        #step_index = Int(step/step_len)
        color_index = Int((step/step_len)+1)
        average_points = points[color_index]
        x, y = separate_columns(average_points[:,1])
        markers = get_marker_vector(length(x))
        if step == 0
            p = plot(x, y, line_z = step, #=marker = (:circle, 6, 0.8, :inferno, Plots.stroke(1, :gray)),=# cbar = true, legend=false,
                 xlabel = L"I(X,Z)", ylabel = L"I(Z,Y)", title = legends["title"], cbartitle = legends["cbartitle"])
            if xlim != nothing plot!(p, xlim=xlim) end
            if ylim != nothing plot!(p, ylim=ylim) end
            Plots.scatter!(p, x, y, color = z_colors[color_index],  m = markers, markersize = 7)

        else
            plot!(p, x, y, line_z = step)
            Plots.scatter!(p, x, y, color = z_colors[color_index], m = markers, markersize = 7)
        end
    end
    savefig(p, imagename)
end



function calculate_MI_output(model::MLPClassifier, input, Ŷ, epochs::Int, step::Int=1)
    epochs_array = 0:step:epochs
    num_epochs = length(epochs_array)
    MI_array = Vector{Float64}(undef, num_epochs)
    for epoch = 1:num_epochs
        if epoch > 0 fit!(model, input, Ŷ; max_epochs=step) end
        model_output = model.net.(input) |> flat_vector
        X_flatten = input |> flat_vector
        Ŷ_flatten =  convert_to_onehot(Ŷ, model.labels) |> flat_vector
        MI_array[epoch] = mutual_information(calculate_markov_chain_probability(Ŷ_flatten, X_flatten, model_output)...)
    end
    (epochs_array, MI_array)
end

function calculate_MI_hiddenlayer(model::MLPClassifier, input, output, hidden_pos::Int, epochs::Int, step::Int=1)
    epochs_array = 0:step:epochs
    num_epochs = length(epochs_array)
    MI_array = Vector{Float64}(undef, num_epochs)
    for epoch = 1:num_epochs
        if epoch > 0 fit!(model, input, output; max_epochs=step) end
        subnet = model.net[1:layer]
        subnet_output = subnet.(input) |> flat_vector
        X_flatten = input |> flat_vector
        MI_array[epoch] = mutual_information([X_flatten], [subnet_output])
    end
    (epochs_array, MI_array)
end

function calculate_MI_hiddenlayer(model::MLPClassifier, input, output, epochs::Int, step::Int=1)
    epochs_array = 0:step:epochs
    layers = number_hidden_layers(model)
    num_epochs = length(epochs_array)
    MI_array = zeros(Float64, (layers, num_epochs))
    for epoch = 1:num_epochs
        if epoch > 0 fit!(model, input, output; max_epochs=step) end
        for layer = 1:layers
            subnet = model.net[1:layer+1]
            subnet_output = subnet.(input) |> flat_vector
            X_flatten = input |> flat_vector
            MI_array[layer, epoch] = mutual_information([X_flatten], [subnet_output])
        end
    end
    (epochs_array, [MI_array[l,:] for l in 1:size(MI_array,1)]...,)
end

function plot_MI_hiddenlayers(model::MLPClassifier, input, output, epochs::Int=1, step::Int=1; imagename::String = "MI-hiddenlayers.png")
    gr()
    data = calculate_MI_hiddenlayer(model, input, output, epochs, step)
    p = plot(xlabel = "Epochs", ylabel = L"I(X,Z)", title = L"Epochs \times I(X,Z)",  legend=true)
    colors = [cgrad(:inferno, [0.01, 0.99])[z] for z ∈ range(0.0, 1.0, length = max(length(data)-1, 2))]
    for layer = 1:length(data)-1
        plot!(p, data[1], data[layer+1], color = colors[layer], marker = (6, 0.8, Plots.stroke(1, :gray)), label="I(X,Z$layer)")
    end
    savefig(imagename)
end

function plot_MI_output(model::MLPClassifier, input, output, epochs::Int=1, step::Int=1; imagename::String = "MI-output.png")
    gr()
    data = calculate_MI_output(model, input, output, epochs, step)
    p = plot(ylims = (0,1), xlabel = "Epochs", ylabel = L"I(X,Z)", title = "Epochs x I(Y,Ŷ)",  legend=true)
    p = plot!(p, data[1], data[2], color = :green, marker = (6, 0.8, Plots.stroke(1, :gray)),  label=L"I(Y,Ŷ)")
    savefig(p, imagename)
end

using Statistics

function plot_MLP_error_rate(nfeatures::Int, labels::Vector{<:Union{Int, Float64}},
        hidden_layers::Vector{Int}, nets_number::Int, train_data, test_data, epochs::Int,
        step_len::Int, imagename::String = "MLP-error_rate.png")

    net_sets = get_net_sets(nfeatures, labels, hidden_layers, nets_number, train_data)


    nets = getindex.(net_sets, 1)
    inputs = getindex.(net_sets, 2)
    outputs = getindex.(net_sets, 3)
    gr()

    steps = 0:step_len:epochs

    mean_train_error_rate = Vector{Float64}(undef, length(steps))
    mean_test_error_rate = Vector{Float64}(undef, length(steps))
    #step_len = 10

    i =  1
    for step = steps
        if step != 0 map((n, in, out) -> fit!(n, in, out; max_epochs=step_len), nets, inputs, outputs) end
        mean_train_error_rate[i] = map(n -> error_rate(train_data, n), nets) |> mean
        mean_test_error_rate[i] = map(n -> error_rate(test_data, n), nets) |> mean
        i += 1
    end

    println("Taxa erro final média (treinamento): ", mean_train_error_rate[end])
    println("Taxa erro final média (test): ", mean_test_error_rate[end])

    p = plot(title="Taxa de erro", xlabel="Época", ylabel="Taxa de erro")
    plot!(p, steps, mean_train_error_rate, label="Conjunto de treinamento", color=:green, line = 2)
    plot!(p, steps, mean_test_error_rate, label="Conjunto de teste", color=:blue, line = 2)
    savefig(p, imagename)
end

function plot_IP_network(nfeatures::Int, labels::Vector{<:Union{Int, Float64}},
        hidden_layers::Vector{Int}, data, epochs::Int, imagename::String = "IP-MLP-network.png",
        nbins::Int=10; language::Symbol=:en)

    net_sets = get_net_sets(nfeatures, labels, hidden_layers, 1, data)


    legends = language == :pt ? IP_legends_pt : IP_legends_en
    net = getindex.(net_sets, 1)
    input = getindex.(net_sets, 2)
    output = getindex.(net_sets, 3)
    gr()
    #step_len = 10

    if epochs > 0
        fit!(net[1], input[1], output[1]; max_epochs=epochs)
    end
    IP_points = evaluate_information_plane(net[1], input[1], output[1], nbins)
    layer_points = collect.(hcat(IP_points...))'
    #average_points = (ones(Float64, (1,1)))*layer_points |> adjoint
    x, y = getindex.(layer_points,1), getindex.(layer_points,2)

    color = cgrad(:inferno, [0.01, 0.99])[0.1]
    markers = get_marker_vector(length(x))

    plot(x, y, legend=false, color = color,
         xlabel = L"I(X,Z)", ylabel = L"I(Z,Y)", title = legends["title"])
    Plots.scatter!(x, y, color = color, m = markers, markersize = 7)

    savefig(imagename)
end
