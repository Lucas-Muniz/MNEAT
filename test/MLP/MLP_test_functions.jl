function show_results(dt, net)
    println("** Test set")
    println("Accuracy: ", accuracy(zip(dt.input_test, dt.output_test), net))
    println("Error rate: ", error_rate(zip(dt.input_test, dt.output_test), net))
    println("** Training set")
    println("Accuracy: ", accuracy(zip(dt.input_train, dt.output_train), net))
    println("Error rate: ", error_rate(zip(dt.input_train, dt.output_train), net))
end

function MLP_error_rate(nfeatures::Int, labels::Vector{<:Union{Int, Float64}},
        hidden_layers::Vector{Int}, nets_number::Int, train_data, test_data, epochs::Int)

    net_sets = get_net_sets(nfeatures, labels, hidden_layers, nets_number, train_data)
    nets = getindex.(net_sets, 1)
    inputs = getindex.(net_sets, 2)
    outputs = getindex.(net_sets, 3)

    map((n, in, out) -> fit!(n, in, out; max_epochs=epochs), nets, inputs, outputs)

    train_error_rate =  map(n -> error_rate(train_data, n), nets) |> mean
    test_error_rate = map(n -> error_rate(test_data, n), nets) |> mean

    println("Taxa erro final média (treinamento): ", train_error_rate)
    println("Taxa erro final média (test): ", test_error_rate)
end
