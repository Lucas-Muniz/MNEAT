#using Tracker
using Random, Flux
using MLDataUtils: kfolds


struct MLPClassifier
    net::Flux.Chain
    labels::Vector{<:Union{Int, Float64}}
    nfeatures::Int
    function MLPClassifier(number_features::Int, labels::Vector{<:Union{Int, Float64}},
                           hidden_layers::Vector{Int}=Int[])
        model = build_classifier_model(number_features, length(labels), hidden_layers, tanh)
        new(model, labels, number_features)
    end
    function MLPClassifier(number_features::Int, labels::Vector{<:Union{Int, Float64}},
                           weights, activations, hidden_layers::Vector{Int}=Int[])
        model = build_classifier_model(number_features, length(labels),  weights,
                                       activations, hidden_layers)
        new(model, labels, number_features)
    end
end

function build_classifier_model(n_inputs::Int, n_outputs::Int, hidden_layers::Vector{Int}=Int[],
                                activation::Function=identity)::Flux.Chain
    layer_sizes = [n_inputs; hidden_layers; n_outputs]
    nlayers = length(layer_sizes)
    layers = map(2:nlayers) do i
        size_below = layer_sizes[i-1]
        size_above = layer_sizes[i]
        if i != nlayers
            Dense(size_below, size_above, activation)
        else
            Dense(size_below, size_above)
        end
    end
    Chain(layers..., softmax)
end

function build_classifier_model(n_inputs::Int, n_outputs::Int,  weights, activations,
                                hidden_layers::Vector{Int}=Int[])::Flux.Chain
    layer_sizes = [n_inputs; hidden_layers; n_outputs]
    nlayers = length(layer_sizes)
    layers = map(2:nlayers) do i
        (w, b) = weights[i-1]
        act = activations[i-1]
        size_below = layer_sizes[i-1]
        size_above = layer_sizes[i]
        Dense(size_below, size_above, get_activation(act); initW = (out, in) -> w, initb = out -> b)
    end
    Chain(layers..., softmax)
end

function get_activation(f::Symbol)
    act_functions = [:tanh, :sigmoid, :σ, :relu, :identity]
    if any(map((s) -> f == s, act_functions))
        return eval(f)
    else
        return identity
    end
end

(m::MLPClassifier)(input::Vector{<:Real}) = m.net(input)

function calculate_topology(model::MLPClassifier)
    net = model.net
    number_layers = length(net)
    topology = Vector{Int}(undef, number_layers)
    output_layer = 0
    for l = 1:(number_layers-1)
        output, input = size(net[l].W)
        topology[l] = input
        if l == (number_layers-1) output_layer = output end
    end
    topology[number_layers] = output_layer
    return topology
end

number_hidden_layers(m::MLPClassifier) = length(m.net) - 2

function fit!(model::MLPClassifier, inputs, outputs ; max_epochs=1)
    function lossMSE(x, y)
        ŷ = model.net(x)
        #Flux.mse(ŷ, Flux.onehot(y[1], model.labels))
        Flux.mse(ŷ, y)
    end

    function loss(x, y)
        ŷ = model.net(x)
        Flux.crossentropy(ŷ, y)
    end

    function loss_all(data, model)
        l = 0f0
        for (x,y) in data
            l += loss(x, y)
        end
        l/length(data)
    end

    #cb = function ()
    #    accuracy() > 0.9 && Flux.stop()
    #end

    η = 0.2
    opt = Descent(η)
    params = Flux.params(model.net)
    new_outputs = convert_to_onehot(outputs, model.labels)
    #println("Outputs:")
    #println(new_outputs)
    dataset = zip(inputs, new_outputs) # (shuffle ∘ collect ∘ zip)

    for ep = 1:max_epochs
        Flux.train!(loss, params, (shuffle ∘ collect)(dataset), opt,  cb = Flux.throttle(() -> print(""), 1800))
    end
    #Flux.@epochs max_epochs Flux.train!(loss, params, (shuffle ∘ collect)(dataset), opt,  cb = Flux.throttle(() -> println("training"), 10))

    model
end

function early_stop_fit!(model::MLPClassifier, inputs, outputs ; max_epochs::Int=1, k_folds::Int=10)
    function lossMSE(x, y)
        ŷ = model.net(x)
        Flux.mse(ŷ, y)
    end

    function loss(x, y)
        ŷ = model.net(x)
        Flux.crossentropy(ŷ, y)
    end

    η = 0.2
    opt = Descent(η)
    #params = Flux.params(model.net)
    new_outputs = convert_to_onehot(outputs, model.labels)
    dataset = zip(inputs, new_outputs)

    # Early-stopping training using K-fold cross validation
    folds = kfolds(dataset, k = k_folds)
    record_loss_n_train = []
    record_loss_n_valid = []
    fold_select = 1
    early_stop = 0
    evaluate = (f, d) -> sum(map((s) -> f(s...), d))
    for epoch_idx in 1:max_epochs

        train, valid = folds[fold_select] # selection of the datasets

        evalcb = () -> (push!(record_loss_n_train, evaluate(loss, train)),
        push!(record_loss_n_valid, evaluate(loss, valid)))

        Flux.train!(loss, params(model), train, opt, cb = throttle(evalcb, 1))

        fold_select += 1 # for selecting the K-fold between 1 and the total number of folds
        if fold_select >= (k_folds+1)
            fold_select = 1
        end

        # for early stop
        if record_loss_n_valid[epoch_idx] > record_loss_n_valid[epoch_idx-1]
            early_stop += 1
        end
        if early_stop > 100
            break
        end
    end

    model
end

function k_fold_crossvalidation(model::MLPClassifier, inputs, outputs ; k_folds::Int=10)
    η = 0.2
    opt = Descent(η)

    function loss(m, x, y)
        ŷ = m.net(x)
        Flux.crossentropy(ŷ, y)
    end

    new_outputs = convert_to_onehot(outputs, model.labels)
    dataset = zip(inputs, new_outputs)
    folds = kfolds(dataset, k = k_folds)
    record_loss_n_valid = Float64[]
    for fold = 1:k_folds
        train, valid = folds[fold]
        m = deepcopy(model)
        evaluate = (f, d) -> sum(map((s) -> f(m, s...), d))
        Flux.train!(loss, params(m), train, opt, cb = throttle(evalcb, 1))
        push!(record_loss_n_valid, evaluate(loss, valid))
    end
    sum(record_loss_n_valid)/k_folds
end


function convert_to_onehot(output, labels)
    return [1*(Flux.onehot(elem[1], labels)) for elem in output]
    #=for i = 1:length(output)
        push!(new_output, Flux.onehot(output[i][1], labels))
    end
    new_output=#
end

function accuracy(data, model::MLPClassifier)
    acc = 0
    for (x,y) in data
        acc += sum(Flux.onecold(model.net(x)) .== Flux.onecold(Flux.onehot(y[1], model.labels)))*1 / size(x,2)
    end
    acc/length(data)
end

function accuracy(data, model::Chain, labels)
    acc = 0
    for (x,y) in data
        acc += sum(Flux.onecold(model(x)) .== Flux.onecold(Flux.onehot(y[1], labels)))*1 / size(x,2)
    end
    acc/length(data)
end

function error_rate(data, model::MLPClassifier)
    acc = 0
    for (x,y) in data
        acc += (predict(model, x) != y[1])
    end
    acc/length(data)
end

function predict(model::MLPClassifier, features)
    r = model.net(features)
    Flux.onecold(r, model.labels)
end

function my_custom_train!(loss, ps, data, opt)
  # training_loss is declared local so it will be available for logging outside the gradient calculation.
  local training_loss
  ps = Flux.Params(ps)
  for d in data
    gs = gradient(ps) do
      training_loss = loss(d...)
      # Code inserted here will be differentiated, unless you need that gradient information
      # it is better to do the work outside this block.
      return training_loss
    end
    # Insert whatever code you want here that nier.jl:88eeds training_loss, e.g. logging.
    # logging_callback(training_loss)
    # Insert what ever code you want here that needs gradient.
    # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
    update!(opt, ps, gs)
    # Here you might like to check validation set accuracy, and break out to do early stopping.
  end
end



#=
model = Chain(Dense(5,10, σ), Dense(10,10, σ), Dense(10, 1))
function loss(x, y)
    x1 = deepcopy(x)
    y1 = deepcopy(y)
    Flux.Losses.mse(model(x1), y1)
end
params = Flux.params(model)
x_train, y_train, x_test, y_test = get_dataset1("../datasets/Drug-Classification")
#println("Acuracy before training: ", accuracy(zip(x_test, y_test), model, dataset1_labels))

η = 0.2
opt = Descent(η)

#new_outputs = convert_to_onehot(y_train, dataset1_labels)
#const x = x_train
#const y = y_train
dataset = zip(x_train, y_train)
#my_custom_train!(loss, params, dataset, opt)
#Flux.train!(loss, params, dataset, opt)
=#
