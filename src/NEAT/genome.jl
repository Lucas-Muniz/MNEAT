abstract type Node end
abstract type Layer end

mutable struct NodeGene <: Node
    #= A node gene encodes the basic artificial neuron model.
    nodetype should be "INPUT", "HIDDEN", or "OUTPUT" =#
    id::Int64
    ntype::Symbol
    bias::Float64
    activation::Symbol
    ninputs::Int64
    weights::Vector{Float64}
    enable_bit::Bool
    function NodeGene(id::Int64, nodetype::Symbol, ninputs::Int64, bias::Float64=0.0,
                      activation::Symbol=:sigm)
        @assert  activation in [:none, :sigm, :tanh, :relu, :identity]
        weights = randn(ninputs)
        new(id, nodetype, bias, activation, ninputs, weights, true)
    end
    function NodeGene(id::Int64, nodetype::Symbol, weights::Vector{Float64},
                      bias::Float64=0.0, activation::Symbol=:sigm)
        @assert  activation in [:none, :sigm, :tanh, :relu, :identity]
        new(id, nodetype, bias, activation, length(weights), weights, true)
    end
end

function Base.show(io::IO, ng::NodeGene)
    @printf(io, "Node %2d -> type=%6s, bias=%+2.10s, inputs=%+2.10s, %s()",
            ng.id, ng.ntype, ng.bias, ng.ninputs, ng.activation)
end

get_weights(ng::NodeGene, enabled_weights::Vector{Bool}) = ng.weights[enabled_weights]
get_bias(ng::NodeGene) = ng.bias
is_same_innov(n1::NodeGene, n2::NodeGene) = n1.id == n2.id

function get_child(ng::NodeGene, other::NodeGene, fittest_parent::Int64)
    # Creates a new NodeGene ramdonly inheriting its attributes from parents
    assert(ng.id == other.id)
    if fittest_parent == 1
        ng = NodeGene(ng.id, ng.ntype, ng.ninputs, ng.bias, ng.activation)
    elseif fittest_parent == 2
        ng = NodeGene(other.id, other.ntype, other.ninputs, other.bias, other.activation)
    else
        ng = NodeGene(ng.id, ng.ntype,
                      randbool() ? ng.bias : other.bias,
                      randbool() ? ng.ninputs : other.ninputs,
                      ng.activation)
    end
    return ng
end

function mutate_enable_bit!(ng::NodeGene)
    ng.enable_bit = xor(ng.enable_bit, true)
end

function mutate_bias!(ng::NodeGene, cf::Config)
    ng.bias += randn() * cf.bias_mutation_power
    #check_weight_value!(ng.bias, cf)
    return
end

function mutate_weights!(ng::NodeGene, cf::Config)
    ng.weights .+= randn() * cf.weight_mutation_power
    #check_weight_value!.(ng.weights, [cf])
    return
end

function check_weight_value!(weight::Float64, cf::Config)
    if weight > cf.max_weight
        weight = cf.max_weight
    elseif weight < cf.min_weight
        weight = cf.min_weight
    end
end

function mutate_node!(ng::NodeGene, cf::Config)
    if rand() < cf.prob_mutatebias mutate_bias!(ng, cf) end
    if rand() < cf.prob_mutate_weight mutate_weights!(ng, cf) end
end

get_new_innov_number!(g::Global) =  g.innov_number += 1
get_layer_id!(g::Global) = g.layerCnt += 1

mutable struct MLP_layer <: Layer
    ntype::Symbol
    position::Int64
    activation::Symbol
    nodes_number::Int64
    nodes::Vector{Node}
    layer_id::Int64
    function MLP_layer(g::Global, ntype::Symbol, position::Int64, activation::Symbol, ninputs::Int64,
                       nodes_number::Int64)
        @assert position >= 1
        nodes = Vector{Node}(undef, nodes_number)
        for i = 1:nodes_number
            nodes[i] = NodeGene(get_new_innov_number!(g), ntype, ninputs, 0.0, activation)
        end
        new(ntype, position, activation, nodes_number, nodes, get_layer_id!(g))
    end
    function MLP_layer(g::Global, ntype::Symbol, position::Int64, activation::Symbol, nodes::Vector{<:Node})
        @assert position >= 1
        new(ntype, position, activation, length(nodes), nodes, get_layer_id!(g))
    end
end
#get_layer_id!(g)
function get_layer_params(l::MLP_layer, previous_enabled_nodes::Vector{Bool}, concatenate::Bool=false)
    number_enabled_nodes = count((n) -> n.enable_bit, l.nodes)
    (get_weight_matrix(l, previous_enabled_nodes, concatenate=concatenate)..., l.activation, number_enabled_nodes)
end



get_enabled_nodes(l::MLP_layer) = filter((n) -> n.enable_bit, l.nodes)
get_not_enabled_nodes(l::MLP_layer) = filter((n) -> !(n.enable_bit), l.nodes)

maxInnov_layer(l::MLP_layer) = map((n) -> n.id , l.nodes) |> maximum

function get_layer_dictionary(layer::MLP_layer)
    dict_keys = [n.id for n in layer.nodes]
    dict_values = map((p) -> (layer.position, p), 1:length(layer.nodes))
    zip(dict_keys, dict_values) |> Dict
end

function get_enabled_bit_position(layer::MLP_layer)
    positions = findall((n) -> n.enable_bit, layer.nodes)
    return rand(positions)
end

function mutate_node_weight!(g::Global, layer::MLP_layer)
    node_pos = get_enabled_bit_position(layer)
    mutate_node!(layer.nodes[node_pos], g.cf)
end

function get_weight_matrix(l::MLP_layer, enabled_weights::Vector{Bool}; concatenate::Bool=false)
    enabled_nodes = get_enabled_nodes(l)
    weights = get_weights.(enabled_nodes, [enabled_weights])
    biases = get_bias.(enabled_nodes)
    weights_mtx = vcat(map((a) -> a', weights)...)
    if !concatenate
        return (weights_mtx, biases)
    else
        return hcat(weights_mtx, biases)
    end
end

function add_node_weight!(l::MLP_layer)
    for node in l.nodes
        push!(node.weights, randn())
        node.ninputs += 1
    end
end

function remove_node_weight!(l::MLP_layer, pos::Int=-1)
    for node in l.nodes
        pos == -1 ? deleteat!(node.weights, length(node.weights)) : deleteat!(node.weights, pos)
        node.ninputs -= 1
    end
end


function swapcols!(X::AbstractMatrix, i::Integer, j::Integer)
    @inbounds for k = 1:size(X,1)
        X[k,i], X[k,j] = X[k,j], X[k,i]
    end
end
