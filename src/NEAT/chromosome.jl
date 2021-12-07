abstract type ChromoType end
abstract type Chromosome end
abstract type FeedForward <: ChromoType end

mutable struct MLPChromosome <: Chromosome
    id::Int64
    input_size::Int64
    output_size::Int64
    n_hidden_layers::Int64
    layers::Vector{Layer}
    fitness::Float64
    species_id::Int64
    parent1_id::Int64
    parent2_id::Int64
    genes_mapping::Dict{Int64, Tuple{Int64, Int64}}
    function MLPChromosome(g::Global)
        @assert g.cf.initial_hidden_nodes >= 1
        ntype = :MLP
        layers = Layer[MLP_layer(g, :HIDDEN, 1, g.cf.nn_activation, g.cf.input_nodes, g.cf.initial_hidden_nodes),
                       MLP_layer(g, :OUTPUT, 2, :identity, g.cf.initial_hidden_nodes, g.cf.output_nodes)]
        genes_mapping = merge(Dict([]), get_layer_dictionary(layers[1]))
        genes_mapping = merge!(genes_mapping, get_layer_dictionary(layers[2]))
        new(incChromeId!(g),
            g.cf.input_nodes, g.cf.output_nodes,
            1,
            layers, # array of layers
            0.0, # stub for fitness function
            0, # species_id
            # parents ids: they help in tracking chromosome's genealogy
            0, 0,
            genes_mapping
            )
    end
    function MLPChromosome(g::Global, parent1::MLPChromosome, parent2::MLPChromosome)
        @assert parent1.species_id == parent2.species_id
        new(incChromeId!(g),
            g.cf.input_nodes, g.cf.output_nodes,
            1,
            Layer[], # array of layers
            0.0, # stub for fitness function
            parent1.species_id, # species_id
            # parents ids: they help in tracking chromosome's genealogy
            parent1.id, parent2.id,
            Dict([])
            )
    end
end

function Base.show(io::IO, ch::MLPChromosome)
    println(io, "id: $(ch.id)")
    println(io, "nº hidden layers: $(ch.n_hidden_layers)")
    nlayers = length(ch.layers)
    println(io, "Layer 0 (INPUT) -> nodes: $(ch.input_size)")
    topology_str = "$(ch.input_size)"
    for l = 1:nlayers
        layer = ch.layers[l]
        ninputs = layer.nodes[1].ninputs
        nnodes = count((n) -> n.enable_bit, layer.nodes)
        l == nlayers ? print(io, "Layer $l (OUTPUT) -> ") : print(io, "Layer $l -> ")
        print(io, "id: $(layer.layer_id), position: $(layer.position), ")
        print(io, "nodes: $(layer.nodes_number)")
        #print(", weights: $(nnodes)x$(ninputs)")
        topology_str = string(topology_str, " -> ", nnodes)
        #if l != nlayers  println() end
        print(io, "\n")
    end
    print(io, "Topology: $topology_str")
end

function get_parameters(ch::MLPChromosome)
    number_layers = length(ch.layers)
    layers_params = Vector{Tuple{Matrix{Float64}, Vector{Float64}, Symbol, Int64}}(undef, number_layers)
    previous_enabled_nodes = ones(Bool, ch.layers[1].nodes[1].ninputs)
    for l = 1:number_layers
        #=if l == 1
            enabled_nodes = ones(Bool, ch.layers[l].nodes[1].ninputs)
            layers_params[l] = get_layer_params(ch.layers[l], enabled_nodes)
        else

        end=#
        layers_params[l] = get_layer_params(ch.layers[l], previous_enabled_nodes)
        previous_enabled_nodes = map((n) -> n.enable_bit, ch.layers[l].nodes)
    end
    #[get_layer_params(l) for l in ch.layers]
    layers_params
end

get_layers(ch::MLPChromosome) = ch.layers

function incChromeId!(g::Global)
    g.chromosomeCnt += 1
    return g.chromosomeCnt
end

create_minimal_chromossome(g::Global)::MLPChromosome = MLPChromosome(g)

function number_nodes(ch::MLPChromosome)
    if length(ch.layers) > 0
        return sum(map((l) -> length(l.nodes), ch.layers))
    end
    return 0
end

function maxInnov(ch::MLPChromosome)
    if length(ch.layers) > 0
        return map((l) -> maxInnov_layer(l), ch.layers) |> maximum
    end
    error("There is no layers in the chromossome.")
end

function mutate_add_node!(ch::MLPChromosome, g::Global)
    layer_pos = rand(1:ch.n_hidden_layers)
    number_inputs = ch.layers[layer_pos].nodes[1].ninputs
    new_node = NodeGene(get_new_innov_number!(g), :HIDDEN, number_inputs, randn(), g.cf.nn_activation)
    push!(ch.layers[layer_pos].nodes, new_node)
    add_node_weight!(ch.layers[layer_pos+1])
    ch.layers[layer_pos].nodes_number += 1
    merge!(ch.genes_mapping, Dict(new_node.id => (layer_pos, ch.layers[layer_pos].nodes_number)))
    return
end

function mutate_enable_bit!(ch::MLPChromosome)
    layer_pos = rand(1:ch.n_hidden_layers)
    l = ch.layers[layer_pos]
    nodes = l.nodes
    len = length(nodes)
    if len > 1
        enabled_nodes = get_enabled_nodes(l)
        len_enabled = length(enabled_nodes)
        if len_enabled <= 1 && len > len_enabled
            not_enabled = get_not_enabled_nodes(l)
            node_index = rand(1:length(not_enabled))
            mutate_enable_bit!(not_enabled[node_index])
        elseif len_enabled > 1
            node_index = rand(1:len)
            mutate_enable_bit!(nodes[node_index])
        end
    end
end

function mutate_remove_node!(ch::MLPChromosome)
    layer_pos = rand(1:ch.n_hidden_layers)
    nodes = ch.layers[layer_pos].nodes
    len = length(nodes)
    if  len > 1
        node_pos = rand(1:len)
        delete!(ch.genes_mapping, nodes[node_pos].id)
        deleteat!(nodes, node_pos)
        remove_node_weight!(ch.layers[layer_pos+1], node_pos)
        ch.layers[layer_pos].nodes_number -= 1
        merge!(ch.genes_mapping, get_layer_dictionary(ch.layers[layer_pos]))
    end
    return
end

function mutant_layer(g::Global, position::Int, ninputs::Int, nodes_number::Int, activation::Symbol)
    nodes = Vector{Node}(undef, nodes_number)
    for i = 1:nodes_number
        #=weights = zeros(Float64, ninputs)
        if i <= ninputs
            weights[i] = 1.0
        end=#
        weights = randn(Float64, ninputs)
        nodes[i] = NodeGene(get_new_innov_number!(g), :HIDDEN, weights, 0.0, activation)
    end
    MLP_layer(g, :HIDDEN, position, activation, nodes)
end

function mutate_add_layer!(ch::MLPChromosome, g::Global)
    #number_layers = length(get_layers(ch))
    number_layers = ch.n_hidden_layers
    new_layer_pos = rand(1:number_layers)
    number_nodes_nl = (Int ∘ floor)((ch.layers[new_layer_pos].nodes_number + ch.layers[new_layer_pos+1].nodes_number)/2)
    #new_layer = MLP_layer(g, :MLP, new_layer_pos+1, :tanh, ch.layers[new_layer_pos].nodes_number, number_nodes_nl)
    new_layer = mutant_layer(g, new_layer_pos+1, ch.layers[new_layer_pos].nodes_number, number_nodes_nl, g.cf.nn_activation)

    for n in ch.layers[new_layer_pos+1].nodes
        old_ninputs = n.ninputs
        n.ninputs = number_nodes_nl
        if number_nodes_nl <= old_ninputs
            n.weights = n.weights[1:number_nodes_nl]
        elseif number_nodes_nl > old_ninputs
            old_weights = n.weights
            new_weights = zeros(Float64, number_nodes_nl)
            new_weights[1:old_ninputs] = old_weights
            n.weights = new_weights
        end
    end
    insert!(ch.layers, new_layer_pos+1, new_layer)
    for i = (new_layer_pos+1):length(ch.layers)
        ch.layers[i].position = i
        merge!(ch.genes_mapping, get_layer_dictionary(ch.layers[i]))
    end
    ch.n_hidden_layers += 1
    return
end

function mutate_weights!(ch::MLPChromosome, g::Global)
    for l in ch.layers
        mutate_node_weight!(g, l)
    end
    return
end

function mutate!(ch::MLPChromosome, g::Global)
    # Mutates the chromosome
    if rand() < g.cf.prob_structural_mutation
        if rand() < g.cf.prob_addnode
            mutate_add_node!(ch, g)
            #println("Add node mutation")
        #=elseif rand() < g.cf.prob_removenode
            mutate_remove_node!(ch)
            #println("Remove node mutation")
        =#
        elseif rand() < g.cf.prob_addlayer
            mutate_add_layer!(ch, g)
            #println("Add layer mutation")
        end

    elseif rand() < g.cf.prob_enable_mutation
        mutate_enable_bit!(ch) # mutate node's enable bit
    else
        mutate_weights!(ch, g) # mutate node's weights and bias
    end
    return ch
end

function convert_to_FluxNet(ch::MLPChromosome, labels::Vector{Float64})::MLPClassifier
    params = get_parameters(ch)
    weights = getindex.(params, 1)
    biases = getindex.(params, 2)
    activations = getindex.(params, 3)
    number_nodes_layers = getindex.(params, 4)
    MLPClassifier(ch.input_size, labels, collect(zip(weights,biases)), activations, number_nodes_layers[1:end-1])
end

function get_node_from_dict(ch::MLPChromosome, key::Int)
    layer, node_pos = ch.genes_mapping[key]
    ch.layers[layer].nodes[node_pos]
end

function nodes_innov_numbers(ch::MLPChromosome)
    innov_numbers = Int[]
    for l in ch.layers
        append!(innov_numbers, map((n) -> n.id, l.nodes))
    end
    innov_numbers
end


# compatibility function
function distance(self::MLPChromosome, other::MLPChromosome, cf::Config)

    # Returns the distance between this chromosome and the other.
    chromo1, chromo2 = number_nodes(self) > number_nodes(other) ? (self,other) : (other,self)

    weight_diff = 0
    matching = 0
    disjoint = 0
    excess = 0

    #biggest_genome = max(length(chromo1.genes_mapping), length(chromo2.genes_mapping))

    max_innov_chromo2 = maxInnov(chromo2)
    max_innov_chromo1 = maxInnov(chromo1)

    max_id = -1

    for (k, g1) in chromo1.genes_mapping
        node1 = get_node_from_dict(chromo1, k)
        if haskey(chromo2.genes_mapping, k)
            # Homologous genes
            node2 = get_node_from_dict(chromo2, k)

            max_id = max(max_id, k)

            node1, node2 = length(node1.weights) > length(node2.weights) ? (node1, node2) : (node2, node1)
            max_len, min_len = length(node1.weights), length(node2.weights)
            for i = 1:max_len
                if i <= min_len
                    weight_diff += abs(node1.weights[i] - node2.weights[i])
                else
                    weight_diff += abs(node1.weights[i])
                end
            end
            matching += 1
        end
    end

    all_innov_numbers = union(nodes_innov_numbers(chromo1), nodes_innov_numbers(chromo2))

    disjoint = sum(map((n) -> n <= max_id, all_innov_numbers)) - matching
    excess = sum(map((n) -> n > max_id, all_innov_numbers))

    d = (cf.excess_coeficient * excess + cf.disjoint_coeficient * disjoint)#/biggest_genome
    #println("match: $matching, exc: $excess, disj: $disjoint e weight diff: $weight_diff.")

    return matching > 0 ? d + cf.weight_coeficient * weight_diff / matching : d
end

#=
function crossover(g::Global, self::MLPChromosome, other::MLPChromosome)
    # Crosses over parents' chromosomes and returns a child

    # This can't happen! Parents must belong to the same species.
    @assert self.species_id == other.species_id

    # If they're of equal fitnesses, choose the shortest
    parent1, parent2 = self.fitness >= other.fitness ? (self,other) : (other,self)
    if parent1.fitness == parent2.fitness
        parent1, parent2 = number_nodes(parent1) >= number_nodes(parent2) ?
                           (parent1, parent2) : (parent2, parent1)
    end

    # create a new child
    #child = MLPChromosome(incChromeId!(g), other.id, self.node_gene_type, self.conn_gene_type)
    child = deepcopy(parent1)
    inherit_genes!(g, child, parent1, parent2)
    child.species_id = parent1.species_id
    child.fitness = 0 #parent1.fitness
    #child._input_nodes = parent1._input_nodes

    return child
end
=#

function crossover(g::Global, self::MLPChromosome, other::MLPChromosome)
    # Crosses over parents' chromosomes and returns a child

    # This can't happen! Parents must belong to the same species.
    @assert self.species_id == other.species_id

    local child
    if self.fitness == other.fitness
        child = generate_and_inherit_genes(g, self, other)
    else
        parent1, parent2 = self.fitness > other.fitness ? (self,other) : (other,self)
        child = MLPChromosome(g, parent1, parent2)
        child.layers = deepcopy(parent1.layers)
        child.n_hidden_layers = length(parent1.layers)-1
        child.genes_mapping = deepcopy(parent1.genes_mapping)
        inherit_genes!(child, g, parent1, parent2)
    end
    child.fitness = 0

    return child
end

function inherit_genes!(child::MLPChromosome, g::Global, parent1::MLPChromosome, parent2::MLPChromosome)
    # Applies the crossover operator.
    @assert parent1.fitness >= parent2.fitness

    # Crossover node genes
    for (key, add) in parent1.genes_mapping
        node1 = get_node_from_dict(parent1, key)
        layer, node_pos = add
        if haskey(parent2.genes_mapping, key)
            node2 = get_node_from_dict(parent2, key)
            gene, parent_number = is_same_innov(node1, node2) ? get_child(g, node1, node2) : (deepcopy(node1), 1)
            child.layers[layer].nodes[node_pos] = gene
        else # Copy excess or disjoint genes from the fittest parent
            child.layers[layer].nodes[node_pos] = deepcopy(node1)
        end
    end
end

function generate_and_inherit_genes(g::Global, parent1::MLPChromosome, parent2::MLPChromosome)
    child = MLPChromosome(g, parent1, parent2)
    layer_nodes, parent_mapping = get_nodes_from_parents(g, parent1, parent2)
    layers = build_layers(g, layer_nodes, parent_mapping, parent1, parent2)
    child.layers = layers
    child.n_hidden_layers = length(layers)-1

    for l in child.layers
        merge!(child.genes_mapping, get_layer_dictionary(l))
    end

    child
end

function get_nodes_from_parents(g::Global, parent1::MLPChromosome, parent2::MLPChromosome)
    parent_mapping = Dict{Int, Int}([])
    p1_number_layers, p2_number_layers = length(parent1.layers), length(parent2.layers)
    number_layers = max(p1_number_layers, p2_number_layers)
    layer_nodes = [NodeGene[] for i=1:number_layers]
    # Crossover node genes
    for (key, add) in parent1.genes_mapping
        is_output_layer = false
        node1 = get_node_from_dict(parent1, key)
        layer_position, node_pos = add
        is_output_layer = layer_position == p1_number_layers
        node_gene = node1
        parent_number = 1
        if haskey(parent2.genes_mapping, key)
            node2 = get_node_from_dict(parent2, key)
            if is_same_innov(node1, node2)
                node_gene, parent_number = get_child(g, node1, node2)
            end
            layer_position_node2, _ = parent2.genes_mapping[key]
            is_output_layer = layer_position_node2 == p2_number_layers
        end
        if is_output_layer
            push!(layer_nodes[end], deepcopy(node_gene))
        else
            push!(layer_nodes[layer_position], deepcopy(node_gene))
        end
        parent_mapping[key] = parent_number
    end
    parent_number = 2
    for (key, add) in parent2.genes_mapping
        if !haskey(parent1.genes_mapping, key)
            layer_position, node_pos = add
            node_gene = get_node_from_dict(parent2, key)
            push!(layer_nodes[layer_position], deepcopy(node_gene))
            parent_mapping[key] = parent_number
        end
    end

    #=for l = 2:length(layer_nodes)
        length_previous_layer = length(layer_nodes[l-1])
        for n in layer_nodes[l] n.ninputs = length_previous_layer  end
    end=#

    sort!(layer_nodes[end], by = x -> x.id)
    layer_nodes, parent_mapping
end

function build_layers(g::Global, layers_nodes::Vector{Vector{NodeGene}}, parent_mapping::Dict, parent1::MLPChromosome, parent2::MLPChromosome)
    number_layers = length(layers_nodes)
    for current_layer = 2:number_layers
        number_nodes_last_layer = length(layers_nodes[current_layer-1])
        for node in layers_nodes[current_layer]
            node.ninputs = number_nodes_last_layer
            node.weights = randn(number_nodes_last_layer)
            parent_number = parent_mapping[node.id]
            node_mapping = parent_number == 1 ? parent1.genes_mapping : parent2.genes_mapping
            parent = parent_number == 1 ? parent1 : parent2
            layer_position, node_pos = node_mapping[node.id]
            last_layer_nodes = filter((n) -> parent_mapping[n[2].id] == parent_number, collect(enumerate(layers_nodes[current_layer-1])))
            for (weight_pos, last_layer_node) in last_layer_nodes
                _ , last_layer_node_pos = node_mapping[last_layer_node.id]
                node.weights[weight_pos] = parent.layers[layer_position].nodes[node_pos].weights[last_layer_node_pos]
            end
        end
    end
    layers = [MLP_layer(g, :HIDDEN, i, g.cf.nn_activation, layers_nodes[i]) for i = 1:(number_layers-1)]
    push!(layers, MLP_layer(g, :OUTPUT, number_layers, :identity, layers_nodes[number_layers]))
    layers
end

function get_child(g::Global, n1::NodeGene, n2::NodeGene)
    weights = deepcopy(n1.weights)
    inputs = min(n1.ninputs, n2.ninputs)
    parent = rand((1:2))
    bias = 0
    act = :sigm
    if parent == 1
        weights[1:inputs] = n1.weights[1:inputs]
        bias = n1.bias
        act = n1.activation
    else
        weights[1:inputs] = n2.weights[1:inputs]
        bias = n2.bias
        act = n2.activation
    end
    NodeGene(n1.id, :HIDDEN, weights, bias, act), parent
end

function chromosome_complexity(ch::MLPChromosome)
    number_weights = 0
    number_nodes = ch.input_size
    for l in ch.layers
        layer_nodes = count((n) -> n.enable_bit, l.nodes)
        number_weights += number_nodes*layer_nodes # + layer_nodes # bias weights
        number_nodes = layer_nodes
    end
    number_weights
end
