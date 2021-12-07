using StatsBase, Distributions, IterTools #StatsPlots

function get_freq(h::Histogram, val)
    x = searchsortedfirst.(h.edges, val)
    h.weights[x...]
end

function get_bin(h::Histogram, val)
    bin = searchsortedfirst.(h.edges, val)
    bin[1] - 1
end

function get_probability(h::Histogram, val)
    x = searchsortedfirst.(h.edges, val)
    h.weights[x...]/sum(h.weights)
end

function mutual_information(X::Vector{Vector{T}}, Y::Vector{Vector{T}}) where T <: Number
    hx = fit(Histogram, (X...,))
    hy = fit(Histogram, (Y...,))
    XY = vcat(X,Y)
    hxy = fit(Histogram, (XY...,))
    MI = 0.0
    rx = create_matrix_range(size(hx.weights))
    ry = create_matrix_range(size(hy.weights))
    for x in Iterators.product(rx...)
        p_x = get_bin_probability(hx, x...)
        for y in Iterators.product(ry...)
            p_y = get_bin_probability(hy, y...)
            index_xy = (x..., y...)
            p_xy = get_bin_probability(hxy, index_xy...)
            if p_xy != 0
                MI += p_xy*log(p_xy/(p_x*p_y))
            end
        end
    end
    MI
end

function mutual_information(p_x, p_y, p_xy)
    MI = 0.0
    for x = 1:length(p_x)
        for y = 1:length(p_y)
            prob_xy = p_xy[x, y]
            prob_log = prob_xy/(p_x[x]*p_y[y])
            if prob_xy != 0 && prob_log != NaN
                MI += prob_xy*log(prob_log)
            end
        end
    end
    MI
end

function mutual_information_unit(X::Vector{T}, Y::Vector{T}) where T <: Int
    max_X, max_Y = maximum(X), maximum(Y)
    hx = fit(Histogram, (X), 0:max_X, closed=:right)
    hy = fit(Histogram, (Y), 0:max_Y, closed=:right)
    XY = hcat([X],[Y])
    hxy = fit(Histogram, (X, Y), (0:max_X, 0:max_Y), closed=:right)
    MI = 0.0
    rx = create_matrix_range(size(hx.weights))
    ry = create_matrix_range(size(hy.weights))
    for x in Iterators.product(rx...)
        p_x = get_bin_probability(hx, x...)
        for y in Iterators.product(ry...)
            p_y = get_bin_probability(hy, y...)
            index_xy = (x..., y...)
            p_xy = get_bin_probability(hxy, index_xy...)
            if p_xy != 0
                MI += p_xy*log(p_xy/(p_x*p_y))
            end
        end
    end
    MI
end

function mutual_information_unit(X::Vector{T}, Y::Vector{T}, lim_X, lim_Y) where T <: Int
    min_X, max_X = lim_X
    min_Y, max_Y = lim_Y
    range_X = min_X:max_X
    range_Y = min_Y:max_Y
    hx = fit(Histogram, (X), range_X, closed=:right)
    hy = fit(Histogram, (Y), range_Y, closed=:right)
    XY = hcat([X],[Y])
    hxy = fit(Histogram, (X, Y), (range_X, range_Y), closed=:right)
    MI = 0.0
    rx = create_matrix_range(size(hx.weights))
    ry = create_matrix_range(size(hy.weights))
    for x in Iterators.product(rx...)
        p_x = get_bin_probability(hx, x...)
        for y in Iterators.product(ry...)
            p_y = get_bin_probability(hy, y...)
            index_xy = (x..., y...)
            p_xy = get_bin_probability(hxy, index_xy...)
            if p_xy != 0
                MI += p_xy*log(p_xy/(p_x*p_y))
            end
        end
    end
    MI
end

get_bin_probability(h::Histogram, x...) = h.weights[x...]/sum(h.weights)

function create_matrix_range(t::Tuple)
    tuple_length = length(t)
    ranges = []
    for r = 1:tuple_length
        range = 1:t[r]
        push!(ranges, range)
    end
    ranges
end

function calculate_joint_probability(X::Vector{<:Number}, Y::Vector{<:Number}, lim_X, lim_Y)
    min_X, max_X = lim_X
    min_Y, max_Y = lim_Y
    range_X = min_X:max_X
    range_Y = min_Y:max_Y
    p_x = fit(Histogram, (X), range_X, closed=:right)
    p_y = fit(Histogram, (Y), range_Y, closed=:right)
    p_xy = fit(Histogram, (X, Y), (range_X, range_Y), closed=:right)
    p_X = p_x.weights/sum(p_x.weights)
    p_Y = p_y.weights/sum(p_y.weights)
    p_XY = p_xy.weights/sum(p_xy.weights)
    return p_X, p_Y, p_XY
end

function calculate_joint_probability(X::Vector{<:Number}, Y::Vector{<:Number}, nbins)
    p_x = fit(Histogram, (X), nbins=nbins[1])
    p_y = fit(Histogram, (Y), nbins=nbins[2])
    p_xy = fit(Histogram, (X, Y), nbins=nbins)
    p_X = p_x.weights/sum(p_x.weights)
    p_Y = p_y.weights/sum(p_y.weights)
    p_XY = p_xy.weights/sum(p_xy.weights)
    return p_X, p_Y, p_XY
end


include("layer_histogram.jl")

function calculate_joint_probability(X::Vector{<:Number}, Y::Vector{<:Number}, limits_X, limits_Y, nbins)
    p_x = create_bin_histogram(X, nbins[1], limits_X[1], limits_X[2])
    p_y = create_bin_histogram(Y, nbins[2], limits_Y[1], limits_Y[2])
    p_xy = fit(Histogram, (X, Y), (p_x.edges[1], p_y.edges[1]))
    p_X = p_x.weights/sum(p_x.weights)
    p_Y = p_y.weights/sum(p_y.weights)
    p_XY = p_xy.weights/sum(p_xy.weights)
    return p_X, p_Y, p_XY
end



# Test
#=
X = randn(10000)
Y = randn(10000)

println("Mutual Information: I(X; Y) = ", mutual_information([X], [Y]))
println("Mutual Information: I(Y; X) = ", mutual_information([Y], [X]))
println("Mutual Information: I(X; (Y,X)) = ", mutual_information([X], [Y, X]))
println("Mutual Information: I(X; X) = ", mutual_information([X], [X]))
println("Mutual Information: I(Y; Y) = ", mutual_information([Y], [Y]))
=#
