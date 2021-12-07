include("../neat_test.jl")

st = get_simulation_state("Resultados_23:07:2021/limited/checkpoint-dt1-limited-10000g.jld2")

#nbins = parse(Int, ARGS[1])

#print("Number of bins: ", nbins)
#plot_neat_IP(st, imagename="IP-dt1-$(nbins)bins-limited.png", nbins=nbins)

bins = [5, 6, 9, 10, 15]
# best resut: 15 bins


for b in bins
    println("Number of bins: ", b)
    plot_neat_IP(st, imagename="IP-dt1-$(b)bins-limited-fixed.png", nbins=b)
    println("Image generated")
end

# nohup julia IP_bins_img_test.jl 19 &> nohup-MLP-dt1-IP-19b-30:07:2021.out &

# nohup julia IP_bins_img_test.jl &> nohup-MLP-dt1-IP-fixed-13:08:2021.out &
