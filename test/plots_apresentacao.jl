using Plots,  LaTeXStrings

function plot_fitness(step=0.1, imagename="fitness-function.png")
    x = 0:step:1
    fitness(i) = 2/(i+1) - 1

    plot(x, fitness.(x), color = :orange, line = 3, guidefontsize=14, legend = false,
         xlabel = L"t_{et}(i)", ylabel = L"f(i)", title = "Função fitness")

    savefig(imagename)
end

function plot_peaks_surface(step=0.1, imagename="peaks-function.png")
    xs = -3:step:3
    ys = -3:step:3
    x_grid = [x for x = xs for y = ys]
    y_grid = [y for x = xs for y = ys]
    peaks(x, y) = 3*(1-x)^2 * exp(-x^2 -(y+1)^2) -10(x/5 - x^3 - y^5)*exp(-x^2-y^2)-1/3*exp(-(x+1)^2-y^2)

    plot(x_grid, y_grid, -1 * peaks.(x_grid, y_grid), guidefontsize=14, label = false,
        xlabel = L"w_i", ylabel = L"w_j", title = L"E(w_i, w_j)", cbar=false,
        camera=(75,45), st = :surface, color = :bluesreds)

    savefig(imagename)
end
