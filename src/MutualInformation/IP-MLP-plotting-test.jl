include("information-plane.jl")

epochs = 10000
number_nets = 50
step_len = 1
#=
x_train, y_train, x_test, y_test = get_datasetDemo("../datasets/Demo", normalise=true)
data = zip(x_train, y_train)
plot_IP_average(datasetDemo_nfeatures, datasetDemo_labels, [2], number_nets, data, epochs, step_len, "IP-dtDemo-MLP.png")

x_train, y_train, x_test, y_test = get_datasetDemoOverlap("../datasets/Demo-overlap", normalise=true)
data = zip(x_train, y_train)
plot_IP_average(datasetOverlap_nfeatures, datasetOverlap_labels, [2], number_nets, data, epochs, step_len, "IP-dtOverlap-MLP.png")

x_train, y_train, x_test, y_test = get_dataset1("../datasets/Drug-Classification", normalise=true)
data = zip(x_train, y_train)
plot_IP_average(dataset1_nfeatures, dataset1_labels, [5], number_nets, data, epochs, step_len, "IP-dt1-MLP.png")

x_train, y_train, x_test, y_test = get_dataset2("../datasets/Income-Classification", normalise=true)
data = zip(x_train, y_train)
plot_IP_average(dataset2_nfeatures, dataset2_labels, [26,13,6], number_nets, data, epochs, step_len, "IP-dt2-MLP.png")
=#
x_train, y_train, x_test, y_test = get_dataset3("../datasets/Wine-Classification", normalise=true)
data = zip(x_train, y_train)
plot_IP_average(dataset3_nfeatures, dataset3_labels, [22,10,15,11], number_nets, data, epochs, step_len, "IP-dt3-MLP.png")

x_train, y_train, x_test, y_test = get_dataset4("../datasets/Drug-Consumism-Classification", normalise=true)
data = zip(x_train, y_train)
plot_IP_average(dataset4_nfeatures, dataset4_labels, [24,6,12,7], number_nets, data, epochs, step_len, "IP-dt4-MLP.png")

x_train, y_train, x_test, y_test = get_dataset5("../datasets/Heart-Disease-Classification", normalise=true)
data = zip(x_train, y_train)
plot_IP_average(dataset5_nfeatures, dataset5_labels,  [26,13,7,5], number_nets, data, epochs, step_len, "IP-dt5-MLP.png")
