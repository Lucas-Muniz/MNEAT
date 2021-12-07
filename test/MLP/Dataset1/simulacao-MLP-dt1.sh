#! /bin/sh

nohup julia MLP_dt1.jl &> nohup-MLP-nets-dt1-12-09-2021.out &

nohup julia MLP_dt1_IP_bin_tests.jl &> nohup-MLP-bins-dt1-29-07-2021.out &
