#! /bin/sh

nohup julia MLP_dt2.jl &> nohup-MLP-nets-dt1-12-09-2021.out &

nohup julia MLP_dt2_IP_bin_tests.jl &> nohup-MLP-bins-dt1-12-09-2021.out &
