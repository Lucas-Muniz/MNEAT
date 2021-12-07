#! /bin/sh

nohup julia MLP_dt3.jl &> nohup-MLP-nets-dt3-12-09-2021-t1.out &

nohup julia MLP_dt3_IP_bin_tests.jl &> nohup-MLP-bins-dt3-12-09-2021.out &
