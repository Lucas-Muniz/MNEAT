#! /bin/sh

nohup julia neat_test_dt1.jl test_dataset1_config_t1.txt 15000 &> nohup-neat-dt1-not-limited-20-07-2021.out &

nohup julia neat_test_dt1.jl test_dataset1_config_t2.txt 15000 &> nohup-neat-dt1-limited-20-07-2021.out &
