#!/bin/sh
wind_levels=("0.01" "0.1" "0.5" "1.0")
Ns=("1000" "10000" "100000" "1000000")
for seed in {0..9}; do
  for level in ${wind_levels[@]}; do
    for N in ${Ns[@]}; do
      python multi_expert_irl.py --wind_level $level --seed $seed --N $N
    done
  done
done