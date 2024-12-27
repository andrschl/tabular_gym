#!/bin/sh
wind_levels=("0.01" "0.1" "0.5" "1.0")
for seed in {0..9}; do
  for level in ${wind_levels[@]}; do
    python get_windy_experts.py --wind_level $level --seed $seed
  done
done