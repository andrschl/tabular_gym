#!/bin/sh
wind_levels=("0.01" "0.1" "0.5" "1.0")
for level in ${wind_levels[@]}; do
  python check_transferability.py --wind_level $level
done
