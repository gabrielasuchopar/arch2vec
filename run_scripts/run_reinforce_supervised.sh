#!/usr/bin/env bash

for s in {1..500}
	do
		python ../arch2vec/search_methods/supervised_reinforce.py --dim 16 --seed $s --bs 16 --output_path saved_logs/rl
    break
	done
