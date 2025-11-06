#!/bin/bash

echo ""
echo "Computing results for Experiment 3"
echo ""

mkdir -p results/experiment3


net="7x200_best.pth"
dataset="mnist"

for rep in "1" "2" "3"
do
	for num_splits in 4 6 8 10
	do
		layer="base"
		method="base"
		fn=results/experiment3/geometrics${num_splits}_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}_${rep}.txt
		if test -f "$fn"; then
			echo "$fn exists; skipping."
		else
			python . -g --netname ${net} --dataset ${dataset} --num_tests 100 --relu_transformer zonotope --data_dir mnist_1_brightness_01_001_proof_transfer${num_splits} |& tee "$fn"
		fi

		for template_domain in box box_cut box_cut_2
		do
		  layer="1 2"
      method=l_infinity
      fn=results/experiment3/${template_domain}_geometrics${num_splits}_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}_${rep}.txt
      if test -f "$fn"; then
        echo "$fn exists; skipping."
      else
        python . -g --netname ${net} --dataset ${dataset} --num_tests 100 --relu_transformer zonotope --data_dir mnist_1_brightness_01_001_proof_transfer${num_splits} --template_method ${method} --template_domain ${template_domain} --template_layers ${layer} |& tee "$fn"
      fi
    done
	done
done

python scripts/summarize_results.py --table 3 | tee results/experiment3/summary.txt
