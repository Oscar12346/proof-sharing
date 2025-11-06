#!/bin/bash

echo ""
echo "Computing results for Experiment 1"
echo ""

mkdir -p results
rep=1
method=l_infinity
net="7x200_best.pth"

for template_domain in box_cut_1 box box_cut_2
do
  for dataset in mnist cifar
  do
    for layer in "0" "1" "2" "3" "4" "5" "6"
    do
      fn=results/experiment1/${template_domain}_patches_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}_${rep}.txt
      if test -f "$fn"; then
        echo "$fn exists; skipping."
      else
        python . -p --netname ${net} --dataset ${dataset} --num_tests 100 --relu_transformer zonotope --patch_size 2 --template_method ${method} --template_domain ${template_domain} --template_layers ${layer} |& tee "$fn"
      fi
    done
  done
done

python scripts/summarize_results.py --table 1 | tee results/experiment1.txt
