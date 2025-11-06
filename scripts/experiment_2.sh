#!/bin/bash

echo ""
echo "Computing results for Experiment 2"
echo ""

mkdir -p results/experiment2
net="7x200_best.pth"

for dataset in mnist cifar; do
  for rep in 1 2 3; do

    # Baseline (no templates)
    method=base
    layer=base
    fn=results/experiment2/patches_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}_${rep}.txt
    if test -f "$fn"; then
      echo "$fn exists; skipping."
    else
      python . -p --netname ${net} --dataset ${dataset} --num_tests 100 \
        --relu_transformer zonotope --patch_size 2 \
        |& tee "$fn"
    fi

    # Box vs Box-Cut on the SAME seed
    method=l_infinity
    for template_domain in box box_cut_1 box_cut_2; do
      for layer in "0" "1" "2" "3" "0 2" "1 2" "1 3" "1 2 3"; do
        fn=results/experiment2/${template_domain}_patches_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}_${rep}.txt
        if test -f "$fn"; then
          echo "$fn exists; skipping."
        else
          python . -p --netname ${net} --dataset ${dataset} --num_tests 100 \
            --relu_transformer zonotope --patch_size 2 \
            --template_method ${method} --template_domain ${template_domain} \
            --template_layers ${layer} \
            |& tee "$fn"
        fi
      done
    done
  done
done

python scripts/summarize_results.py --table 2 | tee results/experiment2/summary.txt
