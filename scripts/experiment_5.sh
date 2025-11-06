#!/bin/bash

echo ""
echo "Computing results for Experiment 5"
echo ""

mkdir -p results
method=l_infinity
net="7x200_best.pth"
template_domain=box_cut_2


for dataset in cifar
do
  for layer in  "2" "4" "6"
  do
    fn=results/experiment5/${template_domain}_patches_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}.txt
    if test -f "$fn"; then
      echo "$fn exists; skipping."
    else
      python . -p --netname ${net} --dataset ${dataset} --num_tests 100 --relu_transformer zonotope --patch_size 2 --template_method ${method} --template_domain ${template_domain} --template_layers ${layer} |& tee "$fn"
    fi
  done
done


for dataset in mnist
do
  for num_splits in 4 8
  do
    layer="1 2"
    method=l_infinity
    fn=results/experiment5/${template_domain}_geometrics${num_splits}_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}.txt
    if test -f "$fn"; then
      echo "$fn exists; skipping."
    else
      python . -g --netname ${net} --dataset ${dataset} --num_tests 100 --relu_transformer zonotope --data_dir mnist_1_brightness_01_001_proof_transfer${num_splits} --template_method ${method} --template_domain ${template_domain} --template_layers ${layer} |& tee "$fn"
    fi
  done
done


python scripts/summarize_results.py --experiment5 | tee results/experiment5/summary.txt