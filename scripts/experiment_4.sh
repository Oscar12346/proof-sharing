#!/bin/bash

echo ""
echo "Computing results for Experiment 4"
echo ""

mkdir -p results
method=l_infinity
net="7x200_best.pth"

layer=2
for template_domain in box box_cut_1
do
  for dataset in mnist cifar
  do
    fn=results/experiment4/${template_domain}_patches_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}.txt
    if test -f "$fn"; then
      echo "$fn exists; skipping."
    else
      python . -p --netname ${net} --dataset ${dataset} --num_tests 100 --relu_transformer zonotope --patch_size 2 --template_method ${method} --template_domain ${template_domain} --template_layers ${layer} |& tee "$fn"
    fi
  done
done

layer=4
for template_domain in box box_cut_1
do
  for dataset in mnist cifar
  do
    fn=results/experiment4/${template_domain}_patches_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}.txt
    if test -f "$fn"; then
      echo "$fn exists; skipping."
    else
      python . -p --netname ${net} --dataset ${dataset} --num_tests 100 --relu_transformer zonotope --patch_size 2 --template_method ${method} --template_domain ${template_domain} --template_layers ${layer} |& tee "$fn"
    fi
  done
done

layer=6
for template_domain in box box_cut_1
do
  for dataset in mnist cifar
  do
    fn=results/experiment4/${template_domain}_patches_${dataset}_${net//_best.pth/}_${layer// /+}_${method//_/}.txt
    if test -f "$fn"; then
      echo "$fn exists; skipping."
    else
      python . -p --netname ${net} --dataset ${dataset} --num_tests 100 --relu_transformer zonotope --patch_size 2 --template_method ${method} --template_domain ${template_domain} --template_layers ${layer} |& tee "$fn"
    fi
  done
done

python scripts/summarize_results.py --table 4 | tee results/experiment4/summary.txt

