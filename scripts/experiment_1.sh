#!/bin/bash

echo ""
echo "Small test"
echo ""

python . --patches --dataset mnist --netname 7x200_best.pth --num_tests 1 --patch_size 2 --template_layers 1 --template_method l_infinity --template_domain box_cut
