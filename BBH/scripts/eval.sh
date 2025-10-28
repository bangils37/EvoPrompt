#!/bin/bash

set -ex

llm=gemini

# ⚡ Move to project root (EvoPrompt)
cd "$(dirname "$(realpath "$0")")/../.."

for task in logical_deduction_seven_objects 
do
OUT_PATH=outputs/$task/eval/$llm/3-shot
for seed in 10
do
mkdir -p $OUT_PATH/seed${seed}
python -m BBH.eval \
    --seed $seed \
    --task $task \
    --batch-size 20 \
    --sample_num 50 \
    --llm_type $llm \
    --setting default \
    --demon 1 \
    --output $OUT_PATH/seed${seed} \
    --content "Let's think step by step."
done
done