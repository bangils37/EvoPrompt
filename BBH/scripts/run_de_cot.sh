#!/bin/bash

set -ex


BUDGET=1
POPSIZE=10
template=v1
llm=gemini
initial=cot
initial_mode=para_topk

# ⚡ Move to project root (EvoPrompt)
cd "$(dirname "$(realpath "$0")")/../.."

for task in logical_deduction_seven_objects
do
for SIZE in 10
do
POPSIZE=$SIZE
OUT_PATH=outputs/$task/$initial/de/bd${BUDGET}_top${POPSIZE}_${initial_mode}_init/$template/$llm
for seed in 10
do
mkdir -p $OUT_PATH/seed${seed}
cache_path=cache/$task/seed$seed
mkdir -p $cache_path
python -m BBH.run \
    --seed $seed \
    --task $task \
    --batch-size 20 \
    --sample_num 50 \
    --budget $BUDGET \
    --popsize $POPSIZE \
    --evo_mode de \
    --llm_type $llm \
    --setting default \
    --initial $initial \
    --initial_mode $initial_mode \
    --template $template \
    --cot_cache_path $cache_path/prompts_cot_$llm.json \
    --output $OUT_PATH/seed${seed}
done
done
done