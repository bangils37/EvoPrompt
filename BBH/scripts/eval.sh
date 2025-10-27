#!/bin/bash

set -ex

llm=gemini

# ⚡ Move to project root (EvoPrompt)
cd "$(dirname "$(realpath "$0")")/../.."

for task in date_understanding multistep_arithmetic_two navigate dyck_languages word_sorting sports_understanding object_counting formal_fallacies causal_judgement web_of_lies boolean_expressions temporal_sequences disambiguation_qa tracking_shuffled_objects_three_objects penguins_in_a_table geometric_shapes snarks ruin_names tracking_shuffled_objects_seven_objects tracking_shuffled_objects_five_objects logical_deduction_three_objects hyperbaton logical_deduction_five_objects logical_deduction_seven_objects movie_recommendation salient_translation_error_detection reasoning_about_colored_objects
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