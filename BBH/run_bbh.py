# Generic BBH evaluator (supports Gemini via llm_client)

import json
import numpy as np
from tqdm import tqdm
from BBH.utils import extract_ans, batchify
from BBH.llm_client import llm_query

MULTIPLE_CHOICE_TASKS = [
        'temporal_sequences', 'disambiguation_qa', 'date_understanding', 'tracking_shuffled_objects_three_objects', 'penguins_in_a_table', 
        'geometric_shapes', 'snarks', 'ruin_names', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_five_objects', 
        'logical_deduction_three_objects', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'movie_recommendation', 
        'salient_translation_error_detection', 'reasoning_about_colored_objects', 
]
FREE_FORM_TASKS = [
        'multistep_arithmetic_two', 'navigate', 'dyck_languages', 'word_sorting', 'sports_understanding', 
        'boolean_expressions', 'object_counting', 'formal_fallacies', 'causal_judgement', 'web_of_lies', 
]


def create_dataset(mode, task_prompt, cot_prompt, eval_data, demon=1, few_shot_examples=None):
    questions = []
    prompt_qs = []
    answers= []
    
    # Build few-shot examples string if provided
    few_shot_str = ""
    if few_shot_examples and len(few_shot_examples) > 0:
        few_shot_parts = []
        for ex in few_shot_examples:
            ex_input = ex['input']
            ex_target = ex['target']
            if mode == 'multiple_choice':
                ex_answer = ex_target[1]  # Extract letter from "(A)"
            else:
                ex_answer = ex_target
            few_shot_parts.append(f"{ex_input}\nAnswer is ({ex_answer})" if mode == 'multiple_choice' else f"{ex_input}\nAnswer is {ex_answer}")
        few_shot_str = "Example:\n" + "\n\n".join(few_shot_parts) + "\n\nProblem:\n"
    
    for q_ in eval_data:
        # Replace <prompt> with cot_prompt
        formatted_task_prompt = task_prompt.replace('<prompt>', cot_prompt)
        
        if demon: 
            q = '\n\nQ: ' + q_['input']
            # Insert few-shot examples after instruction, before the current question
            prompt_q = formatted_task_prompt + "\n" + few_shot_str + q + f"\nA: {cot_prompt}"
        else:
            q = 'Q: ' + q_['input']
            prompt_q = few_shot_str + q + f"\nA: {cot_prompt}"
        
        questions.append(q)
        prompt_qs.append(prompt_q)
        if mode == 'multiple_choice':
            a = q_['target'][1]
        elif mode == 'free_form':
            a = q_['target']
        answers.append(a)
    return prompt_qs, questions, answers


def eval_task(task, task_prompt, cot_prompt, eval_data, client, model_index, logger, demon, few_shot_examples=None, **kwargs):
    # for task in tasks:
    # print('Testing %s ...' % task)
    correct = 0
    mode = 'multiple_choice' if task in MULTIPLE_CHOICE_TASKS else 'free_form'
    print_first = True
    
    # Auto-load few-shot examples if not provided
    if few_shot_examples is None:
        try:
            task_data_file = f"BBH/data/{task}.json"
            with open(task_data_file, 'r') as f:
                task_data = json.load(f)
                few_shot_examples = task_data['examples'][:3]
        except Exception as e:
            logger.warning(f"Could not load few-shot examples for task {task}: {e}")
            few_shot_examples = []
    
    prompt_qs, questions, answers = create_dataset(mode, task_prompt, cot_prompt, eval_data, demon, few_shot_examples)
    # Unified batching: send requests in batches (llm_query supports list inputs)
    batched_prompt_qa = batchify(prompt_qs)
    responses = []
    for batch in tqdm(batched_prompt_qa):
        if print_first:
            logger.info('First prompt: ')
            logger.info(batch[0])
            print_first = False
        resp = llm_query(batch, client, model_index, task, temperature=0, **kwargs)
        # llm_query should return a list for list inputs; handle fallback
        if isinstance(resp, list):
            responses.extend(resp)
        else:
            responses.extend([resp] * len(batch))

    for ans, q, a in zip(responses, questions, answers):
        ans_ = extract_ans(ans, mode)
        if ans_ == a:
            correct += 1
    accuracy = correct / len(eval_data)
    print('%s acc %.4f' % (task, correct / len(eval_data)))
    return accuracy

