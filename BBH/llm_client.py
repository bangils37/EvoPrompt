import json
import os
import atexit
import requests
import sys
from tqdm import tqdm
import openai
import backoff
from termcolor import colored
import time
from BBH.utils import read_yaml_file, batchify

# Google Gemini
try:
    import google.generativeai as genai
except ImportError:
    genai = None


def extract_seconds(text, retried=5):
    words = text.split()
    for i, word in enumerate(words):
        if "second" in word:
            return int(words[i - 1])
    return 60


def form_request(data, type, **kwargs):
    if "davinci" in type:
        request_data = {
            "prompt": data,
            "max_tokens": 1000,
            "top_p": 1,
            "n": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stream": False,
            "logprobs": None,
            "stop": None,
            **kwargs,
        }
    else:
        assert isinstance(data, str)
        messages_list = [
            {"role": "system", "content": "Follow the given examples and answer the question."},
            {"role": "user", "content": data},
        ]
        request_data = {"messages": messages_list, **kwargs}
    return request_data


current_dir = os.path.dirname(os.path.abspath(__file__))
def llm_init(auth_file=os.path.join(current_dir, '../auth.yaml'), llm_type='davinci', setting='default'):
    auth = read_yaml_file(auth_file)[llm_type][setting]
    api_type = auth.get("api_type", "openai")

    if api_type == "gemini":
        if genai is None:
            raise ImportError("Please install google-generativeai (pip install google-generativeai)")
        genai.configure(api_key=auth["api_key"])
    else:
        try:
            openai.api_type = auth.get("api_type", "open_ai")
            openai.api_base = auth.get("api_base", None)
            openai.api_version = auth.get("api_version", None)
        except Exception:
            pass
        openai.api_key = auth["api_key"]

    return auth


def turbo_query(request_data, **kwargs):
    while True:
        retried = 0
        try:
            response = openai.ChatCompletion.create(
                messages=[
                    {"role": "system", "content": "Follow the given examples and answer the question."},
                    {"role": "user", "content": request_data},
                ],
                **kwargs)
            break
        except Exception as e:
            error = str(e)
            print("retring...", error)
            second = extract_seconds(error, retried)
            retried += 1
            time.sleep(second)

    return response['choices'][0]['message']['content']


def davinci_query(data, client, **kwargs):
    retried = 0
    request_data = {"prompt": data, "max_tokens": 1000, "temperature": 0, **kwargs}
    while True:
        try:
            response = openai.Completion.create(**request_data)
            response = [r["text"] for r in response["choices"]]
            break
        except Exception as e:
            error = str(e)
            print("retring...", error)
            second = extract_seconds(error, retried)
            retried += 1
            time.sleep(second)
    return response


def llm_query(data, client, type, task, **config):
    """
    Generic LLM query — supports OpenAI / Azure / Gemini
    """
    hypos = []
    api_type = config.get("api_type", "openai")

    # Gemini branch
    if api_type == "gemini":
        model_name = config.get("model", "gemini-2.0-flash")
        model = genai.GenerativeModel(model_name)
        if isinstance(data, list):
            for d in tqdm(data):
                try:
                    resp = model.generate_content(d)
                    hypos.append(resp.text.strip())
                except Exception as e:
                    print("Gemini error:", e)
                    time.sleep(5)
                    hypos.append("")
        else:
            try:
                resp = model.generate_content(data)
                hypos = resp.text.strip()
            except Exception as e:
                print("Gemini error:", e)
                hypos = ""
        return hypos

    # OpenAI / Azure branch
    model_name = "davinci" if "davinci" in type else "turbo"
    if isinstance(data, list):
        batch_data = batchify(data, 20)
        for batch in tqdm(batch_data):
            retried = 0
            request_data = form_request(batch, model_name, **config)
            if "davinci" in type:
                while True:
                    try:
                        response = openai.Completion.create(**request_data)
                        response = [r["text"] for r in response["choices"]]
                        break
                    except Exception as e:
                        print("retring...", e)
                        second = extract_seconds(str(e), retried)
                        retried += 1
                        time.sleep(second)
            else:
                response = []
                for d in batch:
                    request_data = form_request(d, type, **config)
                    while True:
                        try:
                            result = openai.ChatCompletion.create(**request_data)
                            result = result["choices"][0]["message"]["content"]
                            response.append(result)
                            break
                        except Exception as e:
                            print("retring...", e)
                            second = extract_seconds(str(e), retried)
                            retried += 1
                            time.sleep(second)

            results = [str(r).strip().split("\n\n")[0] if task else str(r).strip() for r in response]
            hypos.extend(results)
    else:
        retried = 0
        while True:
            try:
                if "turbo" in type or "gpt4" in type:
                    request_data = form_request(data, type, **config)
                    response = openai.ChatCompletion.create(**request_data)
                    result = response["choices"][0]["message"]["content"]
                    break
                else:
                    request_data = form_request(data, type=type, **config)
                    response = openai.Completion.create(**request_data)["choices"][0]["text"]
                    result = response.strip()
                break
            except Exception as e:
                print("retring...", e)
                second = extract_seconds(str(e), retried)
                retried += 1
                time.sleep(second)
        if task:
            result = result.split("\n\n")[0]
        hypos = result

    return hypos


def paraphrase(sentence, client, type, **kwargs):
    if isinstance(sentence, list):
        resample_template = [
            f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput:{s}\nOutput:"
            for s in sentence
        ]
    else:
        resample_template = f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput:{sentence}\nOutput:"
    print(resample_template)
    results = llm_query(resample_template, client, type, False, **kwargs)
    return results


if __name__ == "__main__":
    llm_client = None
    llm_type = 'gemini'  # or 'gpt4o'
    start = time.time()

    data = [
        """Q: Tom bought a skateboard for $9.46, and spent $9.56 on marbles. Tom also spent $14.50 on shorts. In total, how much did Tom spend on toys? A: Let's think step by step."""
    ]

    config = llm_init(auth_file="auth.yaml", llm_type=llm_type, setting="default")
    para = paraphrase(data, llm_client, llm_type, **config)
    print(para)

    end = time.time()
    print("Time:", end - start)
