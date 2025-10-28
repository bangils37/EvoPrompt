import json
import os
import requests
import sys
from tqdm import tqdm
import time
from BBH.utils import read_yaml_file, batchify

# Google Gemini
try:
    import google.generativeai as genai
except ImportError:
    genai = None




def form_request(data, **kwargs):
    """
    Form Gemini API request in the standard format
    """
    if isinstance(data, str):
        contents = [
            {
                "parts": [
                    {
                        "text": data
                    }
                ]
            }
        ]
    else:
        raise ValueError("Data must be a string")
    
    request_data = {
        "contents": contents,
        **kwargs
    }
    return request_data


current_dir = os.path.dirname(os.path.abspath(__file__))
def llm_init(auth_file=os.path.join(current_dir, '../auth.yaml'), llm_type='gemini', setting='default'):
    auth = read_yaml_file(auth_file)[llm_type][setting]
    api_type = auth.get("api_type", "gemini")

    if api_type == "gemini":
        # Nếu sử dụng key manager, không cần configure genai
        if not auth.get("use_key_manager", False):
            if genai is None:
                raise ImportError("Please install google-generativeai (pip install google-generativeai)")
            genai.configure(api_key=auth["api_key"])

    return auth




def llm_query(data, client, type, task, **config):
    """
    Gemini-only LLM query with full Gemini API format support
    Supports both direct Gemini API and Key Manager endpoint
    """
    api_type = config.get("api_type", "gemini")
    
    if api_type != "gemini":
        raise ValueError("Only Gemini API is supported. Please set api_type to 'gemini' in auth.yaml")
    
    use_key_manager = config.get("use_key_manager", False)
    
    if use_key_manager:
        return _query_key_manager(data, config, task)
    else:
        return _query_gemini_direct(data, config, task)


def _query_key_manager(data, config, task):
    """
    Query using Key Manager endpoint
    """
    key_manager_url = config.get("api_base", "http://localhost:7749")
    endpoint = f"{key_manager_url}/generate"
    
    hypos = []
    
    if isinstance(data, list):
        for d in tqdm(data):
            try:
                request_data = form_request(d, **_get_generation_config(config))
                response = requests.post(
                    endpoint,
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                if response.status_code == 200:
                    result = response.json()
                    text = result["candidates"][0]["content"]["parts"][0]["text"]
                    hypos.append(text.strip())
                elif response.status_code == 429:
                    print("Key Manager error: All keys rate limited")
                    time.sleep(60)
                    hypos.append("")
                else:
                    print(f"Key Manager error: {response.status_code} - {response.text}")
                    hypos.append("")
            except Exception as e:
                print("Key Manager exception:", e)
                time.sleep(5)
                hypos.append("")
    else:
        try:
            request_data = form_request(data, **_get_generation_config(config))
            response = requests.post(
                endpoint,
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                result = response.json()
                hypos = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            elif response.status_code == 429:
                print("Key Manager error: All keys rate limited")
                time.sleep(60)
                hypos = ""
            else:
                print(f"Key Manager error: {response.status_code} - {response.text}")
                hypos = ""
        except Exception as e:
            print("Key Manager exception:", e)
            hypos = ""
    
    return hypos


def _query_gemini_direct(data, config, task):
    """
    Query using Gemini API directly
    """
    model_name = config.get("model", "gemini-2.0-flash")
    model = genai.GenerativeModel(model_name)
    
    hypos = []
    gen_config = _get_generation_config(config)
    
    # Extract generationConfig if present
    generation_config = gen_config.pop("generationConfig", {})
    tools = gen_config.pop("tools", None)
    safety_settings = gen_config.pop("safetySettings", None)
    
    if isinstance(data, list):
        for d in tqdm(data):
            try:
                kwargs = {}
                if generation_config:
                    kwargs["generation_config"] = generation_config
                if tools:
                    kwargs["tools"] = tools
                if safety_settings:
                    kwargs["safety_settings"] = safety_settings
                
                resp = model.generate_content(d, **kwargs)
                hypos.append(resp.text.strip())
            except Exception as e:
                print("Gemini error:", e)
                time.sleep(5)
                hypos.append("")
    else:
        try:
            kwargs = {}
            if generation_config:
                kwargs["generation_config"] = generation_config
            if tools:
                kwargs["tools"] = tools
            if safety_settings:
                kwargs["safety_settings"] = safety_settings
            
            resp = model.generate_content(data, **kwargs)
            hypos = resp.text.strip()
        except Exception as e:
            print("Gemini error:", e)
            hypos = ""
    
    return hypos


def _get_generation_config(config):
    """
    Extract Gemini API compatible configuration
    """
    gen_config = {}
    
    if "generationConfig" in config:
        gen_config["generationConfig"] = config["generationConfig"]
    else:
        # Default generation config
        gen_config["generationConfig"] = {
            "temperature": config.get("temperature", 0.7),
            "maxOutputTokens": config.get("max_tokens", 1000)
        }
    
    if "tools" in config:
        gen_config["tools"] = config["tools"]
    
    if "safetySettings" in config:
        gen_config["safetySettings"] = config["safetySettings"]
    
    return gen_config



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
    llm_type = 'gemini'
    start = time.time()

    data = [
        """Q: Tom bought a skateboard for $9.46, and spent $9.56 on marbles. Tom also spent $14.50 on shorts. In total, how much did Tom spend on toys? A: Let's think step by step."""
    ]

    config = llm_init(auth_file="auth.yaml", llm_type=llm_type, setting="default")
    para = paraphrase(data, llm_client, llm_type, **config)
    print(para)

    end = time.time()
    print("Time:", end - start)
