import json
import time
from typing import Dict
import pandas as pd
from openai import AsyncOpenAI
import asyncio
import math
import os
import logging
import re


# 1. Core Function: Get Parsed Dictionary
async def deepseek_getResult(input_text: str, api_key: str, model: str = "deepseek-reasoner") -> Dict:
    """
    Get results from DeepSeek reasoning model
    Returns: {"question": str, "answer": str, "reasoning": str}
    """
    logging.info(f"Processing input: {input_text} with model: {model}")
    start_time = time.time()

    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    system_prompt =  """
The user will provide an exam-style question. Your task is to output both an "answer" and a "confidence" score. 

- The "answer" should be the most accurate response based on the question.
- The "confidence" score should be a numerical value between 0 and 1, representing the certainty of the answer. A higher value (closer to 1) indicates high confidence, while a lower value (closer to 0) indicates uncertainty.
- Do not include any additional information in the output.

Format your response as follows:
ANSWER(CONFIDENCE)

EXAMPLE INPUT: 
Which is the highest mountain in the world?

EXAMPLE OUTPUT:
Mount Everest(0.999)
"""
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}]
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages
        )

        answer = response.choices[0].message.content
        match = re.search(r'\((\d+\.\d+)\)', answer)  

        if match:
            confidence = float(match.group(1))

        reasoning_content = None
        if model == "deepseek-reasoner":
            reasoning_content = response.choices[0].message.reasoning_content
        
        time_taken = time.time() - start_time

        return {"question": input_text, "answer": answer, "reasoning": reasoning_content, "confidence": round(confidence, 3), "time_taken": round(time_taken, 3)}

    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}")

async def sf_deepseek_getResult(input_text: str, api_key: str, model: str = "deepseek-reasoner") -> Dict:
    """
    Get results from DeepSeek reasoning model
    Returns: {"question": str, "answer": str, "reasoning": str}
    """
    logging.info(f"Processing input: {input_text} with model: {model}")

    if model == "deepseek-chat":
        sf_model = "Pro/deepseek-ai/DeepSeek-V3"
    elif model == "deepseek-reasoner":
        sf_model = "Pro/deepseek-ai/DeepSeek-R1"
    else:
        sf_model = model
        
    start_time = time.time()

    client = AsyncOpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
    
    system_prompt =  """
The user will provide an exam-style question. Your task is to output both an "answer" and a "confidence" score. 

- The "answer" should be the most accurate response based on the question.
- The "confidence" score should be a numerical value between 0 and 1, representing the certainty of the answer. A higher value (closer to 1) indicates high confidence, while a lower value (closer to 0) indicates uncertainty.
- Do not include any additional information in the output.

Format your response as follows:
ANSWER(CONFIDENCE)

EXAMPLE INPUT: 
Which is the highest mountain in the world?

EXAMPLE OUTPUT:
Mount Everest(0.999)
"""
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}]
    
    try:
        response = await client.chat.completions.create(
            model=sf_model,
            messages=messages
        )

        answer = response.choices[0].message.content
        match = re.search(r'\((\d+\.\d+)\)', answer)  

        if match:
            confidence = float(match.group(1))

        reasoning_content = None
        if model == "deepseek-reasoner":
            reasoning_content = response.choices[0].message.reasoning_content
        
        time_taken = time.time() - start_time

        return {"question": input_text, "answer": answer, "reasoning": reasoning_content, "confidence": round(confidence, 3), "time_taken": round(time_taken, 3)}

    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}")

async def chatgpt_getResult(input_text: str, api_key: str, model: str = "gpt-4o") -> Dict:
    """
    Get results from DeepSeek API and parse the JSON response
    Returns: {"question": str, "answer": str}
    """
    logging.info(f"Processing input: {input_text[100]} with model: {model}")
    start_time = time.time()
    
    client = AsyncOpenAI(api_key = api_key)
    
    system_prompt =  """
The user will provide some exam text. Please provide the answer of the question and parse the "answer", "confidence", "reasoning" and output them in JSON format. 
- The "confidence" score should be a numerical value between 0 and 1, representing the certainty of the answer. A higher value (closer to 1) indicates high confidence, while a lower value (closer to 0) indicates uncertainty.
- The "reasoning" should be a string that explains how the answer was derived.

EXAMPLE INPUT: 
Which is the highest mountain in the world?

EXAMPLE JSON OUTPUT:
{
    "answer": "Mount Everest",
    "confidence": 0.999,
    "reasoning": "Mount Everest is the highest mountain in the world."
}
"""

    messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}]
    
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            logprobs=True,
        )

        time_taken = time.time() - start_time

        return json.loads(response.choices[0].message.content) | {'question': input_text, 'time_taken': round(time_taken, 3)}
        #return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}")
    
async def sf_getResult(input_text: str, api_key: str, model: str = "meta-llama/Meta-Llama-3.1-405B-Instruct") -> Dict:
    """
    Get results from Silicon Flow API and parse the JSON response
    Returns: {"question": str, "answer": str, "reasoning": str}
    """
    logging.info(f"Processing input: {input_text} with model: {model}")
        
    start_time = time.time()

    client = AsyncOpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

    system_prompt = """
The user will provide some exam text. Please provide the answer of the question and parse the "answer", "confidence", "reasoning" and output them in JSON format. 
- The "confidence" score should be a numerical value between 0 and 1, representing the certainty of the answer. A higher value (closer to 1) indicates high confidence, while a lower value (closer to 0) indicates uncertainty.
- The "reasoning" should be a string that explains how the answer was derived.

EXAMPLE INPUT: 
Which is the highest mountain in the world?

EXAMPLE JSON OUTPUT:
{
    "answer": "Mount Everest",
    "confidence": 0.999,
    "reasoning": "Mount Everest is the highest mountain in the world."
}
"""

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}]
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"}
        )

        time_taken = time.time() - start_time

        return json.loads(response.choices[0].message.content) | {'question': input_text, 'time_taken': round(time_taken, 3)}
        #return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"API call failed: {str(e)}")

# 2. Test Function
async def get_validResult(input_text: str, api_key: str, max_retries: int = 3, func: callable = deepseek_getResult, model: str = "deepseek-chat") -> Dict:
    """
    Get valid results from DeepSeek API and parse the JSON response
    Returns: {"question": str, "answer": str}
    """
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt + 1}/{max_retries}")
            result = await func(input_text, api_key, model)
            if not all(key in result for key in ["question", "answer"]):
                raise ValueError("Missing required keys in response")
            return result | {'error': None, "model": model}
        except (ValueError, RuntimeError) as e:
            if attempt == max_retries - 1:
                return {
                    'error': str(e),
                    'model': model,
                    'question': input_text,
                    'answer': None
                }
            await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff 

async def get_modelResult(input_text: str, deepseek_api_key: str, chatgpt_api_key: str, sf_api_key: str, model: str = "deepseek-chat") -> Dict:
    """
    Given the model name and input text, return the model's response.

    Returns:
        dict: A dictionary containing:
            - "question" (str): The input question.
            - "answer" (str or None): The model's response.
            - "error" (str or None): An error message if applicable.
            - "model" (str): The name of the model used.
    """
    # DeepSeek models
    if model in ["deepseek-chat", "deepseek-reasoner", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]:
        # result = await get_validResult(input_text, deepseek_api_key, func=deepseek_getResult, model="deepseek-reasoner")
        result = await get_validResult(input_text, sf_api_key, func=sf_deepseek_getResult, model=model)
        return result

    # ChatGPT model
    elif model in ["gpt-4o", "o3-mini", "o1-mini"]:
        result = await get_validResult(input_text, chatgpt_api_key, func=chatgpt_getResult, model=model)
        return result

    # Other models
    elif model in ["meta-llama/Meta-Llama-3.1-405B-Instruct", "Qwen/Qwen2.5-72B-Instruct"]:
        result = await get_validResult(input_text, sf_api_key, func=sf_getResult, model=model)
        return result

    # Default case: model not found
    return {"question": input_text, "answer": None, "error": "Model not found", "model": model}

# 3. Write function
async def get_allResults(input_text: str, deepseek_api_key: str, chatgpt_api_key: str, sf_api_key: str, path: str, id: str) -> None:
    """
    Get results from DeepSeek API and chatGPT API
    Save the results in a CSV file
    """
    # Output file path
    output_file_path = f"{path}/results_{id}.csv"

    # Check if the file already exists
    if os.path.exists(output_file_path):
        logging.info(f"Output file for index {id} already exists. Skipping analysis.")
        return 

    deepseek_chat_res, deepseek_reasoning_res, chatgpt_res, llama_res, qwen_res, deepseek_qwen_res, deepseek_llama_res = await asyncio.gather(
            get_modelResult(input_text, deepseek_api_key, chatgpt_api_key, sf_api_key, model="deepseek-chat"),
            get_modelResult(input_text, deepseek_api_key, chatgpt_api_key, sf_api_key, model="deepseek-reasoner"),
            get_modelResult(input_text, deepseek_api_key, chatgpt_api_key, sf_api_key, model="gpt-4o"),
            get_modelResult(input_text, deepseek_api_key, chatgpt_api_key, sf_api_key, model="meta-llama/Meta-Llama-3.1-405B-Instruct"),
            get_modelResult(input_text, deepseek_api_key, chatgpt_api_key, sf_api_key, model="Qwen/Qwen2.5-72B-Instruct"),
            get_modelResult(input_text, deepseek_api_key, chatgpt_api_key, sf_api_key, model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
            get_modelResult(input_text, deepseek_api_key, chatgpt_api_key, sf_api_key, model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
        )
    df = pd.DataFrame([deepseek_chat_res, deepseek_reasoning_res, chatgpt_res, llama_res, qwen_res, deepseek_qwen_res, deepseek_llama_res])
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # Write the result for the current row to a separate file
    df.to_csv(output_file_path, index=False, encoding="utf-8")
    
    logging.info(f"Processed and saved results for index {id} to {output_file_path}")

async def process_csv(file_path, models_list, deepseek_api_key, chatgpt_api_key, sf_api_key):
    """
    Process a CSV file containing model results, ensuring results from all models in the `models_list` are present.
    """
    # 1. Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # 2. Remove rows with missing values in the 'result' column
    df_clean = df.dropna(subset=['answer'])

    # Get existing models in the cleaned DataFrame
    existing_models = df_clean['model'].unique()

    # Identify models not present in the cleaned DataFrame
    missing_models = [model for model in models_list if model not in existing_models]

    question = df_clean['question'].iloc[0]

    # Get results for missing models
    tasks = [get_modelResult(question, deepseek_api_key, chatgpt_api_key, sf_api_key, model) for model in missing_models]
    missing_results = await asyncio.gather(*tasks)

    # Add the missing results to the DataFrame
    new_row = pd.DataFrame(missing_results)
    df_clean = pd.concat([df_clean, new_row], ignore_index=True)

    # Save the updated DataFrame back to CSV
    df_clean.to_csv(file_path, index=False)
    logging.info(f"Updated CSV saved to {file_path}")


