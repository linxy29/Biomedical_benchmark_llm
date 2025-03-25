from utilities import process_csv
import json
import pandas as pd
import asyncio
import time
import logging
import getpass
import os

async def async_main():
    # Configure logging
    logging.basicConfig(
        filename='rerun.log',    # log file path
        level=logging.INFO,       # minimum log level
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    # Define your API keys
    deepseek_api_key = getpass.getpass("Enter your Deepseek API key: ")
    chatgpt_api_key = getpass.getpass("Enter your OpenAI API key: ")
    sf_api_key = getpass.getpass("Enter your Silicon Flow API key: ")

    start_time = time.time()
    # Define the path to your JSON file
    results_folder = "/Users/linxy29/Documents/Data/GeneRAG/benchmark/MedMCQA/results"
    models_list = ["deepseek-chat", "deepseek-reasoner", "gpt-4o", "meta-llama/Meta-Llama-3.1-405B-Instruct", "Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]

    # Get a list of all CSV files in the results folder
    csv_files = [os.path.join(results_folder, f) for f in os.listdir(results_folder) if f.endswith('.csv')]

    tasks = [
        process_csv(csv_file, models_list, deepseek_api_key, chatgpt_api_key, sf_api_key)
        for csv_file in csv_files[:10]]

    # Execute the tasks concurrently 100 at a time
    for i in range(0, len(tasks), 4):
        await asyncio.gather(*tasks[i:i + 4])
    
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

# Run the main function
asyncio.run(async_main())