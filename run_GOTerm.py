from utilities import get_allResults
import json
import pandas as pd
import asyncio
import time
import logging
import getpass

async def async_main():
    # Configure logging
    logging.basicConfig(
        filename='run_GOTerm.log',    # log file path
        level=logging.INFO,       # minimum log level
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    # Define your API keys
    sf_deepseek_api_key = getpass.getpass("Enter your Silicon Flow API key: ")
    chatgpt_api_key = getpass.getpass("Enter your OpenAI API key: ")

    start_time = time.time()
    # Define the path to your JSON file
    file_path = "/Users/linxy29/Documents/Data/GeneRAG/benchmark/GOTerm/go_terms_and_genes.csv"
    output_path = "/Users/linxy29/Documents/Data/GeneRAG/benchmark/GOTerm/results"

    df = pd.read_csv(file_path)

    # Create a list of tasks using list comprehension
    tasks = [
        get_allResults(
            input_text=f"""
            You are given a list of genes. Your task is to infer the GO term names. 

            Here is the list of genes:
            Genes: {row['Genes']}
            """,
            deepseek_api_key=sf_deepseek_api_key, 
            chatgpt_api_key=chatgpt_api_key,
            path=output_path,
            id=row['GO_ID'])
        for _, row in df.head(10).iterrows()
    ]

    # Execute the tasks concurrently at a time
    batch_size = 2  # Change this to process more tasks at once
    for i in range(0, len(tasks), batch_size):
        await asyncio.gather(*tasks[i:i + batch_size])

    
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

# Run the main function
asyncio.run(async_main())