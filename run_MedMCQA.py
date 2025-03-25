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
        filename='output.log',    # log file path
        level=logging.INFO,       # minimum log level
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    # Define your API keys
    deepseek_api_key = getpass.getpass("Enter your Silicon Flow API key: ")
    chatgpt_api_key = getpass.getpass("Enter your OpenAI API key: ")

    start_time = time.time()
    # Define the path to your JSON file
    file_path = "/Users/linxy29/Documents/Data/GeneRAG/benchmark/MedMCQA/sampled_data.json"
    output_path = "/Users/linxy29/Documents/Data/GeneRAG/benchmark/MedMCQA/results"

    # Read the file line by line and parse each JSON object
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                json_obj = json.loads(line.strip())  # Parse each line as JSON
                data.append(json_obj)  # Append to the list
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

    # Convert the list of JSON objects into a DataFrame
    df = pd.DataFrame(data)

    tasks = [
    get_allResults(
        f"Please choice the correct option (1, 2, 3 or 4) for the following question and output the answer as 1, 2, 3 or 4. Question: {row['question']} option 1: {row['opa']}, option 2: {row['opb']}, option 3: {row['opc']}, option 4: {row['opd']}",
        deepseek_api_key,  
        chatgpt_api_key,
        path=output_path,
        id=row['id'])
    for _, row in df.iterrows()
]

    # Execute the tasks concurrently 100 at a time
    for i in range(0, len(tasks), 2):
        await asyncio.gather(*tasks[i:i + 2])
    
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

# Run the main function
asyncio.run(async_main())