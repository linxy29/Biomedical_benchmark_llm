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
        filename='run_annotation.log',    # log file path
        level=logging.INFO,       # minimum log level
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    # Let user input the API keys
    sf_deepseek_api_key = getpass.getpass("Enter your Silicon Flow API key: ")
    chatgpt_api_key = getpass.getpass("Enter your OpenAI API key: ")

    start_time = time.time()
    # Define the path to your JSON file
    file_path = "/Users/linxy29/Documents/Data/GeneRAG/benchmark/annotation/41586_2020_2157_MOESM3_ESM.xlsx"
    output_path = "/Users/linxy29/Documents/Data/GeneRAG/benchmark/annotation/HCL/results"
    sheet_name = "MCA markers for 104 clusters"

    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)  # Read without headers

    # Extract cell type names from the first row
    cell_types = df.iloc[0, 1::4].values  # Select every 4th column starting from index 1

    # Extract differentially expressed genes (DEGs) from the second row onward
    df = df.iloc[1:, 1::4]  # Select every 4th column starting from index 1

    # Rename columns with cell type names
    df.columns = cell_types

    # remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Create a list to hold the new rows
    data = []

    # Loop over each cell type (column)
    for cell_type in df.columns:
        # Get the list of genes for this cell type, dropping NaN values
        gene_list = df[cell_type].dropna().tolist()
        data.append({"cell_type": cell_type, "genes": gene_list[1:21]})
    

    # Convert the list of dictionaries to a DataFrame
    result_df = pd.DataFrame(data)

    # Create a list of tasks using list comprehension
    tasks = [
        get_allResults(
            input_text=f"""
            You are given a list of differentially expressed genes (DEGs) for a specific cell type. Your task is to infer the cell type name.

            Here is the list of genes:
            {row['genes']}
            """,
            deepseek_api_key=sf_deepseek_api_key, 
            chatgpt_api_key=chatgpt_api_key,
            path=output_path,
            id=row['cell_type'])
        for _, row in result_df.head(10).iterrows()
    ]

    # Execute the tasks concurrently at a time
    batch_size = 2  # Change this to process more tasks at once
    for i in range(0, len(tasks), batch_size):
        await asyncio.gather(*tasks[i:i + batch_size])

    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

# Run the main function
asyncio.run(async_main())