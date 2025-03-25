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
        filename='run_PubMedQA.log',    # log file path
        level=logging.INFO,       # minimum log level
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    # Define your API keys
    deepseek_api_key = getpass.getpass("Enter your deepseek API key: ")
    chatgpt_api_key = getpass.getpass("Enter your OpenAI API key: ")
    sf_api_key = getpass.getpass("Enter your Silicon Flow API key: ")

    start_time = time.time()
    # Define the path to your JSON file
    file_path = "/Users/linxy29/Documents/Data/GeneRAG/benchmark/PubMedQA/ori_pqal.json"
    output_path_null = "/Users/linxy29/Documents/Data/GeneRAG/benchmark/PubMedQA/results_null"
    output_path_CoR = "/Users/linxy29/Documents/Data/GeneRAG/benchmark/PubMedQA/results_CoR"

    # Load the JSON file
    with open(file_path, "r") as f:
        data = json.load(f)

    # Convert JSON to a structured list for DataFrame
    rows = []
    for pmid, details in data.items():
        row = {
            "PMID": pmid,
            "QUESTION": details.get("QUESTION", ""),
            "CONTEXTS": " ||| ".join(details.get("CONTEXTS", [])),  # Join context paragraphs
            "LABELS": " ||| ".join(details.get("LABELS", [])),  # Join labels
            "MESHES": " ||| ".join(details.get("MESHES", [])),  # Join MeSH terms
            "YEAR": details.get("YEAR", ""),
            "reasoning_required_pred": details.get("reasoning_required_pred", ""),
            "reasoning_free_pred": details.get("reasoning_free_pred", ""),
            "final_decision": details.get("final_decision", ""),
            "LONG_ANSWER": details.get("LONG_ANSWER", ""),
        }
        rows.append(row)

    # Convert list to DataFrame
    df = pd.DataFrame(rows)

    # Create a list of tasks using list comprehension
    tasks_null = [
        get_allResults(
            input_text="You will be provided with a question and a set of contexts. Your task is to answer the question "
        "based on the information in the contexts. Your answer should be one of the following: 'yes', 'no', or 'maybe'. "
        "If the contexts provide clear evidence supporting the question, answer 'yes'. If the contexts provide clear "
        "evidence contradicting the question, answer 'no'. If the contexts are unclear or insufficient to determine the "
        "answer, answer 'maybe'.\n\n"
        f"Question: {row['QUESTION']}\n\n"
        f"Contexts: {row['CONTEXTS']}",
            deepseek_api_key=deepseek_api_key, 
            chatgpt_api_key=chatgpt_api_key,
            sf_api_key=sf_api_key,
            path=output_path_null,
            id=row['PMID'])
        for _, row in df.head(4).iterrows()
    ]

    tasks_CoR = [
        get_allResults(
            input_text="You will be provided with a question and a set of contexts. Your task is to answer the question based on the information in the contexts. "
        "Your answer should be one of the following: 'yes', 'no', or 'maybe'. Follow these steps to determine your answer:\n\n"
        "1. **Understand the Question**: Read the question carefully to identify what is being asked.\n"
        "2. **Analyze the Contexts**: Review the provided contexts to extract relevant information that supports or contradicts the question.\n"
        "3. **Reasoning**: Use the information in the contexts to reason through the answer. If the contexts provide clear evidence supporting the question, answer 'yes'. If the contexts provide clear evidence contradicting the question, answer 'no'. If the contexts are unclear or insufficient to determine the answer, answer 'maybe'.\n"
        "4. **Example**: Use the following example to guide your reasoning:\n\n"
        "**Example Question**: Is high-sensitivity C-reactive protein associated with carotid atherosclerosis in healthy Koreans?\n"
        "**Example Contexts**:\n"
        "- There is a positive association between chronic inflammation and the risk of cardiovascular disease, but whether there is an association between C-reactive protein (CRP) and carotid atherosclerosis is controversial.\n"
        "- We investigated the relationship between high-sensitivity CRP (hsCRP) levels and carotid intima-media thickness (IMT) in healthy Koreans.\n"
        "- Higher hsCRP quartile groups had higher mean IMTs, as compared with the lowest quartile (P<0.001 for the trend across quartiles). However, after adjustment for age, the relationship between hsCRP level and IMT was substantially weaker (P = 0.018). After additional adjustments for conventional cardiovascular risk factors, no significant association was observed (P = 0.548).\n"
        "**Reasoning**: The contexts show that while there is an initial association between hsCRP levels and carotid IMT, this association becomes non-significant after adjusting for age and other cardiovascular risk factors. Therefore, the answer to the question is 'no'.\n\n"
        "**Answer**: no\n\n"
        "Now, answer the following question based on the provided contexts and reasoning steps:\n\n"
        f"Question: {row['QUESTION']}\n\n"
        f"Contexts: {row['CONTEXTS']}",
            deepseek_api_key=deepseek_api_key, 
            chatgpt_api_key=chatgpt_api_key,
            sf_api_key=sf_api_key,
            path=output_path_CoR,
            id=row['PMID'])
        for _, row in df.head(4).iterrows()
    ]

    # Execute the tasks concurrently at a time
    batch_size = 2  # Change this to process more tasks at once
    for i in range(0, len(tasks_null), batch_size):
        await asyncio.gather(*tasks_null[i:i + batch_size])
        await asyncio.gather(*tasks_CoR[i:i + batch_size])

    
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

# Run the main function
asyncio.run(async_main())