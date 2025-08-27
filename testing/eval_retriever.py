
import asyncio
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
import pandas as pd
import os

# Add the project root to the Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.retriever import KnowledgeBase
from llama_index.core import Settings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import time

# Ragas uses OpenAI for some of its evaluations, so ensure the key is set.
# It can be in the .env file, which is loaded by app.config.
from app.config import settings

os.environ["OPENAI_API_KEY"] = settings.openai_api_key 


async def main():
    """
    Main function to run the retriever evaluation.
    """
    print("Starting retriever evaluation...")

    # 1. Load questions from the JSONL file
    questions_file = os.path.join(os.path.dirname(__file__), "space_article_questions.jsonl")
    with open(questions_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    questions = [item["question"] for item in data]
    ground_truths = [[item["answer"]] for item in data]

    # 2. Initialize the retriever
    print("Initializing KnowledgeBase...")
    kb = KnowledgeBase()

    # 3. Retrieve contexts for each question
    print(f"Retrieving contexts for {len(questions)} questions...")
    contexts = []
    for q in questions:
        retrieved_context = await kb.retrieve(q)
        contexts.append([retrieved_context])

    # 4. Generate answers using the LLM from the KnowledgeBase
    print("Generating answers for evaluation...")
    answers = []
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash", 
        temperature=0, 
        api_key=settings.google_api_key
    )
    llm2 = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=settings.openai_api_key)
    
    for q, c in zip(questions, contexts):
        prompt = f"Based on the following context, answer the question.\n\nContext:\n{c[0]}\n\nQuestion: {q}\n\nAnswer:"
        response = llm.invoke(prompt)
        answers.append(response.text)

    # 5. Create a dataset for Ragas
    dataset_dict = {
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(dataset_dict)

    # 6. Define metrics and run evaluation
    print("Running Ragas evaluation...")
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]
    
    result = evaluate(dataset, metrics=metrics, llm=llm2)
    print("Evaluation complete.")
    print(result)

    # 7. Print and save the results
    df = result.to_pandas()
    print("\nEvaluation Scores:")
    print(df)

    results_file = os.path.join(os.path.dirname(__file__), "retriever_evaluation_results.txt")
    with open(results_file, "w") as f:
        f.write("Retriever Evaluation Results\n")
        f.write("=" * 30 + "\n")
        f.write(df.to_string())

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    # Ensure you have a .env file in the root with GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY, and OPENAI_API_KEY
    if not all([settings.google_api_key, settings.qdrant_url, settings.qdrant_api_key]):
        print("Error: Missing required environment variables in .env file.")
        print("Please ensure GOOGLE_API_KEY, QDRANT_URL, and QDRANT_API_KEY are set.")
    else:
        asyncio.run(main())
