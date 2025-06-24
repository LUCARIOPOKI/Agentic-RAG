import os
import json
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

logger = logging.getLogger("query_generator")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("query_validation.log")
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

ENDPOINT = os.getenv("GEN_MODEL_ENDPOINT")
MODEL_NAME = os.getenv("GEN_MODEL")
DEPLOYMENT = os.getenv("GEN_MODEL")
API_KEY = os.getenv("GEN_MODEL_API")
API_VERSION = os.getenv("GEN_MODEL_VERSION")

if not all([ENDPOINT, MODEL_NAME, DEPLOYMENT, API_KEY, API_VERSION]):
    raise EnvironmentError("One or more required environment variables are missing for Azure OpenAI configuration.")

azure_client = AzureOpenAI(
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
)

RESPONSE_FORMAT_EXAMPLE = json.dumps({
    "Validation":"Good/Average/bad",
    "Improvements":"List of suggestions",
    "Summarize":" Summary of the validation and improvements"
}, indent=2)

def Result_validation(user_query: str, context: List, LLMRes: str) -> str:
    """
    Generates structured sub-queries from a complex user query using Azure OpenAI.

    Args:
        user_query: A string containing the user's question.

    Returns:
        A stringified JSON containing the list of sub-queries.
    """
    try:
        prompt = f"""
                You are an expert AI response critic. Your task is to critically evaluate whether the answer provided by another AI is accurate, complete, and relevant based on the given context and question.
                    Carefully read the Question, Context, and the LLM Response.

                    Then, perform the following:
                    Verdict: Does the response answer the question correctly and based only on the context? (Yes / No)
                    Critique: If not, explain what is missing, incorrect, misleading, or hallucinated.
                    Suggestions: Offer a corrected or improved version of the response (if needed), using only the provided context.

                    Question:
                    {user_query}

                    Context:
                    {context}

                    LLM Response:
                    {LLMRes}

                    Now give your evaluation as the AI critic in this format.
                    {RESPONSE_FORMAT_EXAMPLE}

                """

        response = azure_client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model=DEPLOYMENT,
            max_tokens=800,
            temperature=0.3,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        result = response.choices[0].message.content.strip()
        logger.info("Query validation succeeded for user input: %s", user_query)
        return result

    except Exception as e:
        logger.exception("Error during query validation for user input '%s': %s", user_query, str(e))
        return json.dumps({"queries": []})