import os
import json
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

logger = logging.getLogger("query_generator")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("query_generator.log")
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
    "queries": ["sub query 1", "sub query 2", "sub query 3"]
}, indent=2)

def query_generator(user_query: str) -> str:
    """
    Generates structured sub-queries from a complex user query using Azure OpenAI.

    Args:
        user_query: A string containing the user's question.

    Returns:
        A stringified JSON containing the list of sub-queries.
    """
    try:
        prompt = f"""System Prompt:
        - Break down the following user query into distinct, concise sub-queries, each focusing on a single topic or request.
        - Ensure each sub-query is always related to drugs or family cases (strict) 
        - Ensure each sub-query stands on its own and can be used individually for information retrieval.
        - Response Format (strict):
        {RESPONSE_FORMAT_EXAMPLE}

        Important:
        - Do not include any explanation or additional text.
        - Only return a valid JSON object as shown above.

        Question:
        {user_query}"""

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
        logger.info("Query decomposition succeeded for user input: %s", user_query)
        return result

    except Exception as e:
        logger.exception("Error during query generation for user input '%s': %s", user_query, str(e))
        return json.dumps({"queries": []})

print(query_generator("list all cases"))

"""You are a query decomposition tool.
            Instructions:
                - If the user query is related to attendance and it cannot be broken down into smaller meaningful parts, return the original question as a single-item list.
                - If the user query is related to attendance and it can be decomposed, break it into distinct, concise sub-queries. Each sub-query must focus on a single topic or request.
                - For queries not related to attendance, Break down the following user query into distinct, concise sub-queries, each focusing on a single topic or request. 
                - Ensure each sub-query stands on its own and can be used individually for information retrieval.
                - Do NOT include explanations or any text other than the JSON object.
        
            Response Format (strict):
            {RESPONSE_FORMAT_EXAMPLE}
            
            Important:
                - Do not include any explanation or additional text.
                - Only return a valid JSON object as shown above.

            Question:
            {user_query}"""

# import os
# from openai import AzureOpenAI
# from dotenv import load_dotenv

# load_dotenv()

# endpoints = os.getenv("GEN_MODEL_ENDPOINT")
# model_names = os.getenv("GEN_MODEL")
# deployments = os.getenv("GEN_MODEL")
# subscription_keys = os.getenv("GEN_MODEL_API ")
# api_versions = os.getenv("GEN_MODEL_VERSION")

# # user_query = "Which is the largest country and what is its population"

# response_format = """
#                   {
#                       "queries": ["sub query 1", "sub query 2", "sub query 3"]
#                   }
#                 """

# def query_generator(user_query):
    
#     endpoint = endpoints
#     model_name = model_names
#     deployment = deployments
#     subscription_key = subscription_keys
#     api_version = api_versions

#     client = AzureOpenAI(
#         api_version=api_version,
#         azure_endpoint=endpoint,
#         api_key=subscription_key,
#     )

#     prompt = f"""System Prompt:
#                 - Break down the following user query into distinct, concise sub-queries, each focusing on a single topic or request. 
#                 - Ensure each sub-query stands on its own and can be used individually for information retrieval.
#                 - Response Format (strict):
#                   {response_format}
                   
#                   Important:
#                   - Do not include any explanation or additional text.
#                   - Only return a valid queries in the exact JSON format shown above.
                  
#                   Question:
#                   {user_query} """

#     response = client.chat.completions.create(
#         messages=[
#             {
#                 "role": "system",
#                 "content": prompt,
#             }
#         ],
#         max_completion_tokens=800,
#         temperature=0.3,
#         top_p=1.0,
#         frequency_penalty=0.0,
#         presence_penalty=0.0,
#         model=deployment
#     )
#     return response.choices[0].message.content

# # print(query_generator(user_query))