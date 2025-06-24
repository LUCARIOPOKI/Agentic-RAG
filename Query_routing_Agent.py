import os
import json
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    filename="agent_model.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

try:
    ENDPOINT = os.environ["GEN_MODEL_ENDPOINT"]
    DEPLOYMENT_NAME = os.environ["GEN_MODEL"]
    API_KEY = os.environ["GEN_MODEL_API"].strip()
    API_VERSION = os.environ["GEN_MODEL_VERSION"]
except KeyError as e:
    logging.critical(f"Missing environment variable: {e}")
    raise

client = AzureOpenAI(
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
)

RESPONSE_FORMAT = (
    "return the name of the selected knowledge base.\n"
    "drugs|family\n"
    "{\n"
    "    \"case_category\": choose one category from the given two\n"
    "}"
)

def agent_model(user_query: str) -> str:
    """
    Determines the appropriate knowledge base for a given query using Azure OpenAI.

    Args:
        user_query (str): User's input question.

    Returns:
        str: A JSON-formatted string indicating the selected knowledge base.
    """
    try:
        prompt = f"""System Prompt:
                 - You are an Agent responsible for selecting the most appropriate case_category to use for retrieving context for a Retrieval-Augmented Generation (RAG) system.
                 Instructions: 
                 - You have access to the following two case_category:
                     - If the question is related to the drugs choose case_category as drugs.
                     - If the question is related to the family issues case_category as family.
                 - Based on the input context or your internal decision logic, select and return the name of exactly one of these case_category.
                 - If the input context is unclear or does not match any specific case_category, return the name of both case_categories.
                 - If the question doesnot match any of the above domains, return name of both case_categories.
                   Response Format (strict):
                   {RESPONSE_FORMAT}
                  
                   Important:
                   - Do not include any explanation or additional text.
                   - Only return a valid knowledge base name in the exact JSON format shown above.
                 
                   Question:
                   {user_query}
        """

        response = client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            max_completion_tokens=800,
            temperature=0.3,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=DEPLOYMENT_NAME
        )

        reply = response.choices[0].message.content
        logging.info(f"Agent model selected case_category: {reply}")
        return reply

    except Exception as e:
        logging.exception(f"500: Error in agent_model - {e}")
        return json.dumps({"case_category": "Undecided"})
    
"""System Prompt:
                 - You are an Agent responsible for selecting the most appropriate Knowledge Base (KB) to use for retrieving context for a Retrieval-Augmented Generation (RAG) system.
                 Instructions: 
                 - You have access to the following four knowledge bases:
                     - If the question is related to the dinosaurs choose junaidh-text-dino.
                     - If the question is related to the baking recipes and culinary information then choose junaidh-text-bake.
                     - If the question is related to fitness and physical health then choose junaidh-text-fit.
                     - If the question is related to attendance or the user requesting attendance of the user then return junaidh-text-DB.
                 - Based on the input context or your internal decision logic, select and return the name of exactly one of these knowledge bases.
                 - If the input context is unclear or does not match any specific domain, return the name of the default knowledge base (you may define which one is default).
                 - If the question doesnot match any of the above domains, return the name of the NoKB knowledge base.
                   Response Format (strict):
                   {RESPONSE_FORMAT}
                  
                   Important:
                   - Do not include any explanation or additional text.
                   - Only return a valid knowledge base name in the exact JSON format shown above.
                 
                   Question:
                   {user_query}
        """

# print(agent_model("List cases about Section 5 (i) of Republic Act No. 9262"))

# import os
# from openai import AzureOpenAI
# from dotenv import load_dotenv

# load_dotenv()

# endpoints = os.getenv("GEN_MODEL_ENDPOINT")
# model_names = os.getenv("GEN_MODEL")
# deployments = os.getenv("GEN_MODEL")
# subscription_keys = os.getenv("GEN_MODEL_API ")
# api_versions = os.getenv("GEN_MODEL_VERSION")

# # user_query = "How to make a cookie"
# response_format = """return the name of the selected knowledge base.
#                   junaidh-text-[dino|bake|fit|NoKB]
#                   {
#                       "knowledge_base": the knowledge base you chose
#                   }"""

# def agent_model(user_query):
    
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
#                 - You are an Agent responsible for selecting the most appropriate Knowledge Base (KB) to use for retrieving context for a Retrieval-Augmented Generation (RAG) system.

#                 Instructions: 
#                 - You have access to the following three knowledge bases:
#                     - If the question is related to the dinosaurs choose junaidh-text-dino.
#                     - If the question is related to the baking recipes and culinary information then choose junaidh-text-bake.
#                     - If the question is related to fitness and physical health then choose junaidh-text-fit.
#                 - Based on the input context or your internal decision logic, select and return the name of exactly one of these knowledge bases.
#                 - If the input context is unclear or does not match any specific domain, return the name of the default knowledge base (you may define which one is default).
#                 - If the question doesnot match any of the above domains, return the name of the NoKB knowledge base.
#                   Response Format (strict):
#                   {response_format}
                   
#                   Important:
#                   - Do not include any explanation or additional text.
#                   - Only return a valid knowledge base name in the exact JSON format shown above.
                  
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

# # print(agent_model(user_query))