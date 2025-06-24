import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

endpoints = os.getenv("GEN_MODEL_ENDPOINT")
model_names = os.getenv("GEN_MODEL")
deployments = os.getenv("GEN_MODEL")
subscription_keys = os.getenv("GEN_MODEL_API ")
api_versions = os.getenv("GEN_MODEL_VERSION")

user_query = " "
retrieved_document = " "

def rag_model(user_query, retrieved_document):
    
    endpoint = endpoints
    # model_name = model_names
    deployment = deployments
    subscription_key = subscription_keys
    api_version = api_versions

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    prompt = f"""
                # System:
                    Your name is **POKI**, a friendly and knowledgeable AI assistant, helping users with everyday Dinosaur, baking and fitness -related queries.

                    ## Instructions
                    1. **Always ground your answer ONLY in the retrieved documents.**
                       - Do not use your own knowledge to answer the user's question
                       - Do not invent or assume information.

                    2. **If the documents clearly answer the query:**
                       - Your tone should be in a friendly, professional tone, warm, conversational, and encouraging — like a helpful colleague. Always answer in clear, professional English, and keep responses concise but empathetic.
                         For example: "Sure! I’ve got that info for you right here"
                       - Respond directly, summarizing the relevant content.
                       - Keep it short and useful — aim to solve the problem quickly.

                    3. **If the documents partially match the query:**
                       - Just say what is given in the context in a short concise manner.
                       - Invite the user to follow up or clarify if needed.

                    4. **If there is conflicting information:**
                       - Acknowledge the conflict neutrally and suggest how the user might verify or proceed (e.g., contacting HR directly).

                    5. **If the question is vague or ambiguous:**
                       - Ask for clarification in a friendly way, offering examples if helpful.

                    6. **If the answer cannot be found or is outside the scope:**
                       - Be honest and guide the user to the next best action (e.g., I am sorry I dont have that information).
                    
                    7. **Always use the given format for the response**
                        - Mention the topic of the question aswell, Use the context to determine the topic
                        - Select the topic from these 4 options: Dinosaur, Baking, Fitness, General based on the user query
                        - Your response should be in the format:"**Topic**: The Topic goes here","Your response".

                    ## DOCUMENTS RETRIEVED FROM HR KNOWLEDGE BASE:
                    {retrieved_document}

                    ## USER'S QUESTION:
                    {user_query} 
                """

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            }
        ],
        max_completion_tokens=800,
        temperature=0.3,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=deployment
    )

    return response.choices[0].message.content

# print(model(user_query, retrieved_document))