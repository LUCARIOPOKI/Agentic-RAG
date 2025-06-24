from pymongo import MongoClient
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
from bson import json_util
import json 
load_dotenv()
username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")

def get_attendance_data(employee_id):

    encoded_user = quote_plus(username)
    encoded_password = quote_plus(password)

    uri = f"mongodb+srv://{encoded_user}:{encoded_password}@clustertest.zljdxce.mongodb.net/"
    client = MongoClient(uri)
    db = client["test"]
    collection = db["attendance"]
    result = collection.find({"emp_id":"E001"})
    context = []
    for doc in result:
        context.append(doc)
    return context

# print(get_attendance_data("E004"))

# client = AzureOpenAI(
#     api_key=os.getenv("GEN_MODEL_API"),
#     api_version=os.getenv("GEN_MODEL_VERSION"),
#     azure_endpoint = os.getenv("GEN_MODEL_ENDPOINT")
#     )

# # user_query = input("User:")

# functions = [
#     {
#     "type": "function",
#     "name": "get_attendance_data",
#     "description": "Get attandance of an employee using employee ID",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "employee_id": {
#                 "type": "string",
#                 "description": "Unique identification of an employee e.g. 1074, 1075 "
#             }
#         },
#         "required": ["employee_id"],
#         }
#     }
# ]

# messages = [
#         {
#         "role": "user",
#          "content": """You are a helpful assistant. 
#             If the user is asking about attendance, extract the employee ID and call the tool 'get_attendance_data'. 
#             If the user asks something else, continue the conversation normally.
#             If the user doesn’t mention an employee ID, ask them for it for security purposes.
#             """
#         }
#     ]

# while True:
#     user_query = input("User:")
#     if user_query.lower() in {"exit", "quit"}:
#         print("chat ended")
#         break
    
#     messages.append({"role":"user","content":user_query})

#     response = client.chat.completions.create(
#         model=os.getenv("GEN_MODEL"),
#         messages=messages,
#         functions=functions,
#         function_call="auto"  
#     )

#     response_message = response.choices[0].message
#     messages.append(response_message)

#     if response_message.function_call:
#         function_name = response_message.function_call.name
#         function_args = json.loads(response_message.function_call.arguments)
#         print("Function call detected:",function_name)
        
#         if function_name == "get_attendance_data":
#             employee_id = function_args.get("employee_id")
#             result = get_attendance_data(employee_id)

#             function_result_message = {
#                 "role": "function",
#                 "name":function_name,
#                 "content":json_util.dumps(result)
#             }
#             messages.append(function_result_message)

#             second_response = client.chat.completions.create(
#                 model=os.getenv("GEN_MODEL"),
#                 messages=messages,
#             )

#             Final_message = second_response.choices[0].message
#             messages.append(Final_message)
#             print("Assistant:", Final_message.content)
#     else:
#         print("No function call made. Response:")
#         print(response_message.content)


# create a repo to manage the MongoDB (have all kinds of code and schemas) 
# with open(r"C:\Users\junai\Codes\INextLabs\AgenticRag\KB\attendance_dataset_simple_date.json","r") as f:
#     data = json.load(f)
# for doc in data:
#     doc["date"] = datetime.strptime(doc["date"], "%Y-%m-%d")
# collection.insert_many(data)
# print("Uploaded successfully")
# collection.insert_one({"emp_id":1000,"Name":"Poki"})