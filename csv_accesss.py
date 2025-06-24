from openai import AzureOpenAI
import pandas as pd
import os
from dotenv import load_dotenv
import json
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("GEN_MODEL_API"),
    api_version=os.getenv("GEN_MODEL_VERSION"),
    azure_endpoint = os.getenv("GEN_MODEL_ENDPOINT")
    )

user_query = input("User:")

def get_attendance_data(employee_id):
    dataset = pd.read_csv("C:/Users/junai/Codes/INextLabs/AgenticRag/KB/Attandance_sheet.csv")
    result = []
    if employee_id in dataset.columns:
        for column in dataset.columns:
            result.append(f"Column Name: {column}")
            result.append("Values:")
            result.append(dataset[column].tolist())
            result.append("\n")
        return dataset[employee_id].tolist()
    else:
        return "Employee ID not found"
    
functions = [
    {
    "type": "function",
    "name": "get_attendance_data",
    "description": "Get attandance of an employee using employee ID",
    "parameters": {
        "type": "object",
        "properties": {
            "employee_id": {
                "type": "string",
                "description": "Unique identification of an employee e.g. employee 1, employee 2"
            }
        },
        "required": ["employee_id"],
        }
    }
]

messages = [
        {
        "role": "user",
         "content": """You are a helpful assistant. 
            If the user is asking about attendance, extract the employee ID and call the tool 'get_attendance_data'. 
            If the user asks something else, continue the conversation normally.
            If the user doesn’t mention an employee ID, ask them for it for security purposes.
            """
        }
    ]

while True:
    user_query = input("User:")
    if user_query.lower() in {"exit", "quit"}:
        print("chat ended")
        break
    
    messages.append({"role":"user","content":user_query})

    response = client.chat.completions.create(
        model=os.getenv("GEN_MODEL"),
        messages=messages,
        functions=functions,
        function_call="auto"  
    )

    response_message = response.choices[0].message
    messages.append(response_message)

    if response_message.function_call:
        function_name = response_message.function_call.name
        function_args = json.loads(response_message.function_call.arguments)
        print("Function call detected:",function_name)
        
        if function_name == "get_attendance_data":
            employee_id = function_args.get("employee_id")
            result = get_attendance_data(employee_id)

            function_result_message = {
                "role": "function",
                "name":function_name,
                "content":json.dumps(result)
            }
            messages.append(function_result_message)

            second_response = client.chat.completions.create(
                model=os.getenv("GEN_MODEL"),
                messages=messages,
            )

            Final_message = second_response.choices[0].message
            messages.append(Final_message)
            print("Assistant:", Final_message.content)
    else:
        print("No function call made. Response:")
        print(response_message.content)