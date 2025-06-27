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
