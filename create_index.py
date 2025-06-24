from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchableField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    SemanticConfiguration,
    VectorSearchProfile,
    SemanticField,
    SemanticSearch,
    SemanticPrioritizedFields
)
from azure.search.documents.indexes.models import AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
api_key = os.getenv("AZURE_SEARCH_API_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
deployment_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY") 
index_name = os.getenv("INDEX") 

if not all([endpoint, api_key, deployment, deployment_endpoint, openai_api_key]):
    raise ValueError("Missing required environment variables. Please set AZURE_SEARCH_ENDPOINT, "
                   "AZURE_SEARCH_API_KEY, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_ENDPOINT, "
                   "and AZURE_OPENAI_API_KEY")

client = SearchIndexClient(
    endpoint=endpoint, 
    credential=AzureKeyCredential(api_key)
)

vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="MyHnsAlgo",
            m=4,
            ef_construction=100  
        )
    ],
    profiles=[
        VectorSearchProfile(
            name="MyHnsAlgoProfile",
            algorithm_configuration_name="MyHnsAlgo",
            vectorizer="MyVectorizer"  
        )
    ],
    vectorizers=[
        AzureOpenAIVectorizer(
            vectorizer_name ="MyVectorizer", 
            parameters=AzureOpenAIVectorizerParameters(
                resource_url=deployment_endpoint,  
                deployment_name=deployment,  
                api_key=openai_api_key,
                model_name=deployment
            )
        )
    ]
)
semantic_config = SemanticConfiguration(
    name="My-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="id"),
        keywords_fields=[SemanticField(field_name="chunks")],
        content_fields=[SemanticField(field_name="chunks")]
    )
)

semantic_search = SemanticSearch(
    configurations=[semantic_config]
)

# index_name = index_name

fields = [
    SimpleField(
        name="id", 
        type=SearchFieldDataType.String, 
        key=True, 
        sortable=True,
        filterable=True, 
        facetable=True
    ),
    SearchableField(  
        name="chunks", 
        type=SearchFieldDataType.String,
        sortable=False  
    ),
    SearchField(
        name="embeddings", 
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single), 
        searchable=True, 
        vector_search_dimensions=1536,  
        vector_search_profile_name="MyHnsAlgoProfile"
    ),
    SimpleField(
        name="category",
        type=SearchFieldDataType.Collection(SearchFieldDataType.String),
        filterable=True,
        facetable=True
    )
]

index = SearchIndex(
    name=index_name, 
    fields=fields, 
    vector_search=vector_search, 
    semantic_search=semantic_search
)

try:
    result = client.create_index(index)
    print(f"Index '{result.name}' created successfully")
except Exception as e:
    print(f"Error creating index: {e}")
    try:
        result = client.create_or_update_index(index)
        print(f"Index '{result.name}' updated successfully")
        pprint(index.as_dict())
    except Exception as update_error:
        print(f"Error updating index: {update_error}")

# # To delete the index (commented out)
# try:
#     client.delete_index(index_name)
#     print(f"Index '{index_name}' deleted successfully")
# except Exception as e:
#     print(f"Error deleting index: {e}")