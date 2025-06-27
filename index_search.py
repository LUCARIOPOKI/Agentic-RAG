"""
Production-ready implementation of Azure Vector Search with logging, error handling,
connection reuse, environment safety, and performance optimizations.
"""

import os
import logging
from typing import List
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from chunks import vectorize

logger = logging.getLogger("vector_search")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("vector_search.log")
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
api_key = os.getenv("AZURE_SEARCH_API_KEY")

if not endpoint or not api_key:
    raise EnvironmentError("AZURE_SEARCH_ENDPOINT or AZURE_SEARCH_API_KEY not set in environment.")

_client_cache = {}

def get_search_client(index_name: str) -> SearchClient:
    if index_name not in _client_cache:
        _client_cache[index_name] = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key)
        )
    return _client_cache[index_name]

def qstn_vectorize(question: str, index: str) -> List[str]:
    """
    Fetches top 5 relevant context chunks from Azure Search using vector similarity.

    Args:
        question: The user input/question.
        index: The name of the Azure Search index.

    Returns:
        List of string chunks as context.
    """
    try:
        search_client = get_search_client(index)

        question_vector = vectorize(question)
        if not question_vector:
            logger.warning("Vectorization failed for question: %s", question)
            return []

        vector_query = VectorizedQuery(
            vector=question_vector,
            fields="embeddings"
        )

        results = search_client.search(
            search_text=question,
            vector_queries=[vector_query],
            search_fields=["chunks"],
            top=5
        )

        chunks = [result["chunks"] for result in results if "chunks" in result]
        logger.info("Vector search succeeded for question: '%s' | Results: %d", question, len(chunks))
        return chunks

    except Exception as e:
        logger.exception("Error during vector search for question '%s' on index '%s': %s", question, index, str(e))
        return []

# print(qstn_vectorize("How to make a cookie","junaidh-text-bake"))


