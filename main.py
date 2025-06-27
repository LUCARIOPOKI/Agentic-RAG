import asyncio
# import time
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any, Dict, Optional

from index_search import qstn_vectorize
from Query_routing_Agent import agent_model
from rag_llm import rag_model
from database_access import get_attendance_data
from WebSearchTool import search_DDG
from Query_receiver_Agent import query_generator
from Testing import Result_validation

logging.basicConfig(
    filename="chatbot_logs.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class RAGChatBot:
    """
    A modular, production-grade Retrieval-Augmented Generation (RAG) chatbot
    that supports web search fallback and query decomposition.
    """

    def __init__(
        self,
        vectorizer,
        agent,
        rag,
        web_search,
        query_gen,
        Validate,
        DataBase
    ):
        """
        Initialize the RAGChatBot with modular components.

        Args:
            vectorizer: Function to fetch context from knowledge base.
            agent: Function to determine which knowledge base to use.
            rag: Function to generate a final response using LLM + context.
            web_search: Function to search the web.
            query_gen: Function to decompose query into sub-queries.
            classifier: Function to classify query type.
        """
        self.vectorizer = vectorizer
        self.agent = agent
        self.rag = rag
        self.web_search = web_search
        self.query_gen = query_gen
        self.Validate = Validate
        self.DataBase = DataBase
        self.executor = ThreadPoolExecutor()

    async def process_sub_query(self, sub_query: str, user_query: str) -> Optional[str]:
        """
        Processes each sub-query to get context either from KB or web.

        Args:
            sub_query: Individual sub-query generated from the main query.
            user_query: Original user query.

        Returns:
            A string containing the context for the sub-query.
        """
        try:
            agent_resp_raw = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.agent, sub_query
            )
            agent_response = json.loads(agent_resp_raw)
            kb = agent_response.get("knowledge_base", "NoKB")

            logging.info(f"Sub-query '{sub_query}' classified with KB: {kb}")

            if kb == "junaidh-text-NoKB":
                logging.info("Fallback to web search for sub-query")
                return await asyncio.get_event_loop().run_in_executor(
                    self.executor, self.web_search, sub_query
                )
            elif kb == "junaidh-text-DB":
                logging.info("Fallback to database for sub-query")
                return await asyncio.get_event_loop().run_in_executor(
                    self.executor, self.DataBase, sub_query
                )
            else:
                return await asyncio.get_event_loop().run_in_executor(
                    self.executor, self.vectorizer, user_query, kb
                )
        except json.JSONDecodeError:
            logging.error("500: Invalid agent response JSON")
        except Exception as e:
            logging.exception(f"500: Sub-query processing error - {e}")
        return None
    
    async def validate_result(self, user_query: str,  context: list[str], result):
        """
        Validate the LLM result
        """
        try:
            Result_validation = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.Validate, user_query, context, result
            )
            logging.info(f"validation result:{Result_validation}")
        except Exception as e:
            logging.exception(f"500: Validation error - {e}")
            return "Validation failed"

    async def handle_query(self, user_query: str) -> str:
        """
        Handles the entire pipeline of processing a user query.

        Args:
            user_query: The main query input from the user.

        Returns:
            Final string response from the RAG model.
        """
        try:
            queries_json = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.query_gen, user_query
            )
            queries = json.loads(queries_json).get("queries", [])

            if not queries:
                logging.warning("400: No sub-queries generated.")
                return "I'm sorry, I couldn't understand your question."

            tasks = [
                self.process_sub_query(sub_query, user_query) for sub_query in queries
            ]
            context = await asyncio.gather(*tasks)

            context = [c for c in context if c]

            if not context:
                logging.warning("404: No context found for any sub-query.")
                return "I couldn't find enough information to answer that."

            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.rag, user_query, context
            )
            await self.validate_result(user_query,context,result)
            return result

        except json.JSONDecodeError:
            logging.error("500: Invalid sub-query JSON")
            return "Something went wrong while understanding your question."
        except Exception as e:
            logging.exception(f"500: Error in query handling - {e}")
            return "Something went wrong on my side. Please try again."

    async def chat_loop(self):
        """
        Continuous chat loop to accept and respond to user queries.
        """
        print("Bot is ready. Type 'no' to exit.")
        while True:
            user_query = input("User: ").strip()
            if user_query.lower() == "no":
                print("Bot: Feel free to chat again!")
                break
            response = await self.handle_query(user_query)
            print("Bot:", response)

if __name__ == "__main__":
    try:
        chatbot = RAGChatBot(
            vectorizer=qstn_vectorize,
            agent=agent_model,
            rag=rag_model,
            web_search=search_DDG,
            DataBase = get_attendance_data,
            query_gen=query_generator,
            Validate=Result_validation
        )
        asyncio.run(chatbot.chat_loop())
    except KeyboardInterrupt:
        logging.info("Chatbot terminated by user.")
    except Exception as e:
        logging.critical(f"Fatal error: {e}", exc_info=True)
# --------------------------------------------------------------------------------------------------------------------------------------------

