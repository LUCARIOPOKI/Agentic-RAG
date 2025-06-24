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
# import json
# import logging
# from logging.handlers import RotatingFileHandler
# from multiprocessing import Pool, cpu_count
# from typing import Callable, List, Optional

# from index_search import qstn_vectorize
# from Query_routing_Agent import agent_model
# from rag_llm import rag_model
# from WebSearchTool import search_DDG
# from Query_receiver_Agent import query_generator

# file_handler = RotatingFileHandler(
#     filename="chatbot_logs.log",
#     maxBytes=5 * 1024 * 1024,
#     backupCount=3
# )
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[file_handler]
# )

# def sub_query_worker(args):
#     """
#     Worker function to process a single sub-query.
#     Runs inside a separate process.
#     """
#     sub_query, user_query = args
#     try:
#         agent_resp_raw = agent_model(sub_query)
#         agent_response = json.loads(agent_resp_raw)
#         kb = agent_response.get("knowledge_base", "NoKB")

#         logging.info(f"Sub-query '{sub_query}' classified with KB: {kb}")

#         if kb == "NoKB":
#             return search_DDG(sub_query)
#         else:
#             return qstn_vectorize(user_query, kb)

#     except json.JSONDecodeError:
#         logging.error(f"Invalid agent response JSON for sub-query '{sub_query}'")
#     except Exception as e:
#         logging.exception(f"Error in sub-query '{sub_query}': {e}")
#     return None

# class RAGChatBotMP:
#     def __init__(
#         self,
#         vectorizer: Callable,
#         agent: Callable,
#         rag: Callable,
#         web_search: Callable,
#         query_gen: Callable,
#     ):
#         self.vectorizer = vectorizer
#         self.agent = agent
#         self.rag = rag
#         self.web_search = web_search
#         self.query_gen = query_gen
#         self.pool = Pool(processes=cpu_count())  # Max available cores

#     def handle_query(self, user_query: str) -> str:
#         """
#         Synchronous method using multiprocessing to handle sub-queries.
#         """
#         try:
#             queries_json = self.query_gen(user_query)
#             queries = json.loads(queries_json).get("queries", [])

#             if not queries:
#                 logging.warning("No sub-queries generated.")
#                 return "I'm sorry, I couldn't understand your question."

#             args_list = [(q, user_query) for q in queries]

#             context = self.pool.map(sub_query_worker, args_list)
#             context = [c for c in context if c]

#             if not context:
#                 logging.warning("No valid context returned.")
#                 return "I couldn't find enough information to answer that."

#             return self.rag(user_query, context)

#         except json.JSONDecodeError:
#             logging.error("Invalid JSON during query generation.")
#             return "Something went wrong while understanding your question."
#         except Exception as e:
#             logging.exception(f"Error in handle_query: {e}")
#             return "Something went wrong on my side. Please try again."

#     def chat_loop(self):
#         print("Bot is ready. Type 'no' to exit.")
#         while True:
#             user_query = input("User: ").strip()
#             if user_query.lower() == "no":
#                 print("Bot: Feel free to chat again!")
#                 break
#             response = self.handle_query(user_query)
#             print("Bot:", response)


# if __name__ == "__main__":
#     try:
#         chatbot = RAGChatBotMP(
#             vectorizer=qstn_vectorize,
#             agent=agent_model,
#             rag=rag_model,
#             web_search=search_DDG,
#             query_gen=query_generator
#         )
#         chatbot.chat_loop()
#     except KeyboardInterrupt:
#         logging.info("Chatbot terminated by user.")
#     except Exception as e:
#         logging.critical(f"Fatal error: {e}", exc_info=True)


# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# import json
# import logging
# from logging.handlers import RotatingFileHandler
# from typing import List, Optional

# from joblib import Parallel, delayed

# # Import modular components
# from index_search import qstn_vectorize
# from Query_routing_Agent import agent_model
# from rag_llm import rag_model
# from WebSearchTool import search_DDG
# from Query_receiver_Agent import query_generator

# file_handler = RotatingFileHandler(
#     filename="chatbot_logs.log",
#     maxBytes=5 * 1024 * 1024,
#     backupCount=3
# )

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[file_handler]
# )


# class RAGChatBot:
#     def __init__(
#         self,
#         vectorizer,
#         agent,
#         rag,
#         web_search,
#         query_gen,
#     ):
#         self.vectorizer = vectorizer
#         self.agent = agent
#         self.rag = rag
#         self.web_search = web_search
#         self.query_gen = query_gen

#     def process_sub_query(self, sub_query: str, user_query: str) -> Optional[str]:
#         try:
#             agent_resp_raw = self.agent(sub_query)
#             agent_response = json.loads(agent_resp_raw)
#             kb = agent_response.get("knowledge_base", "NoKB")

#             logging.info(f"Sub-query '{sub_query}' classified with KB: {kb}")

#             if kb == "NoKB":
#                 logging.info("Fallback to web search for sub-query")
#                 return self.web_search(sub_query)
#             else:
#                 return self.vectorizer(user_query, kb)

#         except json.JSONDecodeError:
#             logging.error("500: Invalid agent response JSON")
#         except Exception as e:
#             logging.exception(f"500: Sub-query processing error - {e}")
#         return None

#     def handle_query(self, user_query: str) -> str:
#         try:
#             queries_json = self.query_gen(user_query)
#             queries = json.loads(queries_json).get("queries", [])

#             if not queries:
#                 logging.warning("400: No sub-queries generated.")
#                 return "I'm sorry, I couldn't understand your question."

#             context = Parallel(n_jobs=-1)(
#                 delayed(self.process_sub_query)(sub_query, user_query)
#                 for sub_query in queries
#             )

#             context = [c for c in context if c]

#             if not context:
#                 logging.warning("404: No context found for any sub-query.")
#                 return "I couldn't find enough information to answer that."

#             result = self.rag(user_query, context)
#             return result

#         except json.JSONDecodeError:
#             logging.error("500: Invalid sub-query JSON")
#             return "Something went wrong while understanding your question."
#         except Exception as e:
#             logging.exception(f"500: Error in query handling - {e}")
#             return "Something went wrong on my side. Please try again."

#     def chat_loop(self):
#         print("Bot is ready. Type 'no' to exit.")
#         while True:
#             user_query = input("User: ").strip()
#             if user_query.lower() == "no":
#                 print("Bot: Feel free to chat again!")
#                 break
#             response = self.handle_query(user_query)
#             print("Bot:", response)


# if __name__ == "__main__":
#     try:
#         chatbot = RAGChatBot(
#             vectorizer=qstn_vectorize,
#             agent=agent_model,
#             rag=rag_model,
#             web_search=search_DDG,
#             query_gen=query_generator
#         )
#         chatbot.chat_loop()
#     except KeyboardInterrupt:
#         logging.info("Chatbot terminated by user.")
#     except Exception as e:
#         logging.critical(f"Fatal error: {e}", exc_info=True)
# ---------------------------------------------------------------------------------------------------------------------------------------------
# import os
# import json
# import logging
# from threading import Thread
# from typing import Callable, List, Dict, Optional

# from index_search import qstn_vectorize
# from Query_routing_Agent import agent_model
# from rag_llm import rag_model
# from WebSearchTool import search_DDG
# from Query_receiver_Agent import query_generator
# from logging.handlers import RotatingFileHandler

# file_handler = RotatingFileHandler(
#     filename="chatbot_logs.log",
#     maxBytes=5 * 1024 * 1024,
#     backupCount=3
# )

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[file_handler]
# )


# class RAGChatBotThreaded:
#     def __init__(
#         self,
#         vectorizer: Callable,
#         agent: Callable,
#         rag: Callable,
#         web_search: Callable,
#         query_gen: Callable,
#     ):
#         self.vectorizer = vectorizer
#         self.agent = agent
#         self.rag = rag
#         self.web_search = web_search
#         self.query_gen = query_gen

#     def process_sub_query(self, sub_query: str, user_query: str, context_list: List, index: int):
#         """
#         Run agent and fetch content from appropriate source (KB or Web).
#         This function runs in a thread.
#         """
#         try:
#             agent_response = self.agent(sub_query)
#             kb = json.loads(agent_response).get("knowledge_base", "NoKB")

#             logging.info(f"[Thread-{index}] Sub-query '{sub_query}' classified with KB: {kb}")

#             if kb == "NoKB":
#                 result = self.web_search(sub_query)
#             else:
#                 result = self.vectorizer(user_query, kb)

#             context_list[index] = result
#         except Exception as e:
#             logging.exception(f"[Thread-{index}] Error processing sub-query: {e}")
#             context_list[index] = None

#     def handle_query(self, user_query: str) -> str:
#         """
#         Threaded version of query handling (blocking).
#         """
#         try:
#             queries_raw = self.query_gen(user_query)
#             queries = json.loads(queries_raw).get("queries", [])

#             if not queries:
#                 logging.warning("400: No sub-queries generated.")
#                 return "I'm sorry, I couldn't understand your question."

#             context = [None] * len(queries)
#             threads = []

#             for idx, sub_query in enumerate(queries):
#                 thread = Thread(
#                     target=self.process_sub_query,
#                     args=(sub_query, user_query, context, idx)
#                 )
#                 threads.append(thread)
#                 thread.start()

#             for thread in threads:
#                 thread.join()

#             # Filter out failed results
#             filtered_context = [c for c in context if c]

#             if not filtered_context:
#                 logging.warning("404: No context found for any sub-query.")
#                 return "I couldn't find enough information to answer that."

#             result = self.rag(user_query, filtered_context)
#             return result

#         except Exception as e:
#             logging.exception(f"500: Error handling query: {e}")
#             return "Something went wrong on my side. Please try again."

#     def chat_loop(self):
#         """
#         Blocking input/output chat loop.
#         """
#         print("Bot is ready. Type 'no' to exit.")
#         while True:
#             user_query = input("User: ").strip()
#             if user_query.lower() == "no":
#                 print("Bot: Feel free to chat again!")
#                 break
#             response = self.handle_query(user_query)
#             print("Bot:", response)


# if __name__ == "__main__":
#     try:
#         chatbot = RAGChatBotThreaded(
#             vectorizer=qstn_vectorize,
#             agent=agent_model,
#             rag=rag_model,
#             web_search=search_DDG,
#             query_gen=query_generator
#         )
#         chatbot.chat_loop()

#     except KeyboardInterrupt:
#         logging.info("Chatbot terminated by user.")
#     except Exception as e:
#         logging.critical(f"Fatal error: {e}", exc_info=True)
