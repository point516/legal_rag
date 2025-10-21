# Legal RAG

A Retrieval-Augmented Generation (RAG) system for answering user's legal questions by querying and analyzing the legal Codes of Law of Kazakhstan. The system uses parallelized multi-query semantic search with document reranking to retrieve relevant legal documents and generates contextual answers using LLMs. With Telegram Bot as interaction.

## Vector Database

The vector database consists of the codes of Law of Kazakhstan, processed through document processing scripts and **Mistral OCR** for parsing figures and tables. Documents are embedded using OpenAI's `text-embedding-3-large` model and indexed in FAISS for efficient retrieval.

## Files

- **main.py** - Core RAG system. Implements parallelized multi-query retrieval with document deduplication and custom Cohere reranking. Uses Google's Gemini 2.5 Flash Lite as the LLM to generate answers based on retrieved legal documents.

- **telegram_bot.py** - Telegram bot interface that provides access to the RAG system. Handles user queries through Telegram and returns responses split into appropriately sized messages.

- **requirements.txt** - Project dependencies including LangChain, FAISS, Telegram bot library, and API clients for OpenAI, Google Generative AI, and Cohere.
