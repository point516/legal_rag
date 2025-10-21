import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_cohere import CohereRerank

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded.")

prompt = PromptTemplate.from_template(
"""
     Ты — помощник-юрист. Отвечай ТОЛЬКО на основе контекста. 
     Если сведений недостаточно, прямо скажи об этом.
     Каждый ключевой тезис подкрепляй статьей/пунктом из контекста. 
     Формат ответа:
     1) Полный ответ
     2) Обоснование
     3) Источники (предоставь статью и цитату)

     Не используй markdown форматирование.
     
Контекст: {context}  

Вопрос: {input}  
"""
)

multiquery_prompt = PromptTemplate(
    input_variables=["question"],
    template=
    """
    Вы — помощник, работающий с языковой моделью ИИ. Ваша задача — сгенерировать 2 различные версии заданного пользователем вопроса для извлечения соответствующих документов из векторной базы данных.
    Векторная база данных содержит документы, связанные с правовыми и юридическими вопросами. В связи с этим, перефразируйте вопрос на более юридический язык, при этом сохраняя его смысл.
    Gредоставьте эти альтернативные вопросы, разделённые \n переносами. 
    
    Исходный вопрос: {question}
    """
)

# Initialize embeddings (must use the same embeddings model as when creating the index)
logger.info("Initializing embeddings model text-embedding-3-large.")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Load the saved FAISS index
persist_directory = "/Users/abylayturekhassim/Documents/coding/rag/faiss_index_admin_crim_civil_exec_proc_tax"
logger.info("Loading FAISS index from %s", persist_directory)
vectorstore = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
logger.info("FAISS index loaded successfully.")

################################################################################
####################### Initialize the Gemini-2.5-flash-lite ###################
################################################################################
logger.info("Initializing ChatGoogleGenerativeAI with model gemini-2.5-flash-latest-lite.")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2,
    google_api_key=os.getenv("GEMINI_API_KEY"),
    thinking_budget=0,
)
################################################################################
################################################################################
################################################################################


# Create the document chain
logger.info("Creating document combination chain.")
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
    document_separator="\n<break>\n"
)

################################################################################
####################### Initialize the Cohere Reranker #########################
################################################################################
cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_rerank_model = "rerank-v3.5"
cohere_top_n = 3
k_search = 5
################################################################################
################################################################################
################################################################################

logger.info("Preparing base retriever with k=%d.", k_search)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": k_search})

reranker = None
if cohere_api_key:
    logger.info(
        "Initializing Cohere reranker model %s with top_n=%d.",
        cohere_rerank_model,
        cohere_top_n,
    )
    reranker = CohereRerank(
        cohere_api_key=cohere_api_key,
        model=cohere_rerank_model,
        top_n=cohere_top_n,
    )
else:
    logger.warning(
        "COHERE_API_KEY not set. Proceeding without Cohere reranker.")

################################################################################
####################### Create the MultiQueryRetriever #########################
################################################################################
logger.info("Creating MultiQueryRetriever with base retriever.")
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    include_original = True,
    prompt=multiquery_prompt
)
################################################################################
################################################################################
################################################################################

# Example of how to use it
def get_response(query):
    # start_time = time.time()
    logger.info("Received query: %s", query)
    run_manager = CallbackManagerForRetrieverRun.get_noop_manager()

    generated_queries = multiquery_retriever.generate_queries(query, run_manager)

    if isinstance(generated_queries, str):
        queries_list = [q.strip() for q in generated_queries.split("\n") if q.strip()]
    else:
        queries_list = list(generated_queries)

    if multiquery_retriever.include_original:
        queries_list.append(query)

    logger.info("Generated queries: %s", queries_list)
    logger.info(
        "Retrieving documents with MultiQueryRetriever in parallel (queries=%d).",
        len(queries_list),
    )

    def _retrieve(query_text):
        return multiquery_retriever.retriever.invoke(query_text)
    
    ############################################################################
    ####################### Parallelizing the MultiQueryRetriever ##############
    ############################################################################
    max_workers = min(len(queries_list), (os.cpu_count() or 1) * 2)
    documents_collected = []
    with ThreadPoolExecutor(max_workers=max_workers or 1) as executor:
        future_to_query = {
            executor.submit(_retrieve, q): q for q in queries_list
        }
        for future in as_completed(future_to_query):
            query_text = future_to_query[future]
            try:
                docs = future.result()
                documents_collected.append(docs)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error retrieving documents for query '%s': %s", query_text, exc)
    ############################################################################
    ############################################################################
    ############################################################################

    retrieved_docs_all = [doc for docs in documents_collected for doc in docs]
    retrieved_docs = multiquery_retriever.unique_union(retrieved_docs_all)
    logger.info(
        "Retrieved %d documents before deduplication, %d after deduplication.",
        len(retrieved_docs_all),
        len(retrieved_docs),
    )
    # print metadata of retrieved_docs
    # for doc in retrieved_docs:
    #     logger.info("Metadata: %s", doc.metadata)
    logger.info("Retrieved %d documents before reranking.", len(retrieved_docs))

    if reranker:
        logger.info("Reranking documents with Cohere (top_n=%d).", cohere_top_n)
        query = " ".join(queries_list)
        logger.info("Query for reranking: %s", query)
        ranked_docs = reranker.compress_documents(retrieved_docs, query=query)
    else:
        logger.warning(
            "Reranker unavailable. Using first %d documents from retriever.",
            cohere_top_n,
        )
        ranked_docs = retrieved_docs[:cohere_top_n]

    logger.info("Using %d documents as final context for LLM.", len(ranked_docs))
    # end_time = time.time()
    # print(f"Time taken for retrieval: {end_time - start_time} seconds")

    # this is taking a long time
    answer_text = document_chain.invoke({
        "input": query,
        "context": ranked_docs,
    })

    sources = []
    for doc in ranked_docs:
        source_path = doc.metadata.get("source", "")
        source_name = source_path.split("\\")[-1].split("/")[-1]
        sources.append(source_name[:-4] if source_name.endswith(".pdf") else source_name)

    logger.info("Reporting %d source documents.", len(sources))

    # queries_section = "\nСгенерированные запросы:\n" + "\n".join(queries_list)
    sources_section = "Источники: " + ", ".join(sources) if sources else "Источники: —"
    answer_section = answer_text

    return f"{sources_section}\n\n{answer_section}"

# def chat_loop():
#     print("Start chatting (type 'quit' to exit)")
#     while True:
#         query = input("\nВаш вопрос: ")
#         if query.lower() == 'quit':
#             break
            
#         response = get_response(query)
#         print("\nОтвет: ", response)

# # Start the chat
# chat_loop()

# Example usage
if __name__ == "__main__":
    query = input("Ваш вопрос: ")
    start_time = time.time()
    response = get_response(query)
    end_time = time.time()
    print(f"Время выполнения: {end_time - start_time} секунд")
    print("\nОтвет: ", response)