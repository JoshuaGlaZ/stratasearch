import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_compressors import FlashrankRerank
from .prompts import get_template, CONDENSE_QUESTION_TEMPLATE, HYDE_TEMPLATE

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
LLM_MODEL = os.getenv("LLM_MODEL")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.05"))

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "vector_db")


def format_docs(docs):
    return "\n\n".join([f"[Source: {os.path.basename(d.metadata.get('source', 'unknown'))}]\n{d.page_content}" for d in docs])


def format_chat_history(chat_history):
    buffer = []
    for human, ai in chat_history:
        buffer.append(f"Human: {human}")
        buffer.append(f"Assistant: {ai}")
    return "\n".join(buffer)


def answer_question(question_text, chat_history=[]):
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Vector DB not found at {DB_PATH}")

    # 1. Initialize Models & Database
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)
    
    vector_db = FAISS.load_local(
        DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
    base_retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    # 2. Define History Awareness Chain
    # This chain handles conversation history (e.g., "What about async?" -> "How do I use async sessions?")
    condense_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)
    condense_chain = (
        {
            "chat_history": lambda x: format_chat_history(x["chat_history"]),
            "question": itemgetter("question")
        }
        | condense_prompt
        | llm
        | StrOutputParser()
    )

    # 3. Define HyDE Generator (Hypothetical Document Embeddings)
    # This chain hallucinates a "fake" answer to better match vector embeddings
    hyde_prompt = PromptTemplate.from_template(HYDE_TEMPLATE)
    hyde_generator = (
        hyde_prompt
        | llm
        | StrOutputParser()
    )

    def hyde_retrieval(inputs):
        """Helper function to run the HyDE logic"""
        original_q = inputs["question"]

        # Step A: Generate a hypothetical document
        hypothetical_doc = hyde_generator.invoke({"question": original_q})
        print(f"DEBUG: HyDE Doc Generated: {hypothetical_doc[:100]}...")

        # Step B: Retrieve using the HYPOTHETICAL text (better semantic match)
        docs = base_retriever.invoke(hypothetical_doc)
        return docs

    # 4. Initialize Reranker (FlashRank)
    # This model will re-score the top 10 results to ensure accuracy
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=5)

    # 5. Execute: Resolve Effective Query
    # If history exists, rewrite the question. Otherwise, use the raw input.
    if chat_history:
        effective_query = condense_chain.invoke(
            {"question": question_text, "chat_history": chat_history})
    else:
        effective_query = question_text

    print(f"DEBUG: Effective Query: {effective_query}")

    # 6. Execute: HyDE Retrieval
    # We fetch documents that look like the "Modern" hypothetical answer
    initial_docs = hyde_retrieval({"question": effective_query})

    # 7. Execute: Reranking
    # We filter the initial 10 docs down to the best 5 based on the user's ACTUAL query
    reranked_docs = compressor.compress_documents(
        documents=initial_docs, query=effective_query)

    # 8. Execute: Final Answer Generation
    # We feed the highly relevant docs + the effective query to the LLM
    answer_prompt = ChatPromptTemplate.from_template(get_template())
    chain = (
        {"context": lambda x: format_docs(
            reranked_docs), "question": lambda x: effective_query}
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(effective_query)

    # 9. Extract Sources
    sources = []
    seen = set()
    for doc in reranked_docs:
        src = doc.metadata.get('source', 'unknown')
        if src not in seen:
            sources.append({'name': os.path.basename(src), 'url': src})
            seen.add(src)

    return {
        "answer": answer,
        "sources": sources,
        "source_documents": reranked_docs,
        "confidence": "high",
        "query_type": "HyDE"
    }