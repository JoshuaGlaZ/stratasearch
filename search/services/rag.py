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
from .prompts import get_template, CONDENSE_QUESTION_TEMPLATE

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
LLM_MODEL = os.getenv("LLM_MODEL")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.05"))

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "vector_db")


def format_docs(docs):
    return "\n\n".join([f"[Source: {os.path.basename(d.metadata.get('source', 'unknown'))}]\n{d.page_content}" for d in docs])


def format_history(history):
    return "\n".join([f"Human: {h}\nAI: {a}" for h, a in history])


def answer_question(question_text, history=[]):
    """
    RAG pipeline with History Awareness + MultiQuery + Reranking
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Vector Database not found at {DB_PATH}. Run 'ingest.py' first.")

    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)

    vector_db = FAISS.load_local(
        DB_PATH, embeddings, allow_dangerous_deserialization=True)

    # 1. Base Retriever
    retriever = vector_db.as_retriever(
        search_kwargs={"k": 8})

    # 2. Query Translation (Multi-Query)
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm
    )

    # 3. Re-Ranking
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_query_retriever
    )

    # 4. History Handling
    if history:
        condense_prompt = PromptTemplate.from_template(
            CONDENSE_QUESTION_TEMPLATE)
        condense_chain = (
            {"chat_history": lambda x: format_history(
                x), "question": lambda x: x}
            | condense_prompt
            | llm
            | StrOutputParser()
        )
        question = condense_chain.invoke(history)

    # 5. Execution
    retrieved_docs = compression_retriever.invoke(question)

    # 6. Answer Generation
    answer_prompt = ChatPromptTemplate.from_template(get_template())
    chain = (
        {"context": lambda x: format_docs(
            retrieved_docs), "question": lambda x: question}
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)

    sources = []
    seen = set()
    for doc in retrieved_docs:
        src = doc.metadata.get('source', 'unknown')
        if src not in seen:
            sources.append({'name': os.path.basename(src), 'url': src})
            seen.add(src)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": "high" if len(retrieved_docs) > 0 else "low"
    }
