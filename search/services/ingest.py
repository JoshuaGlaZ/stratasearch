import os
from dotenv import load_dotenv

from rich.console import Console
from rich.status import Status

from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

console = Console()

MODEL_NAME = os.getenv("MODEL_NAME")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
DATA_DIR_NAME = os.getenv("DATA_DIR_NAME")

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, DATA_DIR_NAME)
DB_PATH = os.path.join(BASE_DIR, "data", "vector_db")


def load_scraped_document(file_path):
    """
    Loader for scraped .txt files.
    Extracts URL from the first line and uses it as the source.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Extract URL from the first line (e.g., "URL: http://...")
    url_line = lines[0].strip()
    source_url = url_line.split(
        "URL: ")[1] if url_line.startswith("URL:") else file_path

    try:
        content_start_index = lines.index("---\n") + 2
    except ValueError:
        try:
            content_start_index = lines.index("--- CONTENT ---\n") + 1
        except ValueError:
            content_start_index = 1

    content = "".join(lines[content_start_index:])

    return Document(page_content=content, metadata={"source": source_url})


def ingest_docs():
    console.clear()
    console.rule("[bold blue]Stratasearch Ingestion Engine[/bold blue]")

    with console.status("[bold green]Initializing system...[/bold green]", spinner="dots") as status:

        # --- Check Paths ---
        if not os.path.exists(DATA_PATH):
            console.print(
                f"[bold red]❌ Error:[/bold red] Data directory not found at: {DATA_PATH}")
            return

        # --- Load Documents ---
        status.update(
            f"[bold yellow]Scanning '{DATA_DIR_NAME}' for documents...[/bold yellow]")
        documents = []
        all_files_to_ingest = []
        for root, _, files in os.walk(DATA_PATH):
            for file_name in files:
                if file_name.endswith((".pdf", ".txt", ".html")):
                    all_files_to_ingest.append(os.path.join(root, file_name))

        if not all_files_to_ingest:
            console.print(
                "[bold red]❌ No documents found.[/bold red] Please add PDFs or run the scraper!")
            return

        for full_path_to_file in all_files_to_ingest:
            status.update(
                f"[bold cyan]Processing:[/bold cyan] {os.path.basename(full_path_to_file)}")

            try:
                if full_path_to_file.endswith(".pdf"):
                    loader = PyPDFLoader(full_path_to_file)
                    documents.extend(loader.load())
                elif full_path_to_file.endswith(".txt"):
                    doc = load_scraped_document(full_path_to_file)
                    documents.append(doc)
                elif full_path_to_file.endswith(".html"):
                    loader = BSHTMLLoader(full_path_to_file, open_encoding='utf-8')
                    documents.extend(loader.load())
            except Exception as e:
                console.print(
                    f"[red]⚠️  Skipping {os.path.basename(full_path_to_file)}:[/red]")
                console.print_exception(show_locals=True)

        console.print(
            f"[green]✔ Loaded {len(documents)} documents.[/green]")

        # --- Split Text ---
        status.update(
            "[bold yellow]Splitting text into chunks...[/bold yellow]")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        texts = text_splitter.split_documents(documents)
        console.print(
            f"[green]✔ Created {len(texts)} searchable chunks.[/green]")

        # --- Embed & Save ---
        status.update(
            f"[bold purple]Embedding with {MODEL_NAME} (this may take a while)...[/bold purple]")
        embeddings = OllamaEmbeddings(model=MODEL_NAME)

        vector_db = FAISS.from_documents(texts, embeddings)
        vector_db.save_local(DB_PATH)

    console.rule("[bold green]Ingestion Complete[/bold green]")
    console.print(f"Database saved to: [underline]{DB_PATH}[/underline]")


if __name__ == "__main__":
    try:
        ingest_docs()
    except Exception as e:
        console.print_exception()
