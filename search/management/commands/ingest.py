import os
from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils import timezone
from rich.console import Console
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from search.models import ScrapedPage

console = Console()


class Command(BaseCommand):
    help = 'Ingests scraped data from the database into the Vector Store'

    def handle(self, *args, **options):
        load_dotenv()

        MODEL_NAME = os.getenv("MODEL_NAME")
        CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
        CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
        DATA_DIR_NAME = os.getenv("DATA_DIR_NAME")
        DB_DIR_NAME = os.getenv("DB_DIR_NAME")

        base_dir = getattr(settings, 'BASE_DIR', os.getcwd())
        data_path = os.path.join(base_dir, DATA_DIR_NAME)
        db_path = os.path.join(data_path, DB_DIR_NAME)

        os.makedirs(db_path, exist_ok=True)

        # 1. Load Documents from Database
        console.print(
            "[cyan]Loading pending documents from the database...[/cyan]")
        pending_pages = ScrapedPage.objects.filter(status='pending')

        if not pending_pages.exists():
            console.print(
                "[yellow]⚠️ No pending pages to ingest. Run the 'scrape' command first.[/yellow]")
            return

        documents = []
        for page in pending_pages:
            doc = Document(
                page_content=page.content,
                metadata={
                    "source": page.url,
                    "title": page.title,
                }
            )
            documents.append(doc)

        console.rule(
            f"[bold blue]Processing {len(documents)} Documents[/bold blue]")

        # 2. Text Splitting
        console.print(
            f"[cyan]Splitting text (Size={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP})...[/cyan]")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )

        chunks = text_splitter.split_documents(documents)
        console.print(f"[green]✔ Generated {len(chunks)} chunks.[/green]")

        # 3. Embedding & Indexing
        console.print(
            f"[purple]Embedding with {MODEL_NAME} (this may take time)...[/purple]")

        embeddings = OllamaEmbeddings(model=MODEL_NAME)

        try:
            if os.path.exists(db_path) and os.listdir(db_path):
                console.print(
                    "[cyan]Existing Vector DB found. Merging new documents...[/cyan]")
                vector_db = FAISS.load_local(
                    db_path, embeddings, allow_dangerous_deserialization=True)
                vector_db.add_documents(chunks)
            else:
                console.print("[cyan]Creating new Vector DB...[/cyan]")
                vector_db = FAISS.from_documents(chunks, embeddings)

            vector_db.save_local(db_path)

            # 4. Update status in database
            processed_ids = [page.id for page in pending_pages]
            ScrapedPage.objects.filter(id__in=processed_ids).update(
                status='processed',
                processed_at=timezone.now()
            )

            console.rule("[bold green]Ingestion Complete[/bold green]")
            console.print(f"Vector DB saved to: {db_path}")
            console.print(f"{len(processed_ids)} pages marked as processed.")

        except Exception as e:
            console.print(f"[bold red]❌ FAISS Error:[/bold red] {e}")
            console.print(
                "Ensure you have 'langchain-community' and 'faiss-cpu' installed.")
            processed_ids = [page.id for page in pending_pages]
            ScrapedPage.objects.filter(
                id__in=processed_ids).update(status='failed')
