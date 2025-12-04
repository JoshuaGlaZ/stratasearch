import os
import pandas as pd
from django.core.management.base import BaseCommand
from rich.console import Console
from rich.table import Table
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_ollama import ChatOllama, OllamaEmbeddings
from search.services.rag import answer_question

console = Console()

MODEL_NAME = os.getenv("MODEL_NAME")
LLM_MODEL = os.getenv("LLM_MODEL")

class Command(BaseCommand):
    help = 'Runs RAGAS evaluation metrics on the current RAG pipeline'

    def handle(self, *args, **kwargs):
        console.rule("[bold purple]RAGAS Evaluation Suite[/bold purple]")
        
        test_questions = [
            "How do I define a User model in SQLAlchemy 2.0?",
            "What is the difference between session.execute() and session.query()?",
            "How do I create an async engine with aiosqlite?",
            "Explain how to use mapped_column with type hints.",
            "What happened to the old declarative_base() in the new version?"
        ]
        
        console.print(f"[yellow]🧪 Running inference on {len(test_questions)} test questions...[/yellow]")

        data_samples = {
            'question': [],
            'answer': [],
            'contexts': [],
        }

        for query in test_questions:
            try:
                result = answer_question(query) 
                
                # Extract text content from the Document objects
                contexts = [doc.page_content for doc in result.get('source_documents', [])]
                
                data_samples['question'].append(query)
                data_samples['answer'].append(result['answer'])
                data_samples['contexts'].append(contexts)
                
                console.print(f"[green]✔[/green] Processed: {query[:30]}...")
            except Exception as e:
                console.print(f"[red]❌ Error on '{query}': {e}[/red]")

        evaluator_llm = ChatOllama(model=LLM_MODEL, temperature=0)
        evaluator_embeddings = OllamaEmbeddings(model=MODEL_NAME)

        dataset = Dataset.from_dict(data_samples)

        console.print("\n[blue]🧠 Calculating Metrics (Faithfulness & Relevance)...[/blue]")
        
        results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )

        final_scores = results
        
        table = Table(title="RAG Evaluation Report")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="magenta")

        
        f_score = final_scores['faithfulness']
        ar_score = final_scores['answer_relevancy']

        if isinstance(f_score, list):
             import numpy as np
             f_score = np.mean(f_score)
        
        if isinstance(ar_score, list):
             import numpy as np
             ar_score = np.mean(ar_score)

        f_display = f"{f_score:.4f}" if f_score == f_score else "N/A"
        ar_display = f"{ar_score:.4f}" if ar_score == ar_score else "N/A"

        table.add_row("Faithfulness", f_display)
        table.add_row("Answer Relevancy", ar_display)
        
        console.print(table)
        console.print(f"\n[bold]Raw Data:[/bold] {final_scores}")