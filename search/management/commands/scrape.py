import os
import time
import requests
from urllib.parse import urljoin, urlparse, urldefrag
from bs4 import BeautifulSoup
from django.core.management.base import BaseCommand
from django.utils import timezone
from rich.console import Console
from search.models import ScrapedPage

console = Console()


class Command(BaseCommand):
    help = 'Scrapes documentation for RAG ingestion'

    def add_arguments(self, parser):
        parser.add_argument('url', type=str, help='Target URL to scrape')
        parser.add_argument('--depth', type=int, default=3,
                            help='Max crawl depth')
        parser.add_argument('--max', type=int, default=50,
                            help='Max pages to scrape')

    def handle(self, *args, **options):
        start_url = options['url']
        max_pages = options['max']
        depth_limit = options['depth']

        self.crawl(start_url, max_pages, depth_limit)

    def clean_content(self, soup):
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript']):
            tag.decompose()

        title = soup.title.string.strip() if soup.title else "Untitled"
        text = soup.get_text(separator='\n', strip=True)

        code_blocks = len(soup.find_all('pre'))

        return title, text, code_blocks

    def save_page_to_db(self, url, title, content):
        try:
            obj, created = ScrapedPage.objects.update_or_create(
                url=url,
                defaults={
                    'title': title,
                    'content': content,
                    'status': 'pending',
                    'processed_at': None,
                    'scraped_at': timezone.now()
                }
            )
            return created
        except Exception as e:
            console.print(f"[bold red]❌ Database Error:[/bold red] {e}")
            return None

    def crawl(self, start_url, max_pages, depth_limit):
        visited_urls = set(ScrapedPage.objects.values_list('url', flat=True))
        queue = [(start_url, 0)]
        base_domain = urlparse(start_url).netloc
        pages_scraped = 0

        console.rule(f"[bold cyan]🕷️ Starting Crawl: {start_url}[/bold cyan]")

        while queue and pages_scraped < max_pages:
            url, depth = queue.pop(0)
            url, _ = urldefrag(url)

            if url in visited_urls or depth > depth_limit or not url.startswith('http'):
                continue

            if urlparse(url).netloc != base_domain:
                continue

            try:
                console.print(f"[dim]Fetching:[/dim] {url} (Depth: {depth})")
                resp = requests.get(
                    url, headers={'User-Agent': 'StrataSearch-Bot/1.0'}, timeout=10)

                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.content, 'html.parser')
                    title, content, code_count = self.clean_content(soup)

                    if len(content.split()) > 50:
                        created = self.save_page_to_db(url, title, content)
                        if created is not None:
                            action = "[green]✔ Saved[/green]" if created else "[blue]🔄 Updated[/blue]"
                            console.print(
                                f"{action} ({code_count} code blocks): {title}")
                            pages_scraped += 1
                        else:
                            console.print(
                                f"[red]❌ Failed to save:[/red] {title}")
                    else:
                        console.print(
                            "[yellow]⏭️  Skipped (Low Content)[/yellow]")

                    visited_urls.add(url)

                    if depth < depth_limit:
                        for a in soup.find_all('a', href=True):
                            link = urljoin(url, a['href'])
                            if link not in visited_urls:
                                queue.append((link, depth + 1))

                time.sleep(0.5)

            except Exception as e:
                console.print(f"[red]❌ Error:[/red] {e}")

        console.rule(
            f"[bold green]Crawl Complete: Scraped {pages_scraped} new/updated pages.[/bold green]")
