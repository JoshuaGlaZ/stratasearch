import os
import mimetypes
from django.conf import settings
from django.urls import reverse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse, HttpResponse, Http404
from .services.rag import answer_question
import logging

logger = logging.getLogger(__name__)


def index(request):
    """Renders the main chat page."""
    data_dir = os.path.join(settings.BASE_DIR, 'data')
    template_questions = []
    
    subdomains_to_remove = ['www', 'docs', 'developer', 'dev', 'api']
    
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, item)):
                parts = item.split('.')
                if len(parts) > 1 and parts[0] in subdomains_to_remove:
                    friendly_name = parts[1].capitalize()
                else:
                    friendly_name = parts[0].capitalize()
                template_questions.append(friendly_name)
    
    context = {'template_questions': template_questions}
    return render(request, 'search/index.html', context)


@require_http_methods(["POST"])
def chat_message(request):
    """
    Endpoint:
    1. Receives user input
    2. Runs RAG pipeline
    3. Returns HTML fragment with metadata
    """
    user_input = request.POST.get('message', '').strip()
    
    if not user_input:
        return render(request, 'search/partials/message.html', {
            'question': user_input,
            'error': 'Please enter a message.',
            'ai_answer': '⚠️ **Error**: No message provided. Please try again.'
        })
        
    context = {'question': user_input}
        
    if 'chat_history' not in request.session:
        request.session['chat_history'] = []
    
    # Get the last 3 turns
    # Format: [(user, ai), (user, ai)]
    history = request.session['chat_history'][-3:]

    try:
        response_data = answer_question(user_input, history)
        
        request.session['chat_history'].append((user_input, response_data['answer']))
        request.session.modified = True
        
        logger.info(f"Query: {user_input[:50]}... ")

        processed_sources = []
        raw_sources = response_data.get('sources', [])
        data_dir = os.path.join(settings.BASE_DIR, 'data')

        for source in raw_sources:
            url = source['url']
            name = source['name']
            
            if url.startswith('http'):
                processed_url = url
            else:
                try:
                    relative_path = os.path.relpath(url, data_dir).replace('\\', '/')
                    processed_url = reverse('get_document', kwargs={'filename': relative_path})
                except ValueError:
                    # This can happen if the path is on a different drive on Windows
                    processed_url = '#'
            
            processed_sources.append({'name': name, 'url': processed_url})

        context = {
            'ai_answer': response_data['answer'],
            'sources': processed_sources,
        }
        
    except FileNotFoundError as e:
        logger.error(f"Database not found: {e}")
        context = {
            'ai_answer': '🔴 **System Error**: Vector database not found. Please run the ingestion script first.\n\n```bash\npython manage.py ingest_docs\n```',
            'sources': [],
            'error': str(e)
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        context = {
            'ai_answer': f'⚠️ **Processing Error**: {str(e)}\n\nPlease try rephrasing your question or contact support if the issue persists.',
            'sources': [],
            'error': str(e)
        }
    
    return render(request, 'search/partials/message.html', context)


@require_http_methods(["GET"])
def get_document_content(request, filename):
    """
    Fetches the content of a referenced document.
    """
    base_dir = os.path.join(settings.BASE_DIR, 'data')
    file_path = os.path.normpath(os.path.join(base_dir, filename))
    
    if not file_path.startswith(os.path.normpath(base_dir)):
        return JsonResponse({'error': 'Access denied.'}, status=403)

    if not os.path.exists(file_path):
        return JsonResponse({'error': 'File not found.'}, status=404)

    content_type, encoding = mimetypes.guess_type(file_path)
    if not content_type:
        content_type = 'text/plain'

    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
            return HttpResponse(file_content, content_type=content_type)
            
    except Exception as e:
        logger.error(f"Error serving {filename}: {e}", exc_info=True)
        return HttpResponse("Error reading file", status=500)