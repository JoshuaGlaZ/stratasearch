from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('chat/', views.chat_message, name='chat'),
    path('document/<path:filename>/', views.get_document_content, name='get_document'),
]