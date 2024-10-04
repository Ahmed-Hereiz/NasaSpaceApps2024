from django.urls import path
from .views import *

urlpatterns = [
    path('chatbot/', ChatBotView, name='chatbot'),
    path('chat', chat, name='chat'),
]
