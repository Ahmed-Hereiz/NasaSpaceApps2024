from django.urls import path
from .views import *

urlpatterns = [
    path('solution/', solutionView, name="solution"),
    path('home/', homeView, name="home"),
    path('about/', aboutView, name="about"),
]