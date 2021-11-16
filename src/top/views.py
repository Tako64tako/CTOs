from django.shortcuts import render
from django.views.generic import ListView
from .models import Cloth

class IndexView(ListView):
    template_name = 'index.html'
    model = Cloth
    context_object_name = 'garment_records'

