from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic
from .models import Area, Cafe, Utility

class IndexView(generic.ListView):
    template_name = 'list.html'
    context_object_name = 'cafe_list'
    def get_queryset(self):
        """Return the last five published records."""
        return Cafe.objects.order_by('name')[:5]
class DetailView(generic.DetailView):
    model = Cafe
    template_name = 'detail.html'