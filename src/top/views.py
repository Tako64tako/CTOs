from django.shortcuts import render
from django.views.generic import ListView
from .models import Cloth
from django.db.models import Q, query

class IndexView(ListView):
    template_name = 'index.html'
    # model = Cloth
    context_object_name = 'garment_records'
    #queryset = Cloth.objects.filter(color="黒")

    def get_queryset(self):
        queryset = Cloth.objects.order_by('-number')
        keyword = self.request.GET.get('q')

        if keyword:
            queryset = queryset.filter(
                Q(color__icontains=keyword) | Q(kind__icontains=keyword) |
                Q(name__icontains=keyword) | Q(size__icontains=keyword) |
                Q(gender__icontains=keyword)
            )

        return queryset


