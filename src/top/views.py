from django.shortcuts import render
from django.views.generic import TemplateView
from top.models import Cloth
from django.db.models import Q


class IndexView(TemplateView):
    template_name = 'index.html'

def list_cloth(request):
    clother = Cloth.objects.all()
    context = {
        'index':"CLOTH",
        'clothers':clother
    }
    return render(request,'top/templates/list.html',context)

