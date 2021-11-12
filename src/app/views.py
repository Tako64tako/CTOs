from django.http import HttpResponse, HttpResponseRedirect
from django.http.response import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic
from .models import Cloth

#クライアントにHTMLデータを渡す
def index(request):
    #return HttpResponse("Hello World")
    return render(request, 'chat.html')

#顧客の情報を受け取り、メッセージを返す
def send_clother(request):
    clother = Cloth.objects.all()
    context = {
        'index': "CLOTH",
        'clothers': clother
    }
    return render(request, 'list.html', context)


