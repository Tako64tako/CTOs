from django.http import HttpResponse, HttpResponseRedirect
from django.http.response import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic


#クライアントにHTMLデータを渡す
def index(request):
    return HttpResponse("Hello World")
    #return render(request, 'chat.html')

#顧客の情報を受け取り、メッセージを返す
def send_customer(request):
    gender = request.POST.get('gender')
    height = request.POST.get('height')
    type = request.POST.get('type')
    color = request.POST.get('color')
    d={
        "chat":"データを送信しました",
    }
    return JsonResponse(d)