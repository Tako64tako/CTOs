from django.shortcuts import render
from django.http import HttpResponse
from django.http.response import JsonResponse
from .dress_lib import image_cut
import cv2
import matplotlib.pyplot as plt


#クライアントにHTMLデータを渡す
def index(request):
    #return HttpResponse("Hello World")
    return render(request, 'dressed.html')

def ajax_file_send(request):
    file = request.FILES['uploadfile']
    #imgmg = cv2.imread(file)
    #cv2.imwrite("ajax.png",imgmg)
    print("OK")
    d = {
        "img_url":"/static/app_images/dress_img/3_40_haikei.png"
    }
    #image_cut.image_cut()
    return JsonResponse(d)



def chat(request):
    #return HttpResponse("Hello World")
    id = 'bler_Hito_risize.png'
    img = cv2.imread('./DressApp/dress_lib/indexnet_matting/examples/images/'+id)
    cv2.imwrite('./DressApp/test_pp.png',img)
    return render(request, 'chat.html')

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