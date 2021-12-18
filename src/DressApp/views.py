from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.http.response import JsonResponse
from .dress_lib import image_cut

from .form import ImageForm
from .models import Human_body

import cv2
import matplotlib.pyplot as plt
import json


#クライアントにHTMLデータを渡す
def index(request):
    #return HttpResponse("Hello World")
    form = ImageForm()
    context = {'form':form}
    return render(request, 'dressed.html', context)

def ajax_file_send(request):#ajaxでクライアントから画像をもらう
    with open("./DressApp/log.txt", mode='w') as f:
        f.write("kita")
    
    file = request.FILES['picture']
    d = {
        "img_url":"/static/app_images/dress_img/3_40_haikei.png"
    }
    if request.method == "POST":
        #クライアントから受け取ったモデルhuman_bodyに沿うデータと画像を引数にImageFormクラスを作る
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():# 入力された全てのデータが適切だったら
            #human_body = Human_body.objects.all()
            #form.save()# フォームクラス(ImageForm)の適切なデータ(id,name,height,picture)をdbに保存する

            # (id,name,height,picture)をdbに保存する
            human_body = Human_body()#modelクラスをインスタンス化
            human_body.human_name = request.POST['human_name']#human_nameを入れる
            human_body.height = request.POST['height']#heightを入れる
            human_body.picture = request.FILES['picture']#画像データを入れる
            human_body.save()#保存する
            with open("./DressApp/log.txt", mode='w') as f:# python側の処理が見えるようにログファイルに書き込み
                f.write("form.is_valid() = True")
                f.write("\n")
                f.write(str(form.cleaned_data))#formを介したdbへの追加方法だと、file名が重複変更前になる
                f.write("\n")
                '''f.write(human_body.id)# なぜかidが追加されない…error
                f.write("\n")'''
                f.write(human_body.human_name)
                f.write("\n")
                f.write(human_body.height)
                f.write("\n")
                f.write(str(human_body.picture).split('/')[1])
                f.write("\n")
                f.write(str(type(human_body.picture)))
            filename = str(human_body.picture).split('/')[1]

            #timestr,actual_img,human_segm,candidate_json = image_cut.image_cut(filename)# 画像処理開始
            human_segm = cv2.imread("./DressApp/dress_lib/materials/part_segms/test.png")#テストhuman_segm
            actual_img = cv2.imread("./media/cut_images/test.png")#テストactual_img
            candidate_json = '["foo",{"baz":["baz",null,1.0,34]}]'#テストcandidate_json

            #骨格情報JSONデータを保存
            skeleton_path='./DressApp/dress_lib/materials/skeleton_jsons/'+filename.split('.')[0]+".json"
            with open('./DressApp/dress_lib/materials/skeleton_jsons/'+filename.split('.')[0]+".json", 'w') as fp:
                json.dump(candidate_json, fp, indent=4, ensure_ascii=False)
            
            #人物部位セグメンテーション画像を保存
            human_segm_path="./DressApp/dress_lib/materials/part_segms/"+filename
            cv2.imwrite("./DressApp/dress_lib/materials/part_segms/"+filename,human_segm)
            #背景削除画像を保存
            actual_img_path="./media/cut_images/"+filename
            cv2.imwrite("./media/cut_images/"+filename,actual_img)

            return JsonResponse(d)
        else:
            with open("./DressApp/log.txt", mode='w') as f:
                f.write("form.is_valid() = False")
                f.write(str(form.errors))
    else:
        form = ImageForm()

    context = {'form':form}

    return JsonResponse(d)

def showall(request):
    images =  Human_body.objects.all()
    context = {'images':images}
    return render(request, 'showall.html', context)

def upload(request):
    with open("./DressApp/log.txt", mode='w') as f:
        f.write("kita")
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            with open("./DressApp/log.txt", mode='w') as f:
                f.write("form.is_valid() = True")
            form.save()
            return redirect('DressApp:showall')
        else:
            print(type(form.errors))
            with open("./DressApp/log.txt", mode='w') as f:
                f.write("form.is_valid() = False")
                f.write(str(form.errors))
    else:
        form = ImageForm()

    context = {'form':form}
    return render(request, 'dressed.html', context)

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