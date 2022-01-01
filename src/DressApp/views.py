from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.core import serializers
from django.http.response import JsonResponse
from .dress_lib import image_cut
from .dress_lib.change_clothes_lib import clothes_on_top

from .form import ImageForm
from .models import Human_body
from .models import Human_clothes

import cv2
import matplotlib.pyplot as plt
import json
import time
import numpy as np
import os

use_id = None # intデータが必ず入る

#クライアントにHTMLデータを渡す
def index(request):
    #return HttpResponse("Hello World")
    images =  Human_body.objects.all()
    form = ImageForm()
    context = {'form':form,'images':images}
    global use_id
    use_id = None
    return render(request, 'dressed.html', context)

def click_cut(request):
    global use_id
    use_id = request.POST.get('use_id')
    d={
        "chat":"データを送信しました",
    }
    return JsonResponse(d)

def ajax_file_send(request):#ajaxでクライアントから画像をもらう
    start = time.time()
    topProc_start = time.time()
    
    #file = request.FILES['picture']
    if request.method == "POST":
        #クライアントから受け取ったモデルhuman_bodyに沿うデータと画像を引数にImageFormクラスを作る
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():# 入力された全てのデータが適切だったら
            #human_body = Human_body.objects.create(...)
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
                f.write(str(human_body.id))# idはstr型に変換しないと書き込みエラー
                f.write("\n")
                f.write(human_body.human_name)
                f.write("\n")
                f.write(human_body.height)
                f.write("\n")
                f.write(str(human_body.picture).split('/')[1])
                f.write("\n")
                f.write(str(type(human_body.picture)))
                f.write("\n")
                '''all = Human_body.objects.all()
                for one in all:
                    f.write(one.human_name)
                    f.write(" , ")
                    f.write(str(one.height))
                    f.write(" , ")
                    f.write(str(one.picture))
                    f.write(" , ")
                    f.write(str(one.id))
                    f.write("\n")'''

            input_img_path = "./media/"+str(human_body.picture)# mediaディレクトリからのパスにする
            filename = str(human_body.picture).split('/')[1]# human_img/xxxからファイル名xxxを抽出

            topProc_time = time.time() - topProc_start
            with open("./DressApp/log.txt", mode='a') as f:# python側の処理が見えるようにログファイルに書き込み
                f.write("topProc_time:{0}".format(topProc_time) + "[sec]")
                f.write("\n")
            filename,timestr,actual_img,human_segm,candidate_json = image_cut.image_cut(filename,input_img_path,int(human_body.height))# 画像処理開始
            bottomProc_start = time.time()
            with open("./DressApp/log.txt", mode='a') as f:# python側の処理が見えるようにログファイルに書き込み
                f.write(timestr)
                f.write("\n")
            #human_segm = cv2.imread("./DressApp/dress_lib/materials/part_segms/test.png")#テストhuman_segm
            #actual_img = cv2.imread("./media/cut_images/test.png")#テストactual_img
            #candidate_json = '["foo",{"baz":["baz",null,1.0,34]}]'#テストcandidate_json

            #骨格情報JSONデータを保存
            skeleton_path='./DressApp/dress_lib/materials/skeleton_jsons/'+filename.split('.')[0]+".json"
            with open('./DressApp/dress_lib/materials/skeleton_jsons/'+filename.split('.')[0]+".json", 'w') as fp:
                json.dump(candidate_json, fp, indent=4, ensure_ascii=False)
            
            #人物部位セグメンテーション画像を保存
            human_segm_path="./DressApp/dress_lib/materials/part_segms/"+filename
            cv2.imwrite("./DressApp/dress_lib/materials/part_segms/"+filename,human_segm)
            #背景削除画像を保存
            actual_img_path="/media/cut_images/"+filename
            cv2.imwrite("./media/cut_images/"+filename,actual_img)

            human_body.cut_image_path = actual_img_path#背景切り抜き画像があるパス
            human_body.part_segm_path = human_segm_path#人物部位セグム画像があるパス
            human_body.skeleton_json_path = skeleton_path#骨格jsonがあるパス
            human_body.save()#更新
            with open("./DressApp/log.txt", mode='a') as f:# python側の処理が見えるようにログファイルに書き込み
                f.write(str(human_body.id))
                f.write("\n")
                f.write(human_body.cut_image_path)
                f.write("\n")
                f.write(human_body.part_segm_path)
                f.write("\n")
                f.write(human_body.skeleton_json_path)
                f.write("\n")
                f.write(timestr)
                f.write("\n")

            
            images =  Human_body.objects.all()
            json_encode = list(images.values())
            #下でもquerysetからjsonで扱える形式にできる
            #images =  Human_body.objects.all()
            #images = serializers.serialize('json', Human_body.objects.all())
            #images = json.loads(images)
            d = {
                "queryset":json_encode
            }
            bottom_time = time.time() - bottomProc_start
            elapsed_time = time.time() - start
            with open("./DressApp/log.txt", mode='a') as f:# python側の処理が見えるようにログファイルに書き込み
                f.write("bottom_time:{0}".format(bottom_time) + "[sec]")
                f.write("\n")
                f.write("elapsed_time:{0}".format(elapsed_time) + "[sec]")
                f.write("\n")
            return JsonResponse(d)
        else:
            with open("./DressApp/log.txt", mode='w') as f:
                f.write("form.is_valid() = False")
                f.write(str(form.errors))
    else:
        form = ImageForm()

    context = {'form':form}
    d = {
        "img_url":"/static/app_images/dress_img/3_40_haikei.png"
    }
    return JsonResponse(d)

def select_clothes(request):
    part_clothes = int(request.POST.get('part_clothes'))
    clothes_name = request.POST.get('clothes_name')
    send_use_id = request.POST.get('use_id')
    global use_id
    d={
        "chat":"データを送信できましたが、処理中にエラーが起きました",
        "flag":"failure",
    }
    if send_use_id == use_id:# 不整合性回避
        human_body =  Human_body.objects.get(id=use_id)#use_idでデータベースからHuman_bodyのデータを取得
        human_name = human_body.human_name#human_nameを取得
        height = human_body.height#heightを取得
        picture_path = str(human_body.picture)#pictureを取得
        cut_image_path = human_body.cut_image_path#cut_image_pathを取得
        part_segm_path = human_body.part_segm_path#part_segm_pathを取得
        skeleton_json_path = human_body.skeleton_json_path#skeleton_json_pathを取得
        # 既に着せ替えたことがあるデータかどうかを確かめる
        already_list = Human_clothes.objects.all().filter(human_id=use_id, part_clothes=part_clothes, clothes_name=clothes_name)
        already_count = already_list.count()# 既に着せ替えたことがあるデータの数
        if already_count == 0:# 既に着せ替えたことがあるデータがないなら
            actual_img = cv2.imread("."+cut_image_path,-1)
            human_segm = cv2.imread(part_segm_path,0)
            with open(skeleton_json_path) as f:
                candidate_txt = json.load(f)#strになる 元々dict型にするメソッドだが、dictになるjsonデータでないと、str型になる
                print(type(candidate_txt))
                candidate = json.loads(candidate_txt)#listになる
                print(type(candidate))
                candidate = np.array(candidate)#ndarrayになる
                print(type(candidate))
            result_img = None
            if part_clothes == 1:
                import traceback
                try:
                    result_img,brank_img = clothes_on_top.change(actual_img,human_segm,candidate,clothes_name)
                except:
                    with open("./DressApp/log.txt", mode='a') as f:# python側の処理が見えるようにログファイルに書き込み
                        f.write("---point error---\n")
                        f.write(str(traceback.format_exc()))
                        f.write("\n")
                    return JsonResponse(d)
            match_clothes_path = "./DressApp/dress_lib/materials/match_clothes/"+str(use_id)+"_"+str(part_clothes)+"_"+clothes_name+".png"
            cv2.imwrite(match_clothes_path,brank_img)

            human_clothes = Human_clothes()
            human_clothes.human_id = use_id
            human_clothes.part_clothes = part_clothes
            human_clothes.clothes_name = clothes_name
            human_clothes.match_clothes_path = match_clothes_path
            human_clothes.save()#保存する
        else:#着せ替えたことがあるなら、match_clothes_pathからbrank_imgを取得しすぐにmounting
            actual_img = cv2.imread("."+cut_image_path,-1)
            brank_img = cv2.imread(already_list[0].match_clothes_path,-1)
            result_img = clothes_on_top.mounting(actual_img,brank_img)
        
        cv2.imwrite("./media/dress_images/dress.png",result_img)
        d={
            "chat":"着せ替えを行いました",
            "flag":"success",
            "id":use_id,
            "human_name":human_name,
            "height":height,
            "picture_path":picture_path,
            "cut_image_path":cut_image_path,
            "part_segm_path":part_segm_path,
            "skeleton_json_path":skeleton_json_path,
            "result_img_path":"/media/dress_images/dress.png"
        }

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