from django.http import HttpResponse, HttpResponseRedirect
from django.http.response import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic
import app.models
import app.forms
import uuid
from django.contrib.auth.models import User
import re
from django.contrib.auth import authenticate, login as django_login
from django.contrib.auth import authenticate, logout as django_logout
import django.http

#クライアントにHTMLデータを渡す
def index(request):
    #return HttpResponse("Hello World")
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




def has_digit(text):
    if re.search("\d", text):
        return True
    return False

def has_alphabet(text):
    if re.search("[a-zA-Z]", text):
        return True
    return False


def post_new_post(request):
    if request.method == 'POST':
        form = app.forms.InputForm(request.POST)
        if form.is_valid():
            app.models.Post.objects.create(name=request.POST['name'], age=request.POST['age'], comment=request.POST['comment'])
            return django.http.HttpResponseRedirect('/list')
    else:
        form = app.forms.InputForm()
    return render(request, 'post_new_post.html', {'form': form})

def list(request):
    posts = app.models.Post.objects.all()
    return render(request, 'list.html', {'posts': posts}) 

def login_user(request):
    if request.method == 'POST':
        login_form = app.forms.LoginForm(request.POST)
        username = login_form.username
        password = login_form.password
        user = authenticate(request, username=username, password=password)
        if user is not None:
            django_login(request, user)
            return django.http.HttpResponseRedirect('/new_post')
        else:
            login_form.add_error(None, "ユーザー名またはパスワードが異なります。")
            return render(request, 'login.html', {'login_form': login_form})
        return render(request, 'login.html', {'login_form': login_form})
    else:
        login_form = app.forms.LoginForm()
    return render(request, 'login.html', {'login_form': login_form})
    #アカウントとパスワードが合致したら、その人専用の投稿画面に遷移する
    #アカウントとパスワードが合致しなかったら、エラーメッセージ付きのログイン画面に遷移する
    

def registation_user(request):
    if request.method == 'POST':
        registration_form = app.forms.RegistrationForm(request.POST)
        password = request.POST['password']
        if len(password) < 8:
            registration_form.add_error('password', "文字数が8文字未満です。")
        if not has_digit(password):
            registration_form.add_error('password',"数字が含まれていません")
        if not has_alphabet(password):
            registration_form.add_error('password',"アルファベットが含まれていません")
        if registration_form.has_error('password'):
            return render(request, 'registration.html', {'registration_form': registration_form})
        user = User.objects.create_user(username=request.POST['username'], password=password, email=request.POST['email'])
        return django.http.HttpResponseRedirect('/login')
    else:
        registration_form = app.forms.RegistrationForm()
    return render(request, 'registration.html', {'registration_form': registration_form})

def change_password(request):
    if not request.user.is_authenticated: # このURLに移動した時点でログイン認証されていなかった場合
        return django.http.HttpResponseRedirect('/login') # ログイン画面に移動します
    if request.method == 'POST': # ログイン認証がされていて、POSTメソッドが実行された場合
        change_password_form = app.forms.ChangePasswordForm(request.POST) 
        new_password = request.POST['new_password']
        if len(new_password) < 8:
            change_password_form.add_error('new_password', "文字数が8文字未満です。")
        if not has_digit(new_password):
            change_password_form.add_error('new_password',"数字が含まれていません")
        if not has_alphabet(new_password):
            change_password_form.add_error('new_password',"アルファベットが含まれていません")
        if change_password_form.has_error('new_password'):
            return render(request, 'change_password.html', {'change_password_form': change_password_form}) # 新しいパスワードが8文字以上で英字数字含むことを満たしていない場合、パスワード変更画面に戻ります。
        user = request.user # userにリクエストを送ったユーザーの情報を代入します。ログインされているユーザーです。
        user.set_password(request.POST['new_password']) # set_password(password)でそのユーザーのパスワードを引数のパスワードに変更します。
        user.save() # save()でそのユーザーの情報の変更を保存します。
        # なお、set_password(password)でパスワードを変えると、自動的にログアウトされる動きをする
        return django.http.HttpResponseRedirect('/changed_password') # 変更したら、changed_passwordに移動します。
    else:
        change_password_form = app.forms.ChangePasswordForm()
    return render(request, 'change_password.html', {'change_password_form': change_password_form})

def changed_password(request):
    return render(request, 'changed_password.html') # このページに移動されている時にはログアウトされています。

def logout(request):
    django_logout(request) # logoutモジュールのlogoutでログイン認証を解除します。
    return django.http.HttpResponseRedirect('/logout') # logout画面へ移動します。