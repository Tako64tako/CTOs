from django import forms
from django.forms import PasswordInput

class RegistrationForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField()
    email = forms.EmailField()

class LoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField()

class ChangePasswordForm(forms.Form):
    new_password = forms.CharField(label='新しいパスワード', widget=PasswordInput())  # label = ''で画面に表示されるラベルを設定します。widget=PasswordInput()で、入力した値が●●●で表示されます。PasswordInput()を利用するには、PasswordInputをインポートする必要があります。