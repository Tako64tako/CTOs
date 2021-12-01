from PIL import Image
from django import forms

class UpLoadProfileImgForm(forms.Form):
    avator = forms.ImageField(required=True)