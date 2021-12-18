from django import forms
from .models import Human_body

class ImageForm(forms.ModelForm):
    class Meta:
        model = Human_body
        fields = ['id','human_name','height','picture']