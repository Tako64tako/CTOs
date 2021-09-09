from django.shortcuts import render
from .forms import TestForm


def index(request):
    my_dict = {
        'insert_something':"views.pyのinsert_something部分です。",
        'name':'Bashi',
        'test_titles': ['title 1', 'title 2', 'title 3'],
        'form': TestForm(),
    }
    return render(request, 'webtestapp/index.html', my_dict)