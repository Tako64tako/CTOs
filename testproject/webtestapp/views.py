from django.shortcuts import render

def index(request):
    my_dict = {
        'insert_something':"views.pyのinsert_something部分です。",
        'name':'Bashi',
        'test_titles': ['title 1', 'title 2', 'title 3'],
    }
    return render(request, 'webtestapp/index.html', my_dict)