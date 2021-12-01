from django.urls import path
from . import views
app_name = 'DressApp' # これは、他のテンプレートからviewを呼び出す際に用いるが、この記事では無関係
urlpatterns = [
    path('', views.edit_profile_avator, name='index'), #urlに/DressAppと入力するとviews.pyのedit_profile_avator関数が呼び出される。
    path('chat', views.chat, name='chat'), #urlに/DressAppと入力するとviews.pyのindex関数が呼び出される。
    path('ajax-file-send/', views.ajax_file_send, name='ajax_file_send')
]