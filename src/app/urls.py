from django.urls import path
from django.contrib.auth.views import LoginView
from . import views
app_name = 'app'
urlpatterns = [
    path('chat',views.index,name='chat'),
    path('customer',views.send_customer,name='customer')
]