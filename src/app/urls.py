from django.urls import path
from django.contrib.auth.views import LoginView
from . import views
app_name = 'app'
urlpatterns = [
    path('index',views.index,name='index'),
    path('customer',views.send_customer,name='customer')
]