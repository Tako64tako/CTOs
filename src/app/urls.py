from django.urls import path
from django.contrib.auth.views import LoginView
from . import views
app_name = 'app'
urlpatterns = [
    path('', views.IndexView.as_view(), name='list'),
    path('<int:pk>/', views.DetailView.as_view(), name='detail'),
]