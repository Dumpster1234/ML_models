from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_dataset, name='upload_dataset'),
    path('model/<int:dataset_id>/', views.model_development, name='model_development'),
    path('results/', views.results, name='results'),
]