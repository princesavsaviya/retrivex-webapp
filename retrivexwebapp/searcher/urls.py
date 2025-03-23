# searcher/urls.py
from django.urls import path
from .views import home_view, search_view, detail_view

urlpatterns = [
    path('', home_view, name='home_view'),
    path('search/', search_view, name='search_view'),
    path('detail/<path:drug_name>/', detail_view, name='detail_view')
]
