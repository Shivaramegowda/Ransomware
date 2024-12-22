from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
	       path('UserLogin', views.UserLogin, name="UserLogin"),
	       path('UserLoginAction', views.UserLoginAction, name="UserLoginAction"),	   
	       path('Signup', views.Signup, name="Signup"),
	       path('SignupAction', views.SignupAction, name="SignupAction"),
	       path('LoadDGA', views.LoadDGA, name="LoadDGA"),
	       path('RunDGA', views.RunDGA, name="RunDGA"),
	       path('RunDGAAction', views.RunDGAAction, name="RunDGAAction"),	
	       path('Aboutus', views.Aboutus, name="Aboutus"),
	       
]