from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpRequest
from django.shortcuts import render, redirect
#from .forms import *
from django.contrib import messages
from django.shortcuts import render
from django.urls import reverse_lazy
from django.urls import reverse
from django.http import HttpResponse
from django.views.generic import (View,TemplateView,
ListView,DetailView,
CreateView,DeleteView,
UpdateView)
from . import models
from .forms import *
from django.core.files.storage import FileSystemStorage
#from topicApp.Topicfun import Topic
#from ckdApp.funckd import ckd
from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(df2.drop('classification_yes', 1), df2['classification_yes'], test_size = .2, random_state=10)

import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle
import matplotlib.pyplot as plt
#import eli5 #for purmutation importance
#from eli5.sklearn import PermutationImportance
#import shap #for SHAP values
#from pdpbox import pdp, info_plots #for partial plots
np.random.seed(123) #ensure reproduc
class dataUploadView(View):
    form_class = conForm
    success_url = reverse_lazy('success')
    template_name = 'create.html'
    failure_url= reverse_lazy('fail')
    filenot_url= reverse_lazy('filenot')
    def get(self, request, *args, **kwargs):
        form = self.form_class()
        return render(request, self.template_name, {'form': form})
    def post(self, request, *args, **kwargs):
        #print('inside post')
        form = self.form_class(request.POST, request.FILES)
        #print('inside form')
        if form.is_valid():
            form.save()
            data_cement= request.POST.get('cement')
            data_slag=request.POST.get('slag')
            data_flyash=request.POST.get('flyash')
            data_water=request.POST.get('water')
            data_superplasticizer=request.POST.get('superplasticizer')
            data_coarseaggreate=request.POST.get('coarseaggreate')
            data_fineaggreate=request.POST.get('fineaggreate')
            data_age=request.POST.get('age')

            #print (data)


            dataset=pd.read_csv("con_prep.csv")

            def split_scalar(indep_X, dep_Y):
                X_train,X_test,y_train,y_test = train_test_split(indep_X,dep_Y,test_size = 0.25, random_state = 0)

                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)
                return X_train,X_test,y_train,y_test

            def r2_prediction(regressor,X_test,y_test):

                y_pred = regressor.predict(X_test)
                r2 = r2_score(y_test,y_pred)
                return r2

            def random(X_train,y_train,X_test):
                regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
                regressor.fit(X_train,y_train)
                r2 = r2_prediction(regressor,X_test,y_test)
                return regressor,r2

            X = dataset.iloc[:, 0:8].values
            y = dataset.iloc[:, -1].values

            accrf = []

            X_train,X_test,y_train,y_test = split_scalar(X,y)

            reg,r2_r = random(X_train,y_train,X_test)
            accrf.append(r2_r)

            y_pred = reg.predict([[data_cement,data_slag,data_flyash,data_water,data_superplasticizer,data_coarseaggreate,data_fineaggreate,data_age]])

            return render(request, "succ_msg.html", {'data_cement':data_cement,'data_slag':data_slag,'data_flyash':data_flyash,'data_water':data_water,
                                                    'data_superplasticizer':data_superplasticizer,'data_coarseaggreate':data_coarseaggreate,
                                                    'data_fineaggreate':data_fineaggreate,'data_age':data_age,'y_pred' : y_pred})




        else:
            return redirect(self.failure_url)
