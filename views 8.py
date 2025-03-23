from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
import pickle
from django.conf import settings
import os
from pathlib import Path
import cv2
import numpy as np

import pathlib
from PIL import Image
import os, os.path
import glob
import base64
from io import StringIO
import io
import cv2
import base64 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pyautogui as pag

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import cv2
from scipy.stats import kurtosis, skew,entropy
import numpy as np
from scipy import ndimage
import statistics
import base64

BASE_DIR = Path(__file__).resolve().parent.parent

def index(request):

    if request.method == "POST":

        data = pd.read_csv('banknote_authentication.txt', header=None)
        data.columns = ['var', 'skew', 'curt', 'entr', 'auth']
        print(data.head())

        sns.pairplot(data, hue='auth')
        sns.countplot(x=data['auth'])
        target_count = data.auth.value_counts()

        nb_to_delete = target_count[0] - target_count[1]
        data = data.sample(frac=1, random_state=42).sort_values(by='auth')
        data = data[nb_to_delete:]

        x = data.loc[:, data.columns != 'auth']
        y = data.loc[:, data.columns == 'auth']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        scalar = StandardScaler()
        scalar.fit(x_train)
        x_train = scalar.transform(x_train)
        x_test = scalar.transform(x_test)

        clf = LogisticRegression(solver='lbfgs', random_state=42, multi_class='auto')
        clf.fit(x_train, y_train.values.ravel())

        y_pred = np.array(clf.predict(x_test))
        conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred),
                                columns=["Pred.Negative", "Pred.Positive"],
                                index=['Act.Negative', "Act.Positive"])
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        accuracy = round((tn+tp)/(tn+fp+fn+tp), 4)



        try:
            my_uploaded_file = request.FILES['my_uploaded_file'].read()
            my_uploaded_file_base64 = base64.b64encode(my_uploaded_file)

            print('loaded')

            def stringToImage(base64_string):
                imgdata = base64.b64decode(base64_string)
                image = Image.open(io.BytesIO(imgdata))
                return image

            def stringToEdgeImage(base64_string):
                imgdata = base64.b64decode(base64_string)
                image = Image.open(io.BytesIO(imgdata))
                #img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
                img_blur = cv2.GaussianBlur(np.array(image), (3,3), 0)
                sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
                return np.array(sobelxy)

            ######
            opencvImage = cv2.cvtColor(np.array(stringToImage(my_uploaded_file_base64)), cv2.COLOR_RGB2BGR)
            norm_image = cv2.normalize(opencvImage, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            img_blur = cv2.GaussianBlur(norm_image, (3,3), 0)
            sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
            #sobelxy = cv2.imshow('sobelxy', sobelxy)

            var = np.var(norm_image,axis=None)
            sk = skew(norm_image, axis=None)
            kur = kurtosis(norm_image, axis=None)
            ent = entropy(norm_image, axis=None)
            ent = ent/100



            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            result = clf.predict(np.array([[-0.91318,-2.0113,-0.19565,0.066365]]))
            result = clf.predict(np.array([[var,sk,kur,ent]]))
            print(result)

            out = ""
            if result[0] ==0:
                out = "Real Currency"
            else:
                out = "Fake Currency"

            ######

            fig = plt.figure(figsize=(3, 3))
            plt.axis('off')
            plt.imshow(stringToImage(my_uploaded_file_base64))

            imagedata = StringIO()
            fig.savefig(imagedata, format='svg')
            imagedata.seek(0)
            imagedata.getvalue()


            fig2 = plt.figure(figsize=(3, 3))
            plt.axis('off')
            plt.imshow(stringToEdgeImage(my_uploaded_file_base64))

            imagedata2 = StringIO()
            fig2.savefig(imagedata2, format='svg')
            imagedata2.seek(0)
            imagedata2.getvalue()

            if my_uploaded_file_base64 != None:
                return render(request, "result.html",{'original_image':imagedata.getvalue(),
                    'edge_image':imagedata2.getvalue(),
                    'variance':"{:.2f}".format(var),
                    'skew':"{:.2f}".format(sk),
                    'kurtosis':"{:.2f}".format(kur),
                    'entropy':"{:.2f}".format(ent),
                    'accuracy':accuracy,
                    'result':result,
                    'out':out})
        except:
            print("Notes picture not loaded")
    return render(request, "index.html")

def result(request):
    return render(request, "result.html")