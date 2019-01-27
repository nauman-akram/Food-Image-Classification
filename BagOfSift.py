
import os
import cv2
import numpy as np

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler



def gen_sift_features(extract_features,SIFT_discs):

    for feature in extract_features:
        sift = cv2.xfeatures2d.SIFT_create()
        kp, desc = sift.detectAndCompute(feature, None)
        SIFT_discs.append(desc)

def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))



def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

def show_rgb_img(img):
    return plt.imshow(cv2.cvtColor(img, cv2.CV_32S))

def color_hist(feature_matrix,bin16):

    histr=[0]*16
    color = ('b','g','r')

    for img in feature_matrix:
        
        for i,col in enumerate(color):
            histr =histr+ cv2.calcHist([img],[i],None,[16],[0,256])
        hist=np.sum(histr,axis=1)
        bin16.append(hist)


def load_data(target,path,featured_matrix,target_labels,imagelist):
    j=0
    i=0
    for j in range(0,2):
        new_path=path+"\\"+target[j]

        files = os.listdir(new_path)
        for name in files:
            file=new_path+"\\"+name
            image=cv2.imread(file)
            imagelist.append(name)
            featured_matrix.append(image)
            target_labels.append(i)
        i=i+1
#        for i in range(0,50):
#            file=new_path+"\\"+filenames[i]
#            image=cv2.imread(file)
#            imagelist.append(filenames[i])
#            octo_front_desc=color_hist(image)
#            octo_front_desc = octo_front_desc.reshape(1,-1)
#            featured_matrix.put(i,octo_front_desc)
#            target_labels.append(target[j])


def ColorShift(desc,bins):
    bins1=np.multiply(bins, 0.4)
    desc1=np.multiply(desc, 1-0.4)
    ColorSIFT=np.concatenate((desc1, bins1), axis=1)
    return ColorSIFT



import os.path
import pickle

if os.path.isfile("colorSift.pkl") ==True:
    if os.path.isfile("target.pkl") == True:
        ColorSIFT = pickle.load(open("colorsift.pkl", "rb"))
        target = pickle.load(open("target.pkl", "rb"))
else:

    path="D:\\Semester 8\\ML\\Project\\food-101\\food-101\\images"
    target=[]
    featured_matrix=[]
    target_labels=[]
    files = os.listdir(path)
    for name in files:
        target.append(name)

    imagelist=[]
    load_data(target,path,featured_matrix,target_labels,imagelist)
    bin16=[]
    color_hist(featured_matrix,bin16)
    bins16 = np.array(bin16)
    target=np.array(target_labels)
    dim=bins16.shape
    bins16.shape = (dim[0],dim[1])
    SIFT_discs=[]
    gen_sift_features(featured_matrix,SIFT_discs)
    for i in range(0,len(SIFT_discs)):
        SIFT_discs[i]=SIFT_discs[i].sum(axis=0)

    SIFT_discs1 = np.array(SIFT_discs)
    dim=SIFT_discs1.shape
    SIFT_discs1.shape = (dim[0],dim[1])

    ColorSIFT=ColorShift(SIFT_discs,bin16)



#
X_train, X_test, y_train, y_test = train_test_split(ColorSIFT, target, test_size=0.25)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_res=knn.predict(X_test)
print("KNN accuracy :: ",accuracy_score(knn_res, y_test)*100)


scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
clf = MLPClassifier(alpha=0.001, hidden_layer_sizes=(143,100), max_iter=2000)
clf.fit(X_train,y_train)
predict_Clf=clf.predict(X_test)




print(confusion_matrix(y_test,predict_Clf))
print("Accuracy Neural Network :: ",accuracy_score(y_test,predict_Clf)*100)

