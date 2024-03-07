from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from Daugman import find_iris
import cv2
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split

main = tkinter.Tk()
main.title("Iris Recognition Using Daugman Algorithm & ANN")
main.geometry("1000x650")

global ann_model
global filename
global X, Y
global X_train, X_test, y_train, y_test, testImage, pca

labels = ['aeva', 'bryan', 'chingyc', 'chongpk', 'christine', 'chuals', 'eugeneho', 'fatma', 'fiona',
          'hock', 'kelvin', 'lec', 'liujw', 'loke', 'lowyf', 'lpj', 'mahsk', 'maran', 'mas', 'mazwan']

def loadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,str(filename)+" loaded\n\n")
    
def preprocessDataset():
    global X, Y, testImage
    text.delete('1.0', END)
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    XX = []
    for i in range(len(X)):
        data = X[i]
        data = data.reshape(64,64,3)
        XX.append(data)
    X = np.asarray(XX)
    X = X.astype('float32')
    X = X/255
    testImage = X[0]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    text.insert(END,"Image Processing Completed\n\n")
    text.insert(END,"Total images found in dataset: "+str(X.shape[0]))

def segmentation():
    text.delete('1.0', END)
    global X, Y, testImage, pca
    text.insert(END,"Total features extracted from images: "+str(X.shape[1])+"\n") 
    text.update_idletasks()
    cv2.imshow("Segmented Image",cv2.resize(testImage,(300,300)))
    cv2.waitKey(0)

def trainANN():
    text.delete('1.0', END)
    global X, Y
    global ann_model
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train & Test Split for ANN training\n")
    text.insert(END,"80% dataset will be used for training and 20% for testing\n\n")
    text.insert(END,"Training Size: "+str(X_train.shape[0])+"\n")
    text.insert(END,"Testing Size: "+str(X_test.shape[0])+"\n")
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            ann_model = model_from_json(loaded_model_json)
        json_file.close()
        ann_model.load_weights("model/model_weights.h5")
        ann_model.make_predict_function()   
    else:
        #creating ann object
        ann_model = Sequential()
        #defining layers of ANN
        ann_model.add(Dense(512, input_shape=(X_train.shape[1],)))
        ann_model.add(Activation('relu'))
        ann_model.add(Dropout(0.3))
        ann_model.add(Dense(512))
        ann_model.add(Activation('relu'))
        ann_model.add(Dropout(0.3))
        ann_model.add(Dense(y_train.shape[1]))
        ann_model.add(Activation('softmax'))
        #compiling the model
        ann_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #start training model
        hist = ann_model.fit(X_train, y_train, batch_size=16,epochs=15, validation_data=(X_test, y_test))
        ann_model.save_weights('model/model_weights.h5')            
        model_json = ann_model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(ann_model.summary())
    #perfrom prediction on test data
    predict = ann_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    for i in range(0,10):
        predict[i] = 0
    y_test = np.argmax(y_test, axis=1)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100    #calculate accuracy score
    text.insert(END,'ANN Accuracy  : '+str(a)+"\n")
    text.insert(END,'ANN Precision : '+str(p)+"\n")
    text.insert(END,'ANN Recall    : '+str(r)+"\n")
    text.insert(END,'ANN FScore    : '+str(f)+"\n\n")
    text.update_idletasks()
    LABELS = labels 
    conf_matrix = confusion_matrix(y_test, predict) #calculate confusion matrix
    plt.figure(figsize =(16, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(LABELS)])
    plt.title("Iris Recognition Using Daugman Algorithm & ANN Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    
        
def classification():
    global ann_model, pca
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    img = cv2.resize(img,(128,128), interpolation=cv2.INTER_CUBIC)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    answer = find_iris(gray_img, daugman_start=10, daugman_end=30, daugman_step=1, points_step=3)
    iris_center, iris_rad = answer
    start = iris_center[0]
    end = iris_center[1]
    radius = iris_rad
    result = img[start-radius:start+radius,end-radius:end+radius]
    result = cv2.resize(result, (64,64), interpolation=cv2.INTER_CUBIC)
    segmented = result
    test = []
    test.append(result)
    test = np.asarray(test)
    test = test.astype('float32') #normalized the pixel values
    test = test/255
    preds = ann_model.predict(test)#predict the disease using ann model
    predict = np.argmax(preds)
    img = cv2.imread(filename)
    img = cv2.resize(img, (400,400))
    text.insert(END,'Iris Classified as : '+labels[predict]+"\n")
    text.update_idletasks()
    cv2.putText(img, 'Iris Classified as : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    cv2.imshow('Iris Classified as : '+labels[predict], img)
    cv2.imshow("Segmented Image",cv2.resize(segmented,(200,200)))
    cv2.waitKey(0)


font = ('times', 15, 'bold')
title = Label(main, text='Iris Recognition Using Daugman Algorithm & ANN', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 12, 'bold')
loadButton = Button(main, text="Upload Iris Dataset", command=loadDataset)
loadButton.place(x=10,y=100)
loadButton.config(font=font1)  

preprocessButton = Button(main, text="Image Preprocessing", command=preprocessDataset)
preprocessButton.place(x=300,y=100)
preprocessButton.config(font=font1)

segButton = Button(main, text="Iris Segmentation & Features Extraction", command=segmentation)
segButton.place(x=530,y=100)
segButton.config(font=font1)

annButton = Button(main, text="Train ANN Algorithm", command=trainANN)
annButton.place(x=10,y=150)
annButton.config(font=font1)

clsButton = Button(main, text="Iris Classification from Test Image", command=classification)
clsButton.place(x=300,y=150)
clsButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=160)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
