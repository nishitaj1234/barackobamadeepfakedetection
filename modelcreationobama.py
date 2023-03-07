import pandas as pd
import numpy as np
import sklearn.model_selection
import keras

from keras.models import load_model

originalfile=pd.read_csv("obamaoriginal.txt")
originalfile2=pd.read_csv("obamaoriginal2.txt")

deepfakefile=pd.read_csv("obamadeepfake.txt")

originaldata=np.array(originalfile.drop("class",1))
originallabels=np.array(originalfile["class"])

originaldata2=np.array(originalfile2.drop("class",1))
originallabels2=np.array(originalfile2["class"])

print(len(originaldata),len(originallabels))

deepfakedata=np.array(deepfakefile.drop("class",1))
deepfakelabels=np.array(deepfakefile["class"])




'''print(originallabels)
print(deepfakelabels)
print(deepfakelabels2)'''
alloriginallabels=np.concatenate([originallabels,originallabels2])
alloriginaldata=np.concatenate([originaldata,originaldata2])

#alldeepfakelabels=np.concatenate([deepfakelabels,deepfakelabels2])
#alldeepfakedata=np.concatenate([deepfakedata,deepfakedata2])

print(len(deepfakelabels),len(alloriginallabels))
data=np.concatenate([deepfakedata,alloriginaldata])
labels=np.concatenate([deepfakelabels,alloriginallabels])

print(len(labels))

print(len(data))
print(data.shape)


#generate another deepfake based on given video. include in code, variable called all data and all labels
Xtraindata, Xtestdata, Ytraindata, Ytestdata=sklearn.model_selection.train_test_split(data,labels,test_size=0.2)


                        
                        
model=keras.Sequential([#keras.layers.Flatten(input_shape=(137,)),
                        keras.layers.Dense(128,activation='relu'),
                        keras.layers.Dense(128,activation='relu'),
                        keras.layers.Dense(128,activation='relu'),
                        keras.layers.Dense(2,activation='softmax')])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

model.fit(Xtraindata,Ytraindata,epochs=15)

model.save('deepfakeobama.h5')




