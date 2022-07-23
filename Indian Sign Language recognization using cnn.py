# importing necessary library
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report

# changing directory and getting all catergory name
#os.chdir("D:")
cwd = os.getcwd()
os.chdir(cwd)

DATADIR = cwd + "\\data"
folder = os.listdir(DATADIR)
CATEGORIES = folder

for category in CATEGORIES:                                                      # accessing image                        
    path = os.path.join(DATADIR,category)  
    for img in os.listdir(path):  
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.imshow(img_array)  # graph it
        plt.show()  # display!

        break  # we just want one for now so break
    break  #...and one more!
print(img_array.shape)

IMG_SIZE = 64                                                                  # resizing image 
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array)
plt.show()

# creating training data
training_data = []

def create_training_data():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category) 
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=1, 1=2, .....
        print(class_num)
        
        for img in tqdm(os.listdir(path)):  # iterate over each image per category
            img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)  # convert to array
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
            training_data.append([new_array, class_num])  # add this to our training_data
           
            

create_training_data()

random.shuffle(training_data) #shuffling trainig data

for sample in training_data[:100]: # checking data is shuffled
    print(sample[1])
    
X = []
y = []

X,y= zip(*training_data)                # removing features from category

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42) # splitting data for training and testing

X_train = np.array(X_train) 
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.array(X_test)
X_test = np.expand_dims(X_test, axis=-1)
y_train = np.array(y_train)
y_test = np.array(y_test)

X_train = X_train/255.0                # normalizing train data 
X_test = X_test/255.0                  # normalizing test data


      

cnn = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(64, 64, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(35, activation='softmax')
])



                                                  # adding model layers and defining convolution and max pooling layers
cnn.compile(optimizer='adam',                               
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=10)                        

cnn.evaluate(X_test,y_test)

y_pred = cnn.predict(X_test)                                  #|
y_pred[:5]                                                    #|  
y_classes = [np.argmax(element) for element in y_pred]        #|---> checking starting samples of predicted and actual category       
y_classes[:5]                                                 #|  
y_test[:5]                                                    #|          











































            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            