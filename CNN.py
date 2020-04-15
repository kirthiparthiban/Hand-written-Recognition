import pandas as pd
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras import Model
import keras
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation, Input, GlobalAveragePooling2D
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
import pickle
import matplotlib.pyplot as plt

Path = 'C:/Users/kiruthika.parthiban/Downloads/'

# Get the Target label in dictionary format
f=open(Path+'words.txt')
mydict={}
for line in f:
  lineSplit = line.strip().split(' ')
  assert len(lineSplit) >= 9
  char = lineSplit[8]
  filename = lineSplit[0]
  mydict.update({filename:char})


# Get the Input images into dictionary
imgFiles = os.listdir(Path+'Training_images/note_images/')
list_of_images = []
char=[]
for (i,a) in enumerate(imgFiles):
    print(a)
    b = cv2.imread(Path+f'/{a}',cv2.IMREAD_GRAYSCALE)
    list_of_images.append(b)
    lineSplit = a.strip().split('.')
    print(lineSplit)
    val = lineSplit[0]
    charval = mydict[val]
    char.append(charval)

df = pd.DataFrame(list(zip(list_of_images, char)),
               columns =['Name', 'val'])

sam = df.sample(frac=1)

# length of the sample
samples = len(sam)

input_imgs = sam['Name']
chars = sam['val']

# convert to a list of images
input_imgs = np.array(list(input_imgs))
chars = np.array(list(chars))

# split into training and validation set: 95% - 5%
splitIdx = int(0.95 * samples)
trainSamples = input_imgs[:splitIdx]
validationSamples = input_imgs[splitIdx:]

# put labels into lists
trainWords =  chars[:splitIdx]
validationWords =  chars[splitIdx:]

trainWords = np.array(trainWords)
validationWords = np.array(validationWords)

# label encoding of the for the target variable
label_encoder = LabelEncoder()
label_encoder.fit(chars)
len(label_encoder.classes_)
trainWords = label_encoder.transform(trainWords)
validationWords = label_encoder.transform(validationWords)

# Reshape the data
trainWords = trainWords.reshape(-1,1)
validationWords = validationWords.reshape(-1,1)

X_train_shape = trainSamples.shape[0]
X_test_shape = validationSamples.shape[0]

#reshape data to fit model
X_train = trainSamples.reshape(X_train_shape,64,64,1)
X_test = validationSamples.reshape(X_test_shape,64,64,1)

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

#one-hot encode target column
ohe = OneHotEncoder(sparse=False, handle_unknown = 'ignore')
ohe.fit(trainWords)
y_train = ohe.transform(trainWords)
y_test = ohe.transform(validationWords)

# Save the encoder into a pickle for future use
with open(Path+'encoder4.pickle', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Define the parameters
batch_size = 128
num_classes = 41 #(len(label_encoder.classes_))
epochs = 30
filter_pixel=3
noise = 1
droprate=0.25

# The input tensor
inputs = Input(shape=(64, 64, 1)) # Input shape of the model

#1st model layer
model = Conv2D(64, kernel_size = (3,3), activation = 'relu')(inputs) # create feature maps and learn from it with teh bacth samples of size 64
model = BatchNormalization()(model)# To avoid vanishing gradient on the weights and bais during back propagation
model = MaxPooling2D()(model) # To reduce the size of the large matrix by 75% into lower dimension for faster computation
model = Dropout(droprate)(model) # To forget ordrop the few pixels from the layer to avoid the learnign the noise of the parameter.

#2nd model layer
model = Conv2D(64, kernel_size = (3,3) , activation = 'relu')(model) # create feature maps and learn from it with teh bacth samples of size 64
model = BatchNormalization()(model) # To avoid vanishing gradient on the weights and bais during back propagation
#model = MaxPooling2D()(model) # To reduce the size of the large matrix by 75% into lower dimension for faster computation
model = Dropout(droprate)(model) # To forget ordrop the few pixels from the layer to avoid the learnign the noise of the parameter.

#3rd model layer
model = Conv2D(64, kernel_size = (3,3) , activation = 'relu')(model) # create feature maps and learn from it with teh bacth samples of size 64
model = BatchNormalization()(model) # To avoid vanishing gradient on the weights and bais during back propagation
#model = MaxPooling2D()(model) # To reduce the size of the large matrix by 75% into lower dimension for faster computation
model = Dropout(droprate)(model) # To forget ordrop the few pixels from the layer to avoid the learnign the noise of the parameter.

#4thd model layer
model = Conv2D(64, kernel_size = (3,3) , activation = 'relu')(model) # create feature maps and learn from it with teh bacth samples of size 64
model = BatchNormalization()(model) # To avoid vanishing gradient on the weights and bais during back propagation
#model = MaxPooling2D()(model)  # To reduce the size of the large matrix by 75% into lower dimension for faster computation
model = Dropout(droprate)(model) # To forget ordrop the few pixels from the layer to avoid the learnign the noise of the parameter.

# Average Pooling for vanishing gradient
model = GlobalAveragePooling2D()(model)

#Fully connected 1st layer
model = Flatten()(model)
model = Dense(500,use_bias=False)(model) # To connect all the layers as fully connected layers.
model = BatchNormalization()(model) # To avoid vanishing gradient on the weights and bais during back propagation
model = Activation('relu')(model) # To add some bais to the model for having distinguished weights.
model = Dropout(droprate)(model) # To forget ordrop the few pixels from the layer to avoid the learnign the noise of the parameter.

#Fully connected final layer
model = Dense(num_classes)(model) # To connect all the layers as fully connected layers.
model = Activation('softmax')(model) # As its a multi class classfication, softmax is used. This will add the model probability output from each output nodes to 1.


val = Model(inputs, model)

#compile model using accuracy to measure model performance
val.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

#describe the layers
val.summary()

# define path to save model
model_path = Path+'fm_cnn_BN16.h5'

# prepare callbacks
callbacks = [
    EarlyStopping(
        monitor='val_acc',
        patience=10,
        mode='max',
        verbose=1),
    ModelCheckpoint(model_path,
        monitor='val_acc',
        save_best_only=True,
        mode='max',
        verbose=1)
]

#train the model
H = val.fit(X_train, y_train, batch_size = batch_size, verbose = 1, validation_data=(X_test, y_test), epochs=epochs, shuffle=True,callbacks=callbacks)

score = model.evaluate(X_test, y_test, verbose=1)

#print loss and accuracy
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Validation dataset

val_Path = 'C:/Users/kiruthika.parthiban/Desktop/'
# Data preparation
# Scale the image to center and Resize the images into fixed size of 64,64
imgFiles = os.listdir(val_Path)
for (i,a) in enumerate(imgFiles):
    img = val_Path+f'/{a}'
    Conv_hsv_Gray = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)


    val = np.zeros((128,128), np.uint8)
    height,width = val.shape
    for y in range(height):
        for x in range(width):
            if val[y][x] == 0:
                val[y][x] = 255

    h,w = mask.shape
    for y in range(h):
        for x in range(w):
            if mask[y][x] == 255:
                #print(1)
                val[y][x] = 0
            else:
                #print(2)
                val[y][x] = 255

    b = cv2.resize(val, (64,64), interpolation = cv2.INTER_AREA)
    cv2.imwrite(Path+f'/{a}',b)

dir_names = []
files = folders = 0
for _, dirnames, filenames in os.walk(Path):
  # ^ this idiom means "we won't be using this value"
    #files += len(filenames)
    folders += len(dirnames)
    dir_names.append(dirnames)

print("{:,} files, {:,} folders".format(files, folders))
dir_names = dir_names[0]
print(dir_names)

#Prepere the dataset for prediction
import pickle
sent=[]
for i in range(len(dir_names)):
    word = []
    print(i)
    imgFiles = os.listdir(val_Path+f'word{i}')
    for (s,a) in enumerate(imgFiles):
        print(a)
        file = cv2.imread(val_Path+f'word{i}/{a}',cv2.IMREAD_GRAYSCALE)
        word.append(file)
        df_pred = pd.DataFrame(list(zip(word)), columns =['Name'])
        pred_img = df_pred['Name']
        pred_img = np.array(list(pred_img))
        #Normalize the data
        pred_img = pred_img/255
        img_shape = pred_img.shape[0]
        #reshape data to fit model
        pred_img = pred_img.reshape(img_shape,64,64,1)
        model_path = Path+'fm_cnn_BN16.h5'
        # load the saved best model weights
        model= load_model(model_path)

        model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])

        # predict outputs on validation images
        prediction = model.predict(pred_img)
        prediction = np.argmax(prediction,axis=1)
        with open(Path+'encoder4.pickle', 'rb') as handle:
           label_encoder = pickle.load(handle)
        letter = label_encoder.inverse_transform(prediction)

    words = ''.join(letter)
    print(words)
    space = ' '
    sent.append(words)
    sent.append(space)

sentences = ''.join(sent)

