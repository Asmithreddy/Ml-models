import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from keras.datasets import mnist
objects=mnist
(train_img,train_lab),(test_img,test_lab)=objects.load_data()
for i in range(20):
  plt.subplot(4,5,i+1)
  plt.imshow(train_img[i],cmap='gray_r')
  plt.title("Digit : {}".format(train_lab[i]))
  plt.subplots_adjust(hspace=0.5)
  plt.axis('off')
  print('Training images shape : ',train_img.shape)
print('Testing images shape : ',test_img.shape)
print('How image looks like : ')
print(train_img[0])
plt.hist(train_img[0].reshape(784),facecolor='orange')
plt.title('Pixel vs its intensity',fontsize=16)
plt.ylabel('PIXEL')
plt.xlabel('Intensity')
train_img=train_img/255.0
test_img=test_img/255.0
print('How image looks like after normalising: ')
print(train_img[0])
from keras.models import Sequential
from keras.layers import Flatten,Dense
model=Sequential()
input_layer= Flatten(input_shape=(28,28))
model.add(input_layer)
hidden_layer1=Dense(512,activation='relu')
model.add(hidden_layer1)
hidden_layer2=Dense(512,activation='relu')
model.add(hidden_layer2)
output_layer=Dense(10,activation='softmax')
model.add(output_layer)
#compiling the sequential model
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_img,train_lab,epochs=15)
model.save('project.h5')
loss_and_acc=model.evaluate(test_img,test_lab,verbose=2)
print("Test Loss", loss_and_acc[0])
print("Test Accuracy", loss_and_acc[1])
plt.imshow(test_img[0],cmap='gray_r')
plt.title('Actual Value: {}'.format(test_lab[0]))
prediction=model.predict(test_img)
plt.axis('off')
print('Predicted Value: ',np.argmax(prediction[0]))
if(test_lab[0]==(np.argmax(prediction[0]))):
  print('Successful prediction')
else:
  print('Unsuccessful prediction')
# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img
from google.colab import files
uploaded = files.upload()
from IPython.display import Image
Image('5img.jpeg',width=250,height=250)
img = load_image('5img.jpeg')
digit=new_model.predict(img)
print('Predicted value : ',np.argmax(digit))
from google.colab import files
uploaded = files.upload()
from IPython.display import Image
Image('4.jpg')
img = load_image('4.jpg')
digit=model.predict(img)
print(np.argmax(digit))
from google.colab import files
uploaded = files.upload()
model=tf.keras.models.load_model('project1.h5')       # I have renamed the file as project1 in my PC
