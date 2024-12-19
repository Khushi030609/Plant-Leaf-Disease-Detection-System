#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install opencv-python')


# In[111]:


# import tkinter as tk
# from tkinter import filedialog  # Add this line
# from PIL import Image, ImageTk
# import cv2


# def open_upload_page():
#     homepage_frame.pack_forget()
#     upload_frame.pack(pady=20)

# def open_homepage():
#     upload_frame.pack_forget()
#     homepage_frame.pack(pady=20)

# def browse_image():
#     filename = filedialog.askopenfilename(initialdir="/", title="Select Image",
#                                           filetypes=(("Image files", "*.jpg *.png *.jpeg"), ("all files", "*.*")))
#     if filename:
#         load_image(filename)

# def load_image(filename):
#     image = Image.open(filename)
#     image.thumbnail((300, 300))  # Resize image to fit in the GUI
#     photo = ImageTk.PhotoImage(image)
#     label.config(image=photo)
#     label.image = photo
#     detect_button.pack(pady=10)

# def scan_image():
#     def capture_image():
#         ret, frame = cap.read()
#         cv2.imwrite("scanned_leaf.jpg", frame)
#         cap.release()  # Release the camera after capturing the image
#         load_image("scanned_leaf.jpg")
#         detect_button.pack(pady=10)

#     def go_back():
#         cap.release()
#         camera_frame.pack_forget()
#         open_upload_page()

#     cap = cv2.VideoCapture(0)
#     ret, frame = cap.read()

#     if ret:
#         camera_frame.pack_forget()
#         camera_frame.pack(pady=20)

#         def update():
#             ret, frame = cap.read()
#             if ret:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame = cv2.resize(frame, (400, 300))
#                 img = Image.fromarray(frame)
#                 imgtk = ImageTk.PhotoImage(image=img)
#                 label.imgtk = imgtk
#                 label.config(image=imgtk)
#             label.after(10, update)

#         update()

#         capture_button = tk.Button(camera_frame, text="Capture", command=capture_image)
#         capture_button.pack(pady=10)

#         go_back_button = tk.Button(camera_frame, text="Go Back", command=go_back)
#         go_back_button.pack(pady=10)

# def detect_disease():
#     # Placeholder for disease detection functionality
#     pass

# root = tk.Tk()
# root.title("LEAF DISEASE DETECTION SYSTEM")
# root.geometry("800x600")  # Set initial size of the window

# # Homepage frame
# homepage_frame = tk.Frame(root)
# homepage_frame.pack(pady=20)

# label = tk.Label(homepage_frame, text="LEAF DISEASE DETECTION SYSTEM", font=("Arial", 14, "bold"))
# label.pack(pady=20)

# start_button = tk.Button(homepage_frame, text="Start", command=open_upload_page)
# start_button.pack()

# # Upload page frame
# upload_frame = tk.Frame(root)

# upload_button = tk.Button(upload_frame, text="Upload Image", command=browse_image)
# upload_button.pack(pady=5)

# scan_button = tk.Button(upload_frame, text="Scan Image", command=scan_image)
# scan_button.pack(pady=5)

# go_back_button = tk.Button(upload_frame, text="Go Back", command=open_homepage)
# go_back_button.pack(pady=5)

# detect_button = tk.Button(upload_frame, text="Detect Disease", command=detect_disease)  # Placeholder for detect function

# # Camera frame
# camera_frame = tk.Frame(root)

# label = tk.Label(camera_frame)
# label.pack()

# root.mainloop()


# In[4]:



get_ipython().system('pip install tensorflow')


# In[128]:


import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Define the learning rate
learning_rate = 0.0004


# In[129]:


import os

directory_path = '/Leaf dataset/'
print(os.listdir(directory_path))


# In[25]:


import zipfile
import os

# Define the path to the uploaded zip file
zip_file_path = '/Leaf dataset/5.zip'

# Define the directory where you want to extract the contents
extracted_dir_path = '/Leaf dataset/'

# Create the directory if it doesn't exist
os.makedirs(extracted_dir_path, exist_ok=True)

# Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir_path)


# In[130]:


# Now you can use the extracted directory as the source for image_dataset_from_directory
training_set = tf.keras.utils.image_dataset_from_directory(
    '/Leaf dataset/image data/train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

validation_set = tf.keras.utils.image_dataset_from_directory(
    '/Leaf dataset/image data/validation',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)


# In[131]:


cnn = tf.keras.models.Sequential()


# In[132]:


cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[128,128,3]))
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# In[133]:


cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# In[134]:


cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# In[135]:


cnn.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Dropout(0.25))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=1500,activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.4)) #To avoid overfitting


# In[136]:


#Output Layer
cnn.add(tf.keras.layers.Dense(units=13,activation='softmax'))


# In[137]:


cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

cnn.summary()


# In[138]:


training_history = cnn.fit(x=training_set,validation_data=validation_set,epochs=10)


# In[139]:


#Training set Accuracy
train_loss, train_acc = cnn.evaluate(training_set)
print('Training accuracy:', train_acc)


# In[140]:


#Validation set Accuracy
val_loss, val_acc = cnn.evaluate(validation_set)
print('Validation accuracy:', val_acc)


# In[141]:


cnn.save('trained_plant_disease_model3.keras')


# In[142]:


training_history.history #Return Dictionary of history


# In[143]:


#Recording History in json
import json
with open('training_hist.json','w') as f:
  json.dump(training_history.history,f)


# In[144]:


print(training_history.history.keys())


# In[145]:


epochs = [i for i in range(1,11)]
plt.plot(epochs,training_history.history['accuracy'],color='red',label='Training Accuracy')
plt.plot(epochs,training_history.history['val_accuracy'],color='blue',label='Validation Accuracy')
plt.xlabel('No. of Epochs')
plt.title('Visualization of Accuracy Result')
plt.legend()
plt.show()


# In[146]:


class_name = validation_set.class_names


# In[147]:


test_set = tf.keras.utils.image_dataset_from_directory(
    '/Leaf dataset/image data/validation',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=1,
    image_size=(128, 128),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)


# In[148]:


y_pred = cnn.predict(test_set)
predicted_categories = tf.argmax(y_pred, axis=1)


# In[149]:


true_categories = tf.concat([y for x, y in test_set], axis=0)
Y_true = tf.argmax(true_categories, axis=1)


# In[150]:


Y_true


# In[151]:


predicted_categories


# In[152]:


from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(Y_true,predicted_categories)


# In[153]:


# Precision Recall Fscore
print(classification_report(Y_true,predicted_categories,target_names=class_name))


# In[90]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# In[154]:



class_name = validation_set.class_names
print(class_name)


# In[155]:


cnn = tf.keras.models.load_model('trained_plant_disease_model3.keras')


# In[156]:


#Test Image Visualization
import cv2
import matplotlib.pyplot as plt
image_path = 'C:/Leaf dataset/image data/train/strawberry/leaf scorch/1be23894-5bb2-4e82-9e65-216fe9a33c3b___RS_L.Scorch 0077.JPG'
# Reading an image in default mode
img = cv2.imread(image_path)
# Check if the image was read successfully
if img is not None:
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #Converting BGR to RGB
    # Displaying the image 
    plt.imshow(img)
    plt.title('Test Image')
    plt.xticks([])
    plt.yticks([])
    plt.show()
else:
    print("Failed to read the image. Please check the image path.")


# In[157]:


image = tf.keras.preprocessing.image.load_img(image_path,target_size=(128,128))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = cnn.predict(input_arr)


# In[158]:


print(predictions)


# In[159]:


result_index = np.argmax(predictions) #Return index of max element
print(result_index)


# In[160]:


# Displaying the disease prediction
model_prediction = class_name[result_index]
plt.imshow(img)
plt.title(f"Disease Name: {model_prediction}")
plt.xticks([])
plt.yticks([])
plt.show()


# In[ ]:




