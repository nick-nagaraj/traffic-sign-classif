import matplotlib
matplotlib.use("Agg")

# import the necessary packages
#from SmallVggNe import SmallVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from cnn_struct import SmallVGGNet

input_data = []
label = []
labels = []



label = os.listdir('Training_Dataset/Final_Training/Trial_Images')

for i in range (0, len(label)):
    for a,b,root in os.walk('Training_Dataset/Final_Training/Trial_Images/{}'.format(label[i])):
        for j in range (0, len(root)):
            image = cv2.imread('Training_Dataset/Final_Training/Trial_Images/{}/{}'.format(label[i], root[j]))
            image = cv2.resize(image, (64, 64))
            labels.append(label[i])
            input_data.append(image)

input_data = np.array(input_data, dtype = "float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(input_data, labels, test_size=0.25, random_state = 42)


lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

model = SmallVGGNet.build(width=64, height=64, depth=3,
	classes=len(lb.classes_))

INIT_LR = 0.01
EPOCHS = 2

BS = 32

# initialize the model and optimizer (you'll want to use
# binary_crossentropy for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=32)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

plot = 'cnn_plot'

N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(plot)

model_name = 'cnn_struct_model'
label_bin = 'cnn_struct_lb_pickle'
model.save(model_name)
f = open(label_bin, "wb")
f.write(pickle.dumps(lb))
f.close()

test_image = cv2.imread('Test_Dataset/Final_Test/Images/00017.ppm')
image_1 = test_image.copy()
image_1 = cv2.resize(image_1, (32, 32)).flatten()
image_1 = image_1.reshape((1, image_1.shape[0]))
image_1 = image_1.astype("float") / 255.0


model = load_model(model_name)
lb = pickle.loads(open(label_bin, "rb").read())
preds = model.predict(image_1)
i = preds.argmax(axis=1)[0]
label_name = lb.classes_[i]

text = "{}: {:.2f}%".format(label_name, preds[0][i] * 100)
cv2.putText(test_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.1,
	(0, 0, 255), 2)

# show the output image
print ("Predicted: {}, with a %{} chance".format(label_name, preds[0][i] * 100))
