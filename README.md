# traffic-sign-classif
Simple CNN for Classication Of Traffic Signs

# Description of Files
cnn_struct.py - Structure of layers. Currently implementing SmallVGGNet.
cnn_traffic_sign.py - Trains the model.
test.py - Test the model on an actual image
cnn_plot.png - Training accuracy, and a number of other factors plotted to understand nature of training.

# Dataset structure
Training_Dataset
Final_Training
Trial_Images
Kept class name as name of the folder within which images were added. Image resolution can be of any size as we are resizing the image in the training program. 

Due to memory and processor limitations, I ran it on only 5 classes with ~30 images in each class. No data augmentation was implemented due to the above reason. However, feel free to increase the number of images, increase the resizing resolution and increase batch size on your own PC.




