# traffic-sign-classif
Simple CNN for Classication Of Traffic Signs

# Description of Files
1. cnn_struct.py - Structure of layers.
2. Currently implementing SmallVGGNet.
3. cnn_traffic_sign.py - Trains the model.
4. test.py - Test the model on an actual image
5. cnn_plot.png - Training accuracy, and a number of other factors plotted to understand nature of training and check for overfitting

# Dataset structure
1. Training_Dataset
2. Final_Training
3. Trial_Images
Kept class name as name of the folder within which images were added. Image resolution can be of any size as we are resizing the image in the training program. 

Due to memory and processor limitations, I ran it on only 5 classes with ~30 images in each class. No data augmentation was implemented due to the above reason. However, feel free to increase the number of images, increase the resizing resolution and increase batch size on your own PC.

# Link to Dataset
http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset


