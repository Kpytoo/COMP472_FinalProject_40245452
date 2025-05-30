# COMP472_FinalProject_40245452
## Student Name: Daniel Codreanu --- ID: 40245452 --- Section: NN

### Notice!!! for TAs and correctors...
To retrieve the saved models, follow this google drive link: https://drive.google.com/drive/folders/1DH4-PfX7HcVTDARpSKZoqJyeEk3GnG5Q?usp=share_link     

To retrieve the CIFAR10 data, follow this google drive link: https://drive.google.com/drive/folders/1wtWHP34Na-FT0SIB5ct9Y_ZDAYrYbcqW?usp=share_link    

Please download both folders since my code uses them for the project.

***-Why are they in google drive?***  
Because the files are too big to be uploaded to github, and google drive
was the only solution I could find.  
Thank you for your understanding!  

***-Github repo and google drive links have been shared with:***  
marzieh.adeli@gmail.com  
shamita.datta119@gmail.com  
rikinchauhan01@gmail.com  


---


### Enumeration of the contents with description and purpose of each file in this project.

***-Saved_Models:***   

*Saved_Models* is a folder that contains all of the saved trained models for this project.  
There are:  
  - 2 Gaussian Naive Bayes models
  - 6 Decision Trees models
  - 3 Multi-layer Perceptron models
  - 3 VGG11 Convolutional Neural Network models

The folder can be downloaded from google drive as stated above.  
  
***-data:***   

*data* is a folder that contains all of the CIFAR10 dataset used in this project.  
It is from this folder that we retrieve our subsets.  

The folder can be downloaded from google drive as stated above.

***-Main_Driver.py:***  

*Main_Driver.py* is the main driver of this project.  
It imports and creates subsets from the CIFAR10 dataset.   
It uses Resnet18 to do feature extraction.  
It use PCA to further reduce the size of the features.  
It then trains, tests and displays the metrics for each four models and their variants.  

***-Gaussian_Naive_Baye_Models.py:***   

*Gaussian_Naive_Baye_Models.py* contains the implementation of my personal Naive Baye model.  
It contains the train method that trains both my personal Naive Baye model and Scikit's Naive Baye model.  
It also contains the testing methods for both models.  
It finally contains a display method that displays the metric of both models.  

***-Decision_Tree_Models.py:***   

*Decision_Tree_Models.py* contains the implementation of my personal Decision Tree model.  
It contains the train method that trains both my personal Decision Tree model and Scikit's Decision model and all the respective variants.  
It also contains the testing methods for all models.  
It finally contains a display method that displays the metric of all models.  

***-Multi_Layer_Perceptron_Models.py:***  

*Multi_Layer_Perceptron_Models.py* contains the implementation of the base MLP and its variants.  
It contains the train method that trains all models.  
It also contains the testing method for all models.  
It finally contains a display method that displays the metric of all models.  

***-Convolutional_Neural_Network_Models.py:*** 

*Convolutional_Neural_Network_Models.py* contains the implementation of the base VGG11 model and its variants.  
It contains the train method that trains all models.  
It also contains the testing method for all models.  
It finally contains a display method that displays the metric of all models.  

---

### Outline of steps to execute the code for data pre-processing.  

To execute the code for data pre-processing, all you have to do is run the *Main_Driver.py* file.  
Make sure you have all the necessary python packages installed before doing so.  
Make sure you have both the *Saved_Models* and *data* folders before proceeding.  
The *Main_Driver.py* file will pre-process the dataset as needed.  

---

### Steps for running the code to train, evaluate, and apply the models.  

To run this code, only execute the *Main_Driver.py* file.

To run the code for training, make sure to *UNCOMMENT* lines 168, 179, 196, 207 inside *Main_Driver.py*.  
These lines use the training functions for each model and their variants and will retrain them.  

If you wish to just evaluate the models, keep the above mentioned lines *COMMENTED* and simply run the *Main_Driver.py* file.  
The application of the models are done automatically!  

---  

Thank you for reading!


