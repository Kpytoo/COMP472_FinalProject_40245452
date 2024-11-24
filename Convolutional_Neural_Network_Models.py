import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#     *******************************************
# <<< Convolutional Neural Network implementation >>> 
#     *******************************************


#Lots of help from this source: https://debuggercafe.com/implementing-vgg11-from-scratch-using-pytorch/

#Create our VGG11 base model
class VGG11_Base_Model(nn.Module):

    #Create our layers
    def __init__(self):
        super(VGG11_Base_Model, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.linear3 = nn.Linear(4096, 10)

    #Forward function
    def forward(self, sample):
        
        sample = self.conv1(sample)
        sample = self.conv2(sample)
        sample = self.conv3(sample)
        sample = self.conv4(sample)
        sample = self.conv5(sample)
        sample = self.conv6(sample)
        sample = self.conv7(sample)
        sample = self.conv8(sample)

        sample = sample.view(sample.size(0), -1) #Flatten for linear functions

        sample = self.linear1(sample)
        sample = self.linear2(sample)
        sample = self.linear3(sample)

        return sample
    

#Create our VGG11 variant 1 model (Less convolutions - Removed 7th and 8th conv.)
class VGG11_Model_V1(nn.Module):

    #Create our layers
    def __init__(self):
        super(VGG11_Model_V1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        self.linear1 = nn.Sequential(
            nn.Linear(512 * 14 * 14, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.linear3 = nn.Linear(4096, 10)

    #Forward function
    def forward(self, sample):
        
        sample = self.conv1(sample)
        sample = self.conv2(sample)
        sample = self.conv3(sample)
        sample = self.conv4(sample)
        sample = self.conv5(sample)
        sample = self.conv6(sample)
        #Removed conv7
        #Removed conv8

        sample = sample.view(sample.size(0), -1) #Flatten for linear functions

        sample = self.linear1(sample)
        sample = self.linear2(sample)
        sample = self.linear3(sample)

        return sample



#Create our VGG11 variant 2 model (Kernel size 2x2 adjusted in conv3 and conv5)
class VGG11_Model_V2(nn.Module):

    #Create our layers
    def __init__(self):
        super(VGG11_Model_V2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 2, stride = 1, padding = 0), #Adjusted kernel and padding
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 2, stride = 1, padding = 0), #Adjusted kernel and padding
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear1 = nn.Sequential(
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.linear3 = nn.Linear(4096, 10)

    #Forward function
    def forward(self, sample):
        
        sample = self.conv1(sample)
        sample = self.conv2(sample)
        sample = self.conv3(sample)
        sample = self.conv4(sample)
        sample = self.conv5(sample)
        sample = self.conv6(sample)
        sample = self.conv7(sample)
        sample = self.conv8(sample)

        sample = sample.view(sample.size(0), -1) #Flatten for linear functions

        sample = self.linear1(sample)
        sample = self.linear2(sample)
        sample = self.linear3(sample)

        return sample



#Function to train and save all models and their variations
def trainVGG11Models(train_loader):

    #Set CrossEntropyLoss criterion used by all models
    criterion = nn.CrossEntropyLoss()
    #All models will be trained on 5 epochs
    EPOCHS = 5

    #*************************************************************************************

    #Create base model
    vgg_base = VGG11_Base_Model() 
    #Set SGD optimizer
    optimizer = torch.optim.SGD(vgg_base.parameters(), lr=0.01, momentum=0.9)  
    total = 0 #Total number of samples
    correct = 0 #Correct predictions made

    #Train base model
    for epoch in range(EPOCHS):
        vgg_base.train()
        running_loss = 0

        for i, t in train_loader:
            inputs, targets = i, t

            optimizer.zero_grad()  #Zero gradients for every batch

            outputs = vgg_base(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        # Print stats for each epoch
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    
    #Save the model
    torch.save(vgg_base.state_dict(), "Saved_Models/vgg11_base_model.pth")

    #*************************************************************************************

    #Create variant 1 model
    vgg_variant1 = VGG11_Model_V1() 
    #Set SGD optimizer
    optimizer = torch.optim.SGD(vgg_variant1.parameters(), lr=0.01, momentum=0.9)  
    total = 0 #Total number of samples
    correct = 0 #Correct predictions made

    #Train variant 1 model
    for epoch in range(EPOCHS):
        vgg_variant1.train()
        running_loss = 0

        for i, t in train_loader:
            inputs, targets = i, t

            optimizer.zero_grad()  #Zero gradients for every batch

            outputs = vgg_variant1(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        # Print stats for each epoch
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    
    #Save the model
    torch.save(vgg_variant1.state_dict(), "Saved_Models/vgg11_variant1_model.pth")

    #*************************************************************************************

    #Create variant 2 model
    vgg_variant2 = VGG11_Model_V2() 
    #Set SGD optimizer
    optimizer = torch.optim.SGD(vgg_variant2.parameters(), lr=0.01, momentum=0.9)  
    total = 0 #Total number of samples
    correct = 0 #Correct predictions made

    #Train variant 2 model
    for epoch in range(EPOCHS):
        vgg_variant2.train()
        running_loss = 0

        for i, t in train_loader:
            inputs, targets = i, t

            optimizer.zero_grad()  #Zero gradients for every batch

            outputs = vgg_variant2(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        # Print stats for each epoch
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    
    #Save the model
    torch.save(vgg_variant2.state_dict(), "Saved_Models/vgg11_variant2_model.pth")


#Testing function for all VGG11 models
def cnnModels(test_loader):

    #Create base model
    vgg_base = VGG11_Base_Model()
    #Load saved base model
    vgg_base.load_state_dict(torch.load("Saved_Models/vgg11_base_model.pth", weights_only=True))
    #Test the model
    vgg_base.eval() 
    predictions_base= [] #All predicted targets
    test_targets = [] #All test targets

    with torch.no_grad():  # No gradients needed during evaluation
        for i, t in test_loader:
            inputs, targets = i, t

            #Gather data
            outputs = vgg_base(inputs)
            _, predicted = torch.max(outputs, 1)
            
            predictions_base.extend(predicted.cpu().numpy()) #Convert to CPU numpy array
            test_targets.extend(targets.cpu().numpy()) #Convert to CPU numpy array

    #Display confusion matrix for base vgg11 model
    confusion_mtx = confusion_matrix(test_targets, predictions_base)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=np.unique(test_targets))
    display.plot()
    plt.title("VGG11 CNN Base Model Confusion Matrix")
    plt.show()

    #Calculate metrics for base vgg11 model
    accuracy_base  = accuracy_score(test_targets, predictions_base)
    precision_base = precision_score(test_targets, predictions_base, average = "macro", zero_division = 0)
    recall_base = recall_score(test_targets, predictions_base, average = "macro", zero_division = 0)
    f1_base = f1_score(test_targets, predictions_base, average = "macro", zero_division = 0)

    #Format metrics as a tuple
    base_metrics = (accuracy_base, precision_base, recall_base, f1_base)

    #*************************************************************************************

    #Create variant 1 model
    vgg_variant1 = VGG11_Model_V1()
    #Load saved variant 1 model
    vgg_variant1.load_state_dict(torch.load("Saved_Models/vgg11_variant1_model.pth", weights_only=True))
    #Test the model
    vgg_variant1.eval() 
    predictions_variant1= [] #All predicted targets
    test_targets = [] #All test targets

    with torch.no_grad():  # No gradients needed during evaluation
        for i, t in test_loader:
            inputs, targets = i, t

            #Gather data
            outputs = vgg_variant1(inputs)
            _, predicted = torch.max(outputs, 1)
            
            predictions_variant1.extend(predicted.cpu().numpy()) #Convert to CPU numpy array
            test_targets.extend(targets.cpu().numpy()) #Convert to CPU numpy array

    #Display confusion matrix for base vgg11 model
    confusion_mtx = confusion_matrix(test_targets, predictions_variant1)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=np.unique(test_targets))
    display.plot()
    plt.title("VGG11 CNN Variant 1 Model Confusion Matrix")
    plt.show()

    #Calculate metrics for variant 1 vgg11 model
    accuracy_v1  = accuracy_score(test_targets, predictions_variant1)
    precision_v1 = precision_score(test_targets, predictions_variant1, average = "macro", zero_division = 0)
    recall_v1 = recall_score(test_targets, predictions_variant1, average = "macro", zero_division = 0)
    f1_v1 = f1_score(test_targets, predictions_variant1, average = "macro", zero_division = 0)

    #Format metrics as a tuple
    variant1_metrics = (accuracy_v1, precision_v1, recall_v1, f1_v1)


    #*************************************************************************************

    #Create variant 2 model
    vgg_variant2 = VGG11_Model_V2()
    #Load saved variant 2 model
    vgg_variant2.load_state_dict(torch.load("Saved_Models/vgg11_variant2_model.pth", weights_only=True))
    #Test the model
    vgg_variant2.eval() 
    predictions_variant2= [] #All predicted targets
    test_targets = [] #All test targets

    with torch.no_grad():  # No gradients needed during evaluation
        for i, t in test_loader:
            inputs, targets = i, t

            #Gather data
            outputs = vgg_variant2(inputs)
            _, predicted = torch.max(outputs, 1)
            
            predictions_variant2.extend(predicted.cpu().numpy()) #Convert to CPU numpy array
            test_targets.extend(targets.cpu().numpy()) #Convert to CPU numpy array

    #Display confusion matrix for variant 2 vgg11 model
    confusion_mtx = confusion_matrix(test_targets, predictions_variant2)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=np.unique(test_targets))
    display.plot()
    plt.title("VGG11 CNN Variant 2 Model Confusion Matrix")
    plt.show()

    #Calculate metrics for base vgg11 model
    accuracy_v2  = accuracy_score(test_targets, predictions_variant2)
    precision_v2 = precision_score(test_targets, predictions_variant2, average = "macro", zero_division = 0)
    recall_v2 = recall_score(test_targets, predictions_variant2, average = "macro", zero_division = 0)
    f1_v2 = f1_score(test_targets, predictions_variant2, average = "macro", zero_division = 0)

    #Format metrics as a tuple
    variant2_metrics = (accuracy_v2, precision_v2, recall_v2, f1_v2)


    return base_metrics, variant1_metrics, variant2_metrics #Return a tuple of tuples


#Display metrics functions
def displayCNNMetrics(cnnMetrics):

    #Dictionary with data
    data = {
        "Model" : ["VGG11 Base Model", "VGG11 Variant 1 Model", "VGG11 Variant 2 Model"],
        "Accuracy" : [cnnMetrics[0][0], cnnMetrics[1][0], cnnMetrics[2][0]],
        "Precision" : [cnnMetrics[0][1], cnnMetrics[1][1], cnnMetrics[2][1]],
        "Recall" : [cnnMetrics[0][2], cnnMetrics[1][2], cnnMetrics[2][2]], 
        "F1" : [cnnMetrics[0][3], cnnMetrics[1][3], cnnMetrics[2][3]]
    }

    #Create table
    table = pd.DataFrame(data)
    #Display table
    print(table)
    
   
