import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#     *************************************
# <<< Multi-Layer Perceptron implementation >>> 
#     *************************************

#Lots of help from this source: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

#Create our multi-layer perceptron base model
class MLP_Base_Model(nn.Module): 
    def __init__(self):
        super(MLP_Base_Model, self).__init__()

        #The linear functions
        self.linear1 = torch.nn.Linear(50, 512)
        self.linear2 = torch.nn.Linear(512, 512)
        self.linear3 = torch.nn.Linear(512, 10)

        #The relu function
        self.relu = torch.nn.ReLU()

        #The batchnorm function
        self.batch_norm = torch.nn.BatchNorm1d(512)
    
    #Forward function
    def forward(self, sample):

        sample = self.linear1(sample)
        sample = self.relu(sample)
        sample = self.linear2(sample)
        sample = self.batch_norm(sample)
        sample = self.relu(sample)
        sample = self.linear3(sample)

        return sample


#Create our multi-layer perceptron variant 1 model (More layers)
class MLP_Model_V1(nn.Module):
    def __init__(self):
        super(MLP_Model_V1, self).__init__()

        #The linear functions
        self.linear1 = torch.nn.Linear(50, 512)
        self.linear2 = torch.nn.Linear(512, 512)
        self.linear3 = torch.nn.Linear(512, 512) #Added layer
        self.linear4 = torch.nn.Linear(512, 10)

        #The relu function
        self.relu = torch.nn.ReLU()

        #The batchnorm function
        self.batch_norm = torch.nn.BatchNorm1d(512)
    
    #Forward function
    def forward(self, sample):

        sample = self.linear1(sample)
        sample = self.relu(sample)
        sample = self.linear2(sample)
        sample = self.relu(sample)
        sample = self.linear3(sample) #Added layer
        sample = self.batch_norm(sample)
        sample = self.relu(sample)
        sample = self.linear4(sample)

        return sample


#Create our multi-layer perceptron variant 2 model (Smaller sizes)
class MLP_Model_V2(nn.Module):
    def __init__(self):
        super(MLP_Model_V2, self).__init__()

        #The linear functions
        self.linear1 = torch.nn.Linear(50, 256)
        self.linear2 = torch.nn.Linear(256, 256) #Smaller sizes
        self.linear3 = torch.nn.Linear(256, 10)

        #The relu function
        self.relu = torch.nn.ReLU()

        #The batchnorm function
        self.batch_norm = torch.nn.BatchNorm1d(256)
    
    #Forward function
    def forward(self, sample):

        sample = self.linear1(sample)
        sample = self.relu(sample)
        sample = self.linear2(sample)
        sample = self.batch_norm(sample)
        sample = self.relu(sample)
        sample = self.linear3(sample)

        return sample
    


#Function to train and save all models and their variations
def trainPerceptronModels(train_features_tensor, train_targets_tensor):

    #Set CrossEntropyLoss criterion used by all models
    criterion = nn.CrossEntropyLoss()
    #All models will be trained on 5 epochs
    EPOCHS = 5

    #*************************************************************************************

    #Create base MLP model
    MLP_Base = MLP_Base_Model()
    #Set SGD optimizer
    optimizer = torch.optim.SGD(MLP_Base.parameters(), lr=0.1, momentum=0.9) 
    total = 0 #Total number of samples
    correct = 0 #Correct predictions made

    #Train base MLP model
    for epoch in range(EPOCHS):
        MLP_Base.train()
        running_loss = 0

        #Since train_features_tensors is a 2d array, we want to train for batches of 10 images
        for i in range(0, len(train_features_tensor), 10): #Batch size of 10 images

            inputs = train_features_tensor[i:i+10] #Get the next 10 images (batch)
            targets = train_targets_tensor[i:i+10] #Get the next 10 targets (batch)

            optimizer.zero_grad() #Zero gradients for every batch

            outputs = MLP_Base(inputs.float()) #Predictions for this batch

            #Compute loss and gradients
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step() #Adjust learning weights

            #Gather data
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        epoch_loss = running_loss / (len(train_features_tensor) // 10)
        epoch_accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%') #Print the training accuracy

    #Save the model
    torch.save(MLP_Base.state_dict(), "Saved_Models/mlp_base_model.pth")

    #*************************************************************************************

    #Create variant 1 MLP model
    MLP_Variant1 = MLP_Model_V1()
    #Set SGD optimizer
    optimizer = torch.optim.SGD(MLP_Variant1.parameters(), lr=0.1, momentum=0.9) 
    total = 0 #Total number of samples
    correct = 0 #Correct predictions made

    #Train variant 1 MLP model
    for epoch in range(EPOCHS):
        MLP_Variant1.train()
        running_loss = 0

        #Since train_features_tensors is a 2d array, we want to train for batches of 10 images
        for i in range(0, len(train_features_tensor), 10): #Batch size of 10 images

            inputs = train_features_tensor[i:i+10] #Get the next 10 images (batch)
            targets = train_targets_tensor[i:i+10] #Get the next 10 targets (batch)

            optimizer.zero_grad() #Zero gradients for every batch

            outputs = MLP_Variant1(inputs.float()) #Predictions for this batch

            #Compute loss and gradients
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step() #Adjust learning weights

            #Gather data
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        epoch_loss = running_loss / (len(train_features_tensor) // 10)
        epoch_accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%') #Print the training accuracy

    #Save the model
    torch.save(MLP_Variant1.state_dict(), "Saved_Models/mlp_variant1_model.pth")

    #*************************************************************************************

    #Create variant 2 MLP model
    MLP_Variant2 = MLP_Model_V2()
    #Set SGD optimizer
    optimizer = torch.optim.SGD(MLP_Variant2.parameters(), lr=0.1, momentum=0.9) 
    total = 0 #Total number of samples
    correct = 0 #Correct predictions made

    #Train variant 2 MLP model
    for epoch in range(EPOCHS):
        MLP_Variant2.train()
        running_loss = 0

        #Since train_features_tensors is a 2d array, we want to train for batches of 10 images
        for i in range(0, len(train_features_tensor), 10): #Batch size of 10 images

            inputs = train_features_tensor[i:i+10] #Get the next 10 images (batch)
            targets = train_targets_tensor[i:i+10] #Get the next 10 targets (batch)

            optimizer.zero_grad() #Zero gradients for every batch

            outputs = MLP_Variant2(inputs.float()) #Predictions for this batch

            #Compute loss and gradients
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step() #Adjust learning weights

            #Gather data
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        epoch_loss = running_loss / (len(train_features_tensor) // 10)
        epoch_accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%') #Print the training accuracy

    #Save the model
    torch.save(MLP_Variant2.state_dict(), "Saved_Models/mlp_variant2_model.pth")



#Testing function for all perceptron models
def perceptronModels(test_features_tensor, test_targets_tensor):

    #Create base model
    MLP_Base = MLP_Base_Model()
    #Load saved base model
    MLP_Base.load_state_dict(torch.load("Saved_Models/mlp_base_model.pth", weights_only=True))
    #Test the model
    MLP_Base.eval() 
    predictions_base= [] #All predicted targets
    test_targets = [] #All test targets

    with torch.no_grad():
        #Since test_features_tensors is a 2d array, we want to test for batches of 10 images
        for i in range(0, len(test_features_tensor), 10):  # Batch size of 10 images.
            inputs = test_features_tensor[i:i+10] #Get the next 10 images. (batch)
            targets = test_targets_tensor[i:i+10] #Get the next 10 targets. (batch)

            #Gather data
            outputs = MLP_Base(inputs.float())
            _, predicted = torch.max(outputs, 1)

            predictions_base.extend(predicted.cpu().numpy())  #Convert to CPU numpy array
            test_targets.extend(targets.cpu().numpy())  #Convert to CPU numpy array


    #Display confusion matrix for base mlp model
    confusion_mtx = confusion_matrix(test_targets, predictions_base)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=np.unique(test_targets))
    display.plot()
    plt.title("Multi-Layer Perceptron Base Model Confusion Matrix")
    plt.show()

    #Calculate metrics for base mlp model
    accuracy_base  = accuracy_score(test_targets, predictions_base)
    precision_base = precision_score(test_targets, predictions_base, average = "macro")
    recall_base = recall_score(test_targets, predictions_base, average = "macro")
    f1_base = f1_score(test_targets, predictions_base, average = "macro")

    #Format metrics as a tuple
    base_metrics = (accuracy_base, precision_base, recall_base, f1_base)

    #*************************************************************************************

    #Create variant 1 model
    MLP_Variant1 = MLP_Model_V1()
    #Load saved variant 1 model
    MLP_Variant1.load_state_dict(torch.load("Saved_Models/mlp_variant1_model.pth", weights_only=True))
    #Test the model
    MLP_Variant1.eval() 
    predictions_variant1= [] #All predicted targets
    test_targets = [] #All test targets

    with torch.no_grad():
        #Since test_features_tensors is a 2d array, we want to test for batches of 10 images
        for i in range(0, len(test_features_tensor), 10):  # Batch size of 10 images.
            inputs = test_features_tensor[i:i+10] #Get the next 10 images. (batch)
            targets = test_targets_tensor[i:i+10] #Get the next 10 targets. (batch)

            #Gather data
            outputs = MLP_Variant1(inputs.float())
            _, predicted = torch.max(outputs, 1)

            predictions_variant1.extend(predicted.cpu().numpy())  #Convert to CPU numpy array
            test_targets.extend(targets.cpu().numpy())  #Convert to CPU numpy array


    #Display confusion matrix for variant 1 mlp model
    confusion_mtx = confusion_matrix(test_targets, predictions_variant1)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=np.unique(test_targets))
    display.plot()
    plt.title("Multi-Layer Perceptron Variant 1 Model Confusion Matrix")
    plt.show()

    #Calculate metrics for variant 1 mlp model
    accuracy_v1  = accuracy_score(test_targets, predictions_variant1)
    precision_v1 = precision_score(test_targets, predictions_variant1, average = "macro")
    recall_v1 = recall_score(test_targets, predictions_variant1, average = "macro")
    f1_v1 = f1_score(test_targets, predictions_variant1, average = "macro")

    #Format metrics as a tuple
    variant1_metrics = (accuracy_v1, precision_v1, recall_v1, f1_v1)

    #*************************************************************************************

    #Create variant 2 model
    MLP_Variant2 = MLP_Model_V2()
    #Load saved variant 2 model
    MLP_Variant2.load_state_dict(torch.load("Saved_Models/mlp_variant2_model.pth", weights_only=True))
    #Test the model
    MLP_Variant2.eval() 
    predictions_variant2= [] #All predicted targets
    test_targets = [] #All test targets

    with torch.no_grad():
        #Since test_features_tensors is a 2d array, we want to test for batches of 10 images
        for i in range(0, len(test_features_tensor), 10):  # Batch size of 10 images.
            inputs = test_features_tensor[i:i+10] #Get the next 10 images. (batch)
            targets = test_targets_tensor[i:i+10] #Get the next 10 targets. (batch)

            #Gather data
            outputs = MLP_Variant2(inputs.float())
            _, predicted = torch.max(outputs, 1)

            predictions_variant2.extend(predicted.cpu().numpy())  #Convert to CPU numpy array
            test_targets.extend(targets.cpu().numpy())  #Convert to CPU numpy array


    #Display confusion matrix for variant 2 mlp model
    confusion_mtx = confusion_matrix(test_targets, predictions_variant2)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=np.unique(test_targets))
    display.plot()
    plt.title("Multi-Layer Perceptron Variant 2 Model Confusion Matrix")
    plt.show()

    #Calculate metrics for variant 2 mlp model
    accuracy_v2  = accuracy_score(test_targets, predictions_variant2)
    precision_v2 = precision_score(test_targets, predictions_variant2, average = "macro")
    recall_v2 = recall_score(test_targets, predictions_variant2, average = "macro")
    f1_v2 = f1_score(test_targets, predictions_variant2, average = "macro")

    #Format metrics as a tuple
    variant2_metrics = (accuracy_v2, precision_v2, recall_v2, f1_v2)

    return base_metrics, variant1_metrics, variant2_metrics #Return a tuple of tuples
    
#Display metrics functions
def displayPerceptronMetrics(perceptronMetrics):

    #Dictionary with data
    data = {
        "Model" : ["Perceptron Base Model", "Perceptron Variant 1 Model", "Perceptron Variant 2 Model"],
        "Accuracy" : [perceptronMetrics[0][0], perceptronMetrics[1][0], perceptronMetrics[2][0]],
        "Precision" : [perceptronMetrics[0][1], perceptronMetrics[1][1], perceptronMetrics[2][1]],
        "Recall" : [perceptronMetrics[0][2], perceptronMetrics[1][2], perceptronMetrics[2][2]], 
        "F1" : [perceptronMetrics[0][3], perceptronMetrics[1][3], perceptronMetrics[2][3]]
    }

    #Create table
    table = pd.DataFrame(data)
    #Display table
    print(table)
