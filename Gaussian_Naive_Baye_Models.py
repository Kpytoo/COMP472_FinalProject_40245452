import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
import matplotlib.pyplot as plt
import pandas as pd


#     *************************************
# <<< Gaussian Naive's Bayes implementation >>> 
#     *************************************

#Personal Gaussian Naive Bayes implementation
class gaussianBaye:
    
    def __init__(self):
        self.targets = None #Will contain all unique targets. (1d array)
        self.means = None #Will contain all the feature means for each target. (2d array)
        self.variances = None #Will contain all the features variances for each target. (2d array)
        self.priors = None #Will contain all target prior probabilities (1d array)
    
    #Fit method to train the model
    def fit(self, training_features, training_targets):
        self.targets = np.unique(training_targets) #Get all ten unique target values [0,...,9].
        self.priors = np.zeros(len(self.targets)) #1d array containing the prior probability of each class
        self.means = np.zeros((len(self.targets), training_features.shape[1])) #2d array - shape -> (10, 50) where 10 classes with 50 feature means each.
        self.variances = np.zeros((len(self.targets), training_features.shape[1])) #2d array - shape -> (10, 50) where 10 classes with 50 feature variances each.

        #Now we have to calculate the means, variances and prior probabilities.
        target_index = 0
        while target_index < len(self.targets):

            current_target = self.targets[target_index]

            target_vector_features = [] #Will contain all feature vectors of the current target type.
            for tgt in range(len(training_targets)): #For each feature vector in the training data.
                if training_targets[tgt] == current_target: #If the target is equal to the current target.
                    target_vector_features.append(training_features[tgt]) #Append the feature vector to the list.
            
            target_vector_features = np.array(target_vector_features) #Convert the list into a numpy array.

            
            #Calculate the means
            feature_means = target_vector_features.mean(axis=0) #Returns a 1d array.
            self.means[target_index, : ] = feature_means

            #Calculate the variances
            feature_variances = target_vector_features.var(axis=0) #Returns a 1d array.
            self.variances[target_index, : ] = feature_variances

            #Calculate the prior probabilities
            target_prior = target_vector_features.shape[0] / training_targets.shape[0] #Returns the prior probability.
            self.priors[target_index] = target_prior

            target_index = target_index + 1 #Increment to the next target.
        
    #Calculate the likelihood probabilities
    def likelihood(self, testing_vector, target_index):
        feature_mean = self.means[target_index] #Get all the feature means of the current target.
        feature_variance = self.variances[target_index] #Get all the feature variances of the current target.
        
        likelihood = -0.5 * np.log(2 * np.pi * feature_variance) - ((testing_vector - feature_mean) ** 2) / (2 * feature_variance) #Calculate the log likelihood.

        return likelihood #Return a 1d array.
    
    #Calculate the posterios probabilities
    def posterior(self, testing_vector):

        posteriors = [] #Will contain all 10 posterior probability for each target.

        target_index = 0
        while target_index < len(self.targets): 

            prior = np.log(self.priors[target_index]) #Get the appropriate prior probability of the target. Use log in case of small number.

            likelihood = self.likelihood(testing_vector, target_index) #Get the log likelihood of the target. It is returned as a 1d array.

            posterior = prior + np.sum(likelihood) #Calculate the posterior of the target.
            posteriors.append(posterior) #Add the target specific-posterior to the list.

            target_index = target_index + 1 #Go to the next target.
        
        return posteriors #Returns a list of all 10 target-specific posteriors.
    
    #Predict targets on test data
    def predict(self, testing_features):
        predicted_targets = []

        for testing_vector in testing_features: #Get the posteriors for all 10 targets.

            posteriors = self.posterior(testing_vector) #Get the list of all 10 targets.

            predicted_target = self.targets[np.argmax(posteriors)] #Get the highest probable target.
            predicted_targets.append(predicted_target) #Append the target to the list

        return np.array(predicted_targets) #Return all predicted targets as a numpy array.
    

#Function to train and save both models
def trainGaussianBayes(train_features_pca, train_targets):
    #Create my model
    gnb = gaussianBaye()
    #Train the model
    gnb.fit(train_features_pca, train_targets)
    #Save my trained model
    with open("Saved_Models/gaussian_baye_personal_model.pkl", "wb") as file1:
        pickle.dump(gnb, file1)

    #Creathe scikit model
    gnb_skt = GaussianNB()
    #Train the model
    gnb_skt.fit(train_features_pca, train_targets)
    #Save scikit trained model
    with open("Saved_Models/gaussian_baye_sktlearn_model.pkl", "wb") as file2:
        pickle.dump(gnb, file2)


#Personal Gaussian Naive Baye
def myGaussianBaye(test_features_pca, test_targets):
    #Create model
    gnb = gaussianBaye()
    #Load trained model
    with open("Saved_Models/gaussian_baye_personal_model.pkl", "rb") as file:
        gnb = pickle.load(file)
    #Predict on test data
    predictions = gnb.predict(test_features_pca)

    #Display confusion matrix
    confusion_mtx = confusion_matrix(test_targets, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=np.unique(test_targets))
    display.plot()
    plt.title("My Gaussian Baye Confusion Matrix")
    plt.show()

    #Compute and the metrics (Accuracy, precision, recall, F1)
    accuracy = accuracy_score(test_targets, predictions)
    precision = precision_score(test_targets, predictions, average="macro")
    recall = recall_score(test_targets, predictions, average="macro")
    f1 = f1_score(test_targets, predictions, average="macro")

    return accuracy, precision, recall, f1


#Scikit Gaussian Naive Bayes
def scikitGaussianBaye(test_features_pca, test_targets):
    #Creathe model
    gnb_skt = GaussianNB()
    #Load trained model
    with open("Saved_Models/gaussian_baye_sktlearn_model.pkl", "rb") as file:
        gnb_skt = pickle.load(file)
    #Predict on test data
    predictions = gnb_skt.predict(test_features_pca)

    #Display confusion matrix
    confusion_mtx = confusion_matrix(test_targets, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=np.unique(test_targets))
    display.plot()
    plt.title("Scikit Gaussian Baye Confusion Matrix")
    plt.show()

    #Compute and the metrics (Accuracy, precision, recall, F1)
    accuracy = accuracy_score(test_targets, predictions)
    precision = precision_score(test_targets, predictions, average="macro")
    recall = recall_score(test_targets, predictions, average="macro")
    f1 = f1_score(test_targets, predictions, average="macro")

    return accuracy, precision, recall, f1


#Display metrics function
def displayBayesMetrics(myBaye, scikitBaye):

    #Dictionary with data
    data = {
        "Model" : ["My Gaussian Baye", "Scikit Gaussian Baye"],
        "Accuracy" : [myBaye[0], scikitBaye[0]],
        "Precision" : [myBaye[1], scikitBaye[1]],
        "Recall" : [myBaye[2], scikitBaye[2]],
        "F1" : [myBaye[3], scikitBaye[3]],
    }

    #Create table
    table = pd.DataFrame(data)
    #Display table
    print(table)
    