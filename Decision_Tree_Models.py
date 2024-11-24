import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

#     *******************************************
# <<< Decision Tree Classification implementation >>> 
#     *******************************************

#Personal Decision Tree Classification implementation
#Lots of help from this source: https://medium.com/@enozeren/building-a-decision-tree-from-scratch-324b9a5ed836

#Node class
class Node():
    def __init__(self, gini_value , predicted_target, nb_of_images, images_each_target):
        self.gini = gini_value
        self.predicted_target = predicted_target
        self.nb_of_images = nb_of_images
        self.images_each_target = images_each_target
        self.left = None
        self.right = None
        self.feature_idx = None
        self.threshold = None
        return

#Decision Tree class
class decisionTree():

    def __init__(self, max_depth):
        self.max_depth = max_depth 
        self.root_node = None #Will store the root node
        return
    
    #Function to calculate gini of a node
    def calculate_gini(self, targets):
        
        nb_images = len(targets)
        #If there are no images in the node
        if nb_images == 0:
            return 0
        
        #Calculate the gini value 
        target_counts = np.bincount(targets) #An array that contains all target counts
        target_probabilities = target_counts / nb_images #An array that contains all target probabilities
        return 1 - np.sum(target_probabilities**2) #Calculate the gini value and return
    
    #Splitting function
    def split(self, features, targets, feature_idx, threshold):

        left_idxs = features[:, feature_idx] <= threshold #Returns an array of true and false given the condition
        right_idxs = ~left_idxs #Returns an array of true and false opposite to "left_idxs"

        return features[left_idxs], features[right_idxs], targets[left_idxs], targets[right_idxs] #Return the appropriate split left and right images and targets
    
    #Function that finds the best split
    def find_best_split(self, features, targets):
        nb_features = features.shape[1] #Total number of features (50 in our case)
        best_gini_value = 1.0 #Initiate a value of 1, so we can compare it later and change it
        best_feature_idx = None #Indicates the best feature found for the split
        best_threshold = None #Indicates the best threshold found for the feature
        best_splits = None #Tuple that holds the best split

        for feature_idx in range(nb_features):
            all_values = np.unique(features[:, feature_idx]) #Get all values for the specific feature we are now iterating

            for threshold in all_values: #For each threshold in a specific feature, calculate its gini value
                left_features, right_features, left_targets, right_targets = self.split(features, targets, feature_idx, threshold)
                gini_left = self.calculate_gini(left_targets) #Calculate the gini value on the left split
                gini_right = self.calculate_gini(right_targets) #Calculate the gini value on the right split
                total_weighted_gini = (len(left_targets)/len(targets) * gini_left) + (len(right_targets)/len(targets) * gini_right) #Calculate the total weighted gini value

                #Update the "best" variables if calculated gini is better than the last one
                if total_weighted_gini < best_gini_value:
                    best_gini_value = total_weighted_gini
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    best_splits = (left_features, right_features, left_targets, right_targets)
                
        return best_feature_idx, best_threshold, best_splits #Return the best split
    
    #Building function
    def tree_build(self, features, targets, depth):
        nb_images = len(targets) #Number of total images
        nb_targets = len(np.unique(targets)) #Number of unique targets
        nb_images_each_target = np.bincount(targets, minlength=nb_targets) #An array containing the number of images for of each target
        predicted_target = np.argmax(nb_images_each_target) #The predicted target for this node

        #Create our node
        node = Node(
            gini_value = self.calculate_gini(targets),
            predicted_target = predicted_target,
            nb_of_images = nb_images,
            images_each_target = nb_images_each_target
        )

        #Check whether we have a gini of 0 or if we reached the maximum depth before splitting
        if node.gini == 0 or depth >= self.max_depth:
            return node
        
        #Now let's find the best split for this node
        feature_idx, threshold, splits = self.find_best_split(features, targets)
        if splits == None: #If no splits were found
            return node
        
        #Using recursion to continue building our tree
        left_features, right_features, left_targets, right_targets = splits #Unpack the tuple containing our split data
        node.feature_idx = feature_idx
        node.threshold = threshold

        node.left = self.tree_build(left_features, left_targets, depth + 1) #Build left node
        node.right = self.tree_build(right_features, right_targets, depth + 1) #Build right node

        return node
    
    #Build the tree (train the data)
    def fit(self, training_features, training_targets):

        #Fit the model by starting at the root node
        self.root_node = self.tree_build(training_features, training_targets, depth = 0)
    
    #Function to predict the target of one image
    def predict_image(self, node, image):

        #If the node is a leaf
        if node.left is None and node.right is None:
            return node.predicted_target
        
        #Use recursion to go through the tree and predict the target
        if image[node.feature_idx] <= node.threshold:
            return self.predict_image(node.left, image)
        else:
            return self.predict_image(node.right, image)


    def predict(self, testing_features):

        predicted = [] #List will contain all predicted targets

        for image in testing_features:
            
            prediction = self.predict_image(self.root_node, image)

            predicted.append(prediction)

        return predicted
    

#Function to train and save all models and their variations
def trainDecisionTrees(train_features_pca, train_targets):
    #Create my model with max_depth 50
    dt50 = decisionTree(max_depth = 50)
    #Train the model
    dt50.fit(train_features_pca, train_targets)
    #Save my trained model
    with open("Saved_Models/decision_tree_personal_depth50_model.pkl", "wb") as file1:
        pickle.dump(dt50, file1)

    #Create my model variation with max_depth 20
    dt20 = decisionTree(max_depth = 20)
    #Train the model
    dt20.fit(train_features_pca, train_targets)
    #Save my trained model
    with open("Saved_Models/decision_tree_personal_variation_depth20_model.pkl", "wb") as file2:
        pickle.dump(dt20, file2)

    #Create my model variation with max_depth 80
    dt80 = decisionTree(max_depth = 80)
    #Train the model
    dt80.fit(train_features_pca, train_targets)
    #Save my trained model
    with open("Saved_Models/decision_tree_personal_variation_depth80_model.pkl", "wb") as file3:
        pickle.dump(dt80, file3)

    # ******************************************************************************************

    #Create scikit model with depth 50
    dt50_skt = DecisionTreeClassifier(criterion='gini', max_depth=50)
    #Train the model
    dt50_skt.fit(train_features_pca, train_targets)
    #Save scikit trained model
    with open("Saved_Models/decision_tree_sktlearn_depth50_model.pkl", "wb") as file4:
        pickle.dump(dt50_skt, file4)
    
    #Create scikit model variation with depth 20
    dt20_skt = DecisionTreeClassifier(criterion='gini', max_depth=20)
    #Train the model
    dt20_skt.fit(train_features_pca, train_targets)
    #Save scikit trained model
    with open("Saved_Models/decision_tree_sktlearn_variation_depth20_model.pkl", "wb") as file5:
        pickle.dump(dt20_skt, file5)
    
    #Create scikit model variation with depth 80
    dt80_skt = DecisionTreeClassifier(criterion='gini', max_depth=80)
    #Train the model
    dt80_skt.fit(train_features_pca, train_targets)
    #Save scikit trained model
    with open("Saved_Models/decision_tree_sktlearn_variation_depth80_model.pkl", "wb") as file6:
        pickle.dump(dt80_skt, file6)


#Personal Decision Tree
def myDecisionTrees(test_features_pca, test_targets):
    
    #Create model depth 50
    dt50 = decisionTree(max_depth = 50)
    #Load trained model
    with open("Saved_Models/decision_tree_personal_depth50_model.pkl", "rb") as file1:
        dt50 = pickle.load(file1)
    
    #Predict on test data
    predictions_base = dt50.predict(test_features_pca)

    #Display confusion matrix
    confusion_mtx = confusion_matrix(test_targets, predictions_base)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=np.unique(test_targets))
    display.plot()
    plt.title("My Decision Tree Depth 50 Confusion Matrix")
    plt.show()

    #Calculate metrics for the decision tree
    accuracy_base = accuracy_score(test_targets, predictions_base)
    precision_base = precision_score(test_targets, predictions_base, average="macro")
    recall_base = recall_score(test_targets, predictions_base, average="macro")
    f1_base = f1_score(test_targets, predictions_base, average="macro")

    #Format each metrics as tuple
    base_metrics = (accuracy_base, precision_base, recall_base, f1_base)

    
    #Create model variant depth 20
    dt20 = decisionTree(max_depth = 20)
    #Load trained model
    with open("Saved_Models/decision_tree_personal_variation_depth20_model.pkl", "rb") as file2:
        dt20 = pickle.load(file2)

    #Predict on test data
    predictions_variant1 = dt20.predict(test_features_pca)

    #Display Confusion matrix variant 1
    confusion_mtx1 = confusion_matrix(test_targets, predictions_variant1)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=np.unique(test_targets))
    display.plot()
    plt.title("My Decision Tree Variant Depth 20 Confusion Matrix")
    plt.show()

    #Calculate metrics for the decision tree variant 1
    accuracy_v1 = accuracy_score(test_targets, predictions_variant1)
    precision_v1 = precision_score(test_targets, predictions_variant1, average="macro")
    recall_v1 = recall_score(test_targets, predictions_variant1, average="macro")
    f1_v1 = f1_score(test_targets, predictions_variant1, average="macro")

    #Format each metrics as tuple
    variant1_metrics = (accuracy_v1, precision_v1, recall_v1, f1_v1)


    #Create model variant depth 80
    dt80 = decisionTree(max_depth = 80)
    #Load trained model
    with open("Saved_Models/decision_tree_personal_variation_depth80_model.pkl", "rb") as file3:
        dt80 = pickle.load(file3)

    #Predict on test data
    predictions_variant2 = dt80.predict(test_features_pca)

    #Display Confusion matrix variant 2
    confusion_mtx = confusion_matrix(test_targets, predictions_variant2)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=np.unique(test_targets))
    display.plot()
    plt.title("My Decision Tree Variant Depth 80 Confusion Matrix")
    plt.show()

    #Calculate metrics for the decision tree variant 2
    accuracy_v2 = accuracy_score(test_targets, predictions_variant2)
    precision_v2 = precision_score(test_targets, predictions_variant2, average="macro")
    recall_v2 = recall_score(test_targets, predictions_variant2, average="macro")
    f1_v2 = f1_score(test_targets, predictions_variant2, average="macro")

    #Format each metrics as tuple
    variant2_metrics = (accuracy_v2, precision_v2, recall_v2, f1_v2)

    return base_metrics, variant1_metrics, variant2_metrics #Return a tuple of tuples


#Scikit Decision Tree
def scikitDecisionTrees(test_features_pca, test_targets):

    #Create model depth 50
    dt50_skt = DecisionTreeClassifier(criterion='gini', max_depth = 50)
    #Load trained model
    with open("Saved_Models/decision_tree_sktlearn_depth50_model.pkl", "rb") as file1:
        dt50_skt = pickle.load(file1)
    
    #Create model variant depth 20
    dt20_skt = DecisionTreeClassifier(criterion='gini', max_depth = 20)
    #Load trained model
    with open("Saved_Models/decision_tree_sktlearn_variation_depth20_model.pkl", "rb") as file2:
        dt20_skt = pickle.load(file2)

    #Create model variant depth 80
    dt80_skt = DecisionTreeClassifier(criterion='gini', max_depth = 80)
    #Load trained model
    with open("Saved_Models/decision_tree_sktlearn_variation_depth80_model.pkl", "rb") as file3:
        dt80_skt = pickle.load(file3)

    #Predict on test data
    predictions_base = dt50_skt.predict(test_features_pca)
    predictions_variant1 = dt20_skt.predict(test_features_pca)
    predictions_variant2 = dt80_skt.predict(test_features_pca)

    #Display confusion matrix
    confusion_mtx = confusion_matrix(test_targets, predictions_base)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=np.unique(test_targets))
    display.plot()
    plt.title("Scikit Decision Tree Depth 50 Confusion Matrix")
    plt.show()
    #Display Confusion matrix variant 1
    confusion_mtx = confusion_matrix(test_targets, predictions_variant1)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=np.unique(test_targets))
    display.plot()
    plt.title("Scikit Decision Tree Variant Depth 20 Confusion Matrix")
    plt.show()
    #Display Confusion matrix variant 2
    confusion_mtx = confusion_matrix(test_targets, predictions_variant2)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=np.unique(test_targets))
    display.plot()
    plt.title("Scikit Decision Tree Variant Depth 80 Confusion Matrix")
    plt.show()

    #Calculate metrics for the decision tree
    accuracy_base = accuracy_score(test_targets, predictions_base)
    precision_base = precision_score(test_targets, predictions_base, average="macro")
    recall_base = recall_score(test_targets, predictions_base, average="macro")
    f1_base = f1_score(test_targets, predictions_base, average="macro")
    #Calculate metrics for the decision tree variant 1
    accuracy_v1 = accuracy_score(test_targets, predictions_variant1)
    precision_v1 = precision_score(test_targets, predictions_variant1, average="macro")
    recall_v1 = recall_score(test_targets, predictions_variant1, average="macro")
    f1_v1 = f1_score(test_targets, predictions_variant1, average="macro")
    #Calculate metrics for the decision tree variant 2
    accuracy_v2 = accuracy_score(test_targets, predictions_variant2)
    precision_v2 = precision_score(test_targets, predictions_variant2, average="macro")
    recall_v2 = recall_score(test_targets, predictions_variant2, average="macro")
    f1_v2 = f1_score(test_targets, predictions_variant2, average="macro")

    #Format each metrics as tuples
    base_metrics = (accuracy_base, precision_base, recall_base, f1_base)
    variant1_metrics = (accuracy_v1, precision_v1, recall_v1, f1_v1)
    variant2_metrics = (accuracy_v2, precision_v2, recall_v2, f1_v2)

    return base_metrics, variant1_metrics, variant2_metrics #Return a tuple of tuples


#Display metrics function
def displayDecisionTreeMetrics(myTree, scikitTree):

    #Dictionary with data
    data = {
        "Model" : ["My Decision Tree", "My Decision Tree Variant 1", "My Decision Tree Variant 2",
                   "Scikit Decision Tree", "Scikit Decision Tree Variant 1", "Scikit Decision Tree Variant 2"],
        "Accuracy" : [myTree[0][0], myTree[1][0], myTree[2][0],
                      scikitTree[0][0], scikitTree[1][0], scikitTree[2][0]],
        "Precision" : [myTree[0][1], myTree[1][1], myTree[2][1],
                      scikitTree[0][1], scikitTree[1][1], scikitTree[2][1]],
        "Recall" : [myTree[0][2], myTree[1][2], myTree[2][2],
                      scikitTree[0][2], scikitTree[1][2], scikitTree[2][2]],
        "F1" : [myTree[0][3], myTree[1][3], myTree[2][3],
                      scikitTree[0][3], scikitTree[1][3], scikitTree[2][3]]
    }

    #Create table
    table = pd.DataFrame(data)
    #Display table
    print(table)