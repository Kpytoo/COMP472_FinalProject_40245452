import torch
from torch.utils.data import Subset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from sklearn.decomposition import PCA
import Gaussian_Naive_Baye_Models
import Decision_Tree_Models
import Multi_Layer_Perceptron_Models
import Convolutional_Neural_Network_Models



#     *************
# <<< Main Function >>>
#     *************

#Main function
def main():

    #     *******************
    # <<< Import CIFAR10 data >>>
    #     *******************

    print("<<< Importing CIFAR10 data... >>>")

    transform = transforms.Compose([ 
        transforms.Resize((224, 224)), #Resize the images to 224x224.
        transforms.ToTensor(), #Translate images in term of tensors.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #Normalize the images for ResNet-18.
    ])

    #Our Train data - CIFAR10. (50,000 train images) - Returns a list of tuples, ex: [ (Tensor, target), ...]
    train_data = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    #Our Test data - CIFAR10. (10,000 test images)
    test_data = torchvision.datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    #Function that returns a subset of a dataset.
    def dataSubset(data_set, num_of_images):

        #Create a dictionary that has 10 empty lists.
        class_targets = {} 
        for i in range(10):
            class_targets[i] = []

        #While loop that iterates over the dataset.
        index = 0
        while index < len(data_set):
            target = data_set[index][1] #Extract tuple. Target is only needed. --> (Tensor, target)

            if len(class_targets[target]) < num_of_images: #If the list of the target hasn't reached num_of_images,
                class_targets[target].append(index)        #we append the index of the image in the dataset to the list.

            #Check if every target has the num_of_images required for the subset.
            classes_full = True
            for lis in class_targets.values():
                if len(lis) < num_of_images:
                    classes_full = False
                    break
            
            #Break out of the while loop if all targets reached the required num_of_images.
            if classes_full:
                break

            #Go to the next item in the dataset.
            index = index + 1

        #Create a list that will host all images in the subset.
        subset_images = []
        for lis in class_targets.values():
            for img in lis:
                subset_images.append(img)

        #Return the subset of data.
        return Subset(data_set, subset_images)


    #Create a subset of training data, 500 images from each target. Returns a list of tuples, ex: [ (Tensor, target), ...]
    train_data_subset = dataSubset(train_data, 500) 

    #Create a subset of testing data, 100 images from each target.
    test_data_subset = dataSubset(test_data, 100) 


    #Create a data loader for the training subset data. Returns an iterable DataLoader object.
    train_loader = DataLoader(train_data_subset, batch_size = 10, shuffle = True)

    #Create a data loader for the testing subset data.
    test_loader = DataLoader(test_data_subset, batch_size = 10, shuffle = False)


    #     *******************************************************
    # <<< Use a pretrained ResNet-18 model for feature extraction >>>
    #     *******************************************************

    print("<<< Using a pretrained ResNet-18 model for feature extraction... >>>")

    #Import the pretrained ResNet-18 model.
    resnet18 = models.resnet18(weights="IMAGENET1K_V1")

    #Remove the last layer from the ResNet-18 model (the fc layer). Creates a list of resnet children and removes the last one.
    new_resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1])

    #Set to evaluation mode for testing.
    new_resnet18.eval()


    #Function that will extract features from a data_loader.
    def featureExtraction(model, data_loader):
        features = [] #List that will contain 2 dimensional extracted feature vectors.
        targets = [] #The targets of each feature.

        #Don't consider gradients while extracting features.
        with torch.no_grad():
            #Get images and targets from the data_loader. image tensor shape --> tensor[batch_size, 3, 224, 224]
            for images, target in data_loader: #Dataloader is an "iterator" of "lists", where each list contains 2 tensors, first being the images, second being the targets.
                #Make sure we are using the cpu.
                images = images.to('cpu') 
                #Extract the feature of the image through ResNet-18 as a numpy vector. reduce tensor to 2 dimensions and convert to numpy.
                feature = (model(images).squeeze()).numpy() #feature is now a 2d numpy array
                #Append the extracted feature vector to the features list.
                features.append(feature)
                #Append the targets vector to the targets list.
                targets.append(target.numpy())
        #Return features as a two dimensional array using vstack and targets as a one dimensional array using hstack.
        return np.vstack(features), np.hstack(targets)

    # Get extracted features and targets from the training data.
    train_features, train_targets = featureExtraction(new_resnet18, train_loader) #features: (5000, 512) targets: (5000)

    # Get extracted features and targets from the testing data.
    test_features, test_targets = featureExtraction(new_resnet18, test_loader) #features: (1000, 512) targets: (1000)


    #     *************************************************
    # <<< Use PCA to further reduce the size of our vectors >>>
    #     *************************************************

    print("<<< Using PCA to further reduce the size of our vectors... >>>")

    #Create our Principal Component Analysis object.
    pca = PCA(n_components=50)

    #Reducing the size of the image vectors inside train_features to 50x1, with fit.
    train_features_pca = pca.fit_transform(train_features) #features: from (5000, 512) to --> (5000, 50)

    #Reducing the size of the image vectors inside test_features to 50x1, without fit.
    test_features_pca = pca.transform(test_features) #features: from (1000, 512) to --> (1000, 50)
    

    #     *************************************
    # <<< Gaussian Naive's Bayes implementation >>> 
    #     *************************************

    print("<<< Gaussian Naive Bayes Models... >>>")
    # Gaussian_Naive_Baye_Models.trainGaussianBayes(train_features_pca, train_targets)    # ---------- Uncomment to retrain all models ---------
    my_gnb_metrics = Gaussian_Naive_Baye_Models.myGaussianBaye(test_features_pca, test_targets)
    scikit_gnb_metrics = Gaussian_Naive_Baye_Models.scikitGaussianBaye(test_features_pca, test_targets)
    Gaussian_Naive_Baye_Models.displayBayesMetrics(my_gnb_metrics, scikit_gnb_metrics)
    print()

    #     *******************************************
    # <<< Decision Tree Classification implementation >>> 
    #     *******************************************
    
    print("<<< Decision Tree Classification models... >>>")
    # Decision_Tree_Models.trainDecisionTrees(train_features_pca, train_targets)    # ---------- Uncomment to retrain all models ---------
    my_dt_metrics = Decision_Tree_Models.myDecisionTrees(test_features_pca, test_targets)
    scikit_dt_metrics = Decision_Tree_Models.scikitDecisionTrees(test_features_pca, test_targets)
    Decision_Tree_Models.displayDecisionTreeMetrics(my_dt_metrics, scikit_dt_metrics)
    print()

    #     *************************************
    # <<< Multi-Layer Perceptron implementation >>> 
    #     *************************************

    #Revert our data to tensors instead of numpy for the MLP
    print("<<< Reverting data from PCA to tensors for MLPs... >>>")
    train_features_tensor = torch.from_numpy(train_features_pca)
    train_targets_tensor = torch.from_numpy(train_targets)
    test_features_tensor = torch.from_numpy(test_features_pca)
    test_targets_tensor = torch.from_numpy(test_targets)
    print("<<< Multi-Layer Perceptron models... >>>")
    # Multi_Layer_Perceptron_Models.trainPerceptronModels(train_features_tensor, train_targets_tensor)    # ---------- Uncomment to retrain all models ---------
    perceptron_metrics = Multi_Layer_Perceptron_Models.perceptronModels(test_features_tensor, test_targets_tensor)
    Multi_Layer_Perceptron_Models.displayPerceptronMetrics(perceptron_metrics)
    print()


    #     *******************************************
    # <<< Convolutional Neural Network implementation >>> 
    #     *******************************************

    print("<<< Convolutional Neural Network models... >>>")
    # Convolutional_Neural_Network_Models.trainVGG11Models(train_loader) # ---------- Uncomment to retrain all models ---------
    cnn_metrics = Convolutional_Neural_Network_Models.cnnModels(test_loader)
    Convolutional_Neural_Network_Models.displayCNNMetrics(cnn_metrics)
    print()
    
    

#Running this file (Main_Driver.py)
if __name__ == "__main__":
    main()
