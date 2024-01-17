'''
Here are codes for evaluating the model, not for users to use.
If you are interested how our model was evaluated, you can take a look.
'''



######### Import packages #########
# For model using
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision.models import resnet50, densenet121
import torch.nn.functional as F
# For evaluating classification performance and figure presentation
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
# Other tools
import os
import numpy as np
from PIL import Image



######### Define the Net class #########
class Net(nn.Module):
    def __init__(self):
        '''
        @brief:
            Initialize the CNN architecture using the pre-trained model
        '''
        super(Net, self).__init__()
        self.resnet = densenet121()
        self.resnet.fc = nn.Linear(2048, 2)

    # The foward method transforms inputs to output predictions
    def forward(self, x):
        '''
        @brief:
            Forward pass through the CNN
        @args:
            x: Input data
        @returns:
            Output of the CNN
        '''
        return self.resnet(x)


model_path = 'model/path'  #please input the model path
benign_dir = 'benign/directory'  #please input the path of test image files (benign)
malignant_dir = 'malignant/directory'  #please input the path of test image files (malignant)


model = torch.load(model_path, map_location='cpu') #load our model
model.eval() #set our model to evaluation mode


# Define the transformation for the test images
transform = transforms.Compose([
    transforms.Resize([224, 224]), #resizes input images to a fixed size (224x224 pixels)
    transforms.ToTensor(), #convert the image to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #normalize tensor values
])



# Set empty lists to store the true labels, predicted labels, predicted probabilities
true_labels_benign = []
predicted_labels_benign = []
true_labels_malignant = []
predicted_labels_malignant = []
predicted_probs = []
# For later interpretation of the results
tumor_type = ["benign", "malignant"]



######### Run over test  (benign) #########
# Go through all the images in the benign directory
for filename in os.listdir(benign_dir):
    img_path = os.path.join(benign_dir, filename)

    try:#try to open the image and preprocess it
        img = torch.unsqueeze(transform(Image.open(img_path).convert("RGB")), dim=0) #read, pre-processe, and convert the image to a tensor
    except (Image.UnidentifiedImageError, OSError): #if there is an issue, then skip it and print the following message
        print(f"Skipping {img_path} as it is not a valid image file.")
        continue

    with torch.no_grad(): #prevent PyTorch from tracking operations
        output = model(img) #get the output through the model
        probabilities = F.softmax(output, dim=1) #get class probabilities
        pred = torch.argmax(model(img), dim=-1).cpu().numpy()[0] #determines the predicted class

    true_label = 0 #all the images in this directory are benign

    true_labels_benign.append(true_label) #add the true label to the list of true labels
    predicted_labels_benign.append(pred) #add the predicted label to the list of predicted labels
    predicted_probs.append(probabilities[0, 1].item()) #add the probability of belonging to malignant  to the list of predicted probabilities



######### Run over test  (malignant) #########
# After evaluation we found 2 false negative predictions, so we create a list to store the two images
false_negative_filenames = []
# Similar to the previous section
for filename in os.listdir(malignant_dir):
    img_path = os.path.join(malignant_dir, filename)

    try:
        img = torch.unsqueeze(transform(Image.open(img_path).convert("RGB")), dim=0)
    except (Image.UnidentifiedImageError, OSError):
        print(f"Skipping {img_path} as it is not a valid image file.")
        continue

    with torch.no_grad():
        output = model(img)
        probabilities = F.softmax(output, dim=1)
        pred = torch.argmax(output, dim=-1).cpu().numpy()[0]

    true_label = 1 #all the images in this directory are benign

    true_labels_malignant.append(true_label)
    predicted_labels_malignant.append(pred)
    predicted_probs.append(probabilities[0, 1].item())



######### Check  false negatives #########
# After evaluation, we found two false negatives
# We want to present the figures
    if true_label == 1 and pred == 0: #true label is malignant and predicted label is benign
        false_negative_filenames.append(filename) #add the file name to the list


# Calculate the overall true labels and predicted labels
true_labels = true_labels_benign + true_labels_malignant
predicted_labels = predicted_labels_benign + predicted_labels_malignant


# Make the confusion matrix and calculate the true negative, false positive, false negative and true positive
confusion_mat = confusion_matrix(true_labels, predicted_labels)
tn, fp, fn, tp = confusion_mat.ravel()

######### Calculate  indicators #########
# Print indicators and present the figures
print('Accuracy:', accuracy_score(true_labels, predicted_labels))
print('Recall:', recall_score(true_labels, predicted_labels))
print('Precision:', precision_score(true_labels, predicted_labels))
print('Specificity:', tn / (tn + fp))
print('F1 score:', f1_score(true_labels, predicted_labels))
print('AUC:', roc_auc_score(true_labels, predicted_probs))

# Present the confusion matrix and add labels
print("Confusion Matrix:")
print(confusion_mat)
disp = ConfusionMatrixDisplay(confusion_mat)
disp.plot()
plt.show()
print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=tumor_type))

######### Calculate the Receiver Operating Characteristic (ROC) curve and its AUC (area under curve) #########
def draw_roc(true_labels, predicted_probs):
    '''
    @brief:
        Plot the ROC curve based on true labels and predicted probabilities
    @args:
        true_labels: True labels of the data
        predicted_probs: Predicted probabilities for the positive class
    '''
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs) #compute the ROC curve (fpr: False Positive Rate; tpr: True Positive Rate;  _: additional information, not used here)
    roc_auc = auc(fpr, tpr) #compute auc
    plt.figure()
    plt.plot(fpr, tpr,
             lw=3, label='ROC curve (area = %0.2f)' % roc_auc) #plot the curve and label
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--') #plot a diagonal line representing a random classifier
    # Set the limits and labels for x-axis and y-axis, set the title and legends
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")

    plt.show() #show the plot

# Draw the ROC curve
draw_roc(true_labels, predicted_probs)

# Show the false negative predictions
filter(lambda x:x, false_negative_filenames)  #get rid of the blank values
print(false_negative_filenames)
# Show the images corresponding to false negatives (similar to previous parts)
for filename in false_negative_filenames:
    img_path = os.path.join(malignant_dir, filename)

    try:
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(f'File Name: {filename}')
        plt.show()

    except (Image.UnidentifiedImageError, OSError):
        print(f"Skipping {img_path} as it is not a valid image file.")