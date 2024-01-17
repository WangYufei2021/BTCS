import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import io, morphology,measure
import os
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision.models import densenet121
import cv2
import PySimpleGUI as sg
import imghdr
# Import each step of segmentation 
from Segmentation import parameter_defining, preprocess_image,segment_image,labeling_picture,filter_regions,remove_inner_objects,instance_segmentation,get_edge
# NETS: densenet121
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.resnet = densenet121()
        self.resnet.fc = nn.Linear(2048, 2)

    def forward(self,x):
        return self.resnet(x)

model = Net()

#Create GUI  Window
sg.SetOptions(
font=("Arial", 18), 
auto_size_buttons=True, 
button_color=("#ebf5d8", "#9db898"), 
background_color="#ebf5d8",
element_background_color="#ebf5d8",
text_element_background_color="#ebf5d8",
text_color="#5a7b29",
element_padding=(8, 8)
)


layout = [
    [sg.Text("Choose an image"),sg.FileBrowse(key="input_image")],
    [sg.HorizontalSeparator()],
    [sg.Text("Choose a model for classification"),sg.FileBrowse(key="input_model"),sg.Button("Classification")],
    [sg.HorizontalSeparator()],
    [sg.Text("You think the number of cells (n) is (few: n<=15, median:15<n<50, many:n>=50)"), sg.Combo(["few", "median", "many"], key="number_user_given", default_value="median", readonly=True)], 
    [sg.Column([[sg.Checkbox("Measure cell number?", default=False, key="if_measure_cells"), sg.VerticalSeparator(), sg.Checkbox("Measure nucleus region (area)?", default=False, key="if_measure_regions"),
                 sg.VerticalSeparator(),sg.Checkbox("Calculate nucleus length?", default=False, key="if_calculate_length"), sg.VerticalSeparator(), sg.Checkbox("Measure nucleus eccentricity?", default=False, key="if_measure_eccentricity")]])],
    [sg.Checkbox("Plot image?", default=False, key="plot_image"), sg.Checkbox("Plot all image (show all processing steps)?", default=False, key="plot_all_image")],
    [sg.Checkbox("Save to file", default=False, key="save_file")],
    [sg.Text("Output file name\n"+" (If not given, default is the original name with '_segmented' at the end)",justification='center'),sg.InputText(default_text="", key="output_file")],      
    [sg.Text("Choose a folder to save\n"+"(If not given, default is the uploaded image folder)",justification='center'), sg.FolderBrowse(key="save_path")],
    [sg.Button("Segmentation")],
    [sg.Button("Exit")]
    ]

window = sg.Window("BTCS",layout=layout,element_justification='center')

while(True):
    event, values = window.read()
    #Variables are assigned values for subsequent processing
    image_path = values["input_image"]
    model_path = values["input_model"]
    User_given = values["number_user_given"]
    image_savename = values["output_file"]
    if_save = values["save_file"]
    if_save_path = values["save_path"]
    decide_if_save_path = ''
    if_plot = values["plot_image"]
    if_plot_all = values["plot_all_image"]
    decide_if_plot_all = ''
    if_measure_cells = values["if_measure_cells"]
    if_measure_regions = values["if_measure_regions"]
    if_calculate_length = values["if_calculate_length"]
    if_measure_eccentricity = values["if_measure_eccentricity"]

    def nuclei_segmentation(image_path,User_given,Plot='one',save=False, save_path=None,Measure_cells=False,Measure_regions=False,Calculate_length=False,Measure_eccentricity=False):
        """
        @brief:
            Nuclei segmentation undergoes preprocessing, segmentation, binary labeling, closing, removing inner objects, instance segmentation, filter regions, and getting edges. 
        @args:
            image_path: The image path.
            User_given: User given input estimation. "Few": <15 cells; "median": 15~50 cells; "many": > 50 cells
            Plot: Do you want to plot only the final edged image ("one"), or the whole processing steps ("all"). 
        @returns:
            Nothing, directly plot or print results.
        """
        global image_savename
        r, r2, k = parameter_defining(User_given)

        image = io.imread(image_path)

        processed_image = preprocess_image(image)

        segmented = segment_image(processed_image, k=k)

        binary_segmented=labeling_picture(processed_image,segmented)

        closed_image = morphology.binary_closing(binary_segmented, footprint=morphology.disk(r))

        removed_image=remove_inner_objects(closed_image)

        # if User given == 'few', this step will not be processed. We implement this through several steps to find the optimal results
        labels=instance_segmentation(removed_image,User_given) 

        updated_regions=filter_regions(labels,User_given)

        Last_image = morphology.binary_closing(updated_regions, footprint=morphology.disk(r2))

        second_removed_image=remove_inner_objects(Last_image)

        edged_image=get_edge(image, second_removed_image)

        label_image = measure.label(second_removed_image)
        region_props = measure.regionprops(label_image)
        
        
        Results = ""
        areas = np.array([region.area for region in region_props])
        if Measure_cells:
            Results += f"The number of cells (nuclei): {len(areas)}\n"
        if Measure_regions:
            Results += f"The mean area of nuclei is: {round(np.mean(areas),2)}\n"
            Results += f"The median area of nuclei is: {round(np.median(areas),2)}\n"
        if Calculate_length:
            majors=[region.major_axis_length for region in region_props]
            minors=[region.minor_axis_length for region in region_props]
            Results += f"The mean length of the major axis of the ellipse nuclei is: {round(np.mean(majors),2)}; The median is: {round(np.median(majors),2)}\n"
            Results += f"The mean length of the minor axis of the ellipse nuclei is: {round(np.mean(minors),2)}; The median is: {round(np.median(minors),2)}\n"
        if Measure_eccentricity:
            Eccentricities = np.array([region.eccentricity for region in region_props])
            Results += f"The mean eccentricity of nuclei is: {round(np.mean(Eccentricities),2)}\n"
            Results += f"The median eccentricity of nuclei is: {round(np.median(Eccentricities),2)}\n"
        if not (Measure_cells or Measure_regions or Calculate_length or Measure_eccentricity):
            sg.popup("You didn't choose any function",title="Reminder")
        sg.popup(Results, title="Result of Segmentation")


        if save: 
            if save_path==None:
                directory = os.path.dirname(image_path)
        # The file will be stored in where the original image stays with "_segmented" at the end.
                if image_savename == '':
                    file_name = os.path.basename(image_path)
                    save_list=file_name.split(".")
                    # If the filename contains more than one dot, we need to collect them as list for the subsequent processing.
                    format=save_list[-1]
                    save_list.pop()
                    save_list.append("_segmented.")
                    save_list.append(format)
                    image_savename=''.join(save_list)
                real_save_path = os.path.join(directory,image_savename) 
            else: # if save_path!=None
                if image_savename == '': 
                    file_name = os.path.basename(image_path)
                    save_list=file_name.split(".")
                    format=save_list[-1] 
                    save_list.pop()
                    save_list.append("_segmented.")
                    save_list.append(format)
                    image_savename=''.join(save_list)
                real_save_path = os.path.join(save_path,image_savename) 

        if Plot == 'one':
            plt.imshow(edged_image) 
            plt.show()
            if save:
                image = Image.fromarray(edged_image)
                try:
                    image.save(real_save_path)
                    sg.popup(f"Your segmented image has been saved in the path: {real_save_path}",title="Information") 
                except: # In case of saving error regarding the path.
                    sg.popup("Saving failure. Please recheck your saving path.",title="Reminder") 
            if not save and (save_path!=None or image_savename!=""): 
                sg.popup("Please check the 'save' checkbox.",title="Reminder")
            
        elif Plot =='all': 
        # Intrepret the results
            fig, ax = plt.subplots(2, 4, figsize=(18, 8))
            ax[0,0].imshow(processed_image,cmap='gray')
            ax[0,0].set_title('Preprocessed Image')
            ax[0,1].imshow(segmented, cmap='gray')
            ax[0,1].set_title('Segmented Image')
            ax[0,2].imshow(binary_segmented, cmap='gray')
            ax[0,2].set_title('Binary Segmented Image')
            ax[0,3].imshow(closed_image, cmap='gray')
            ax[0,3].set_title('Image after closing')
            ax[1,0].imshow(removed_image,cmap='gray')
            ax[1,0].set_title('After removing small inner objects',fontsize=11)
            ax[1,1].imshow(updated_regions,cmap='gray')
            ax[1,1].set_title('After filtering regions')
            ax[1,2].imshow(second_removed_image,cmap='gray')
            ax[1,2].set_title('After removeing small innner objects again',fontsize=9)
            ax[1,3].imshow(edged_image)
            ax[1,3].set_title('Final edging')
            
            if save:              
                try:
                    fig.savefig(real_save_path)
                    sg.popup(f"Your segmented image has been saved in the path: {real_save_path}",title="Information") 

                except: #In case of saving error regarding the path.
                    sg.popup("Saving failure. Please recheck your saving path.",title="Reminder") 
  
            if not save and (save_path!=None or image_savename!=""):
                sg.popup("Please check the 'save' checkbox.",title="Reminder")
            plt.show()
            

    if if_save_path == '':
        decide_if_save_path = None
    else:
        decide_if_save_path = if_save_path


    if if_plot == False and if_plot_all == True:
        sg.popup('Please choose to plot image',title="Reminder")
    elif if_plot == False: 
        decide_if_plot_all = 'None' 
    elif if_plot == True:
        if if_plot_all == False:
            decide_if_plot_all = 'one'
        else:
            decide_if_plot_all = 'all'


    if event == "Exit" or event == sg.WIN_CLOSED:
        window.close()
        break
    elif event == "Classification":
        if values['input_image'] == "" or values["input_model"] == "":
            sg.popup("please choose files first",title="Reminder")
        else:
            if imghdr.what(image_path) is None:
                sg.popup("please choose a proper image",title="Reminder")
            else:
                img = cv2.imread(image_path)
                plt.imshow(img) #print the input image
                plt.title("This is your uploaded image.")
                plt.show() 
                transform = transforms.Compose([
                            transforms.Resize([224,224]), #resize the image to the specified size
                            transforms.ToTensor(), #convert the image to tensor
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #normalize the images 
                            ])
                model = torch.load(model_path, map_location='cpu') #load the trained model and save to cpu for general users no matter Mac or other devices
                model.eval()
                tumor_type = ["benign", "malignant"]
                image = torch.unsqueeze(transform(Image.open(image_path).convert("RGB")), dim=0)
                with torch.no_grad():
                    pred = torch.argmax(model(image),dim=-1).cpu().numpy()[0]

                word_result = f"This breast tumor histopathological slide is predicted to be: {tumor_type[pred]}"
                sg.popup(word_result,title="Breast tumor predictor")                
    elif event == "Segmentation":
        if values['input_image'] == "" or imghdr.what(image_path) is None:
            sg.popup("please choose a proper image first",title="Reminder")
        else:
            nuclei_segmentation(image_path,User_given,decide_if_plot_all,save=if_save,save_path=decide_if_save_path,Measure_cells=if_measure_cells,Measure_regions=if_measure_regions,Calculate_length=if_calculate_length,Measure_eccentricity=if_measure_eccentricity)


window.close() 
