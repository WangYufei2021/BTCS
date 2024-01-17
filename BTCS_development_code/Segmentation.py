import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, feature, morphology,measure,img_as_float
from skimage.transform import resize
from sklearn.cluster import KMeans
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_yen
from PIL import Image
import os


def parameter_defining(User_given):
    """
    @brief:
        According to different user-given estimations, set different parameters for subsequent processing.
        How parameters defined is based on accumulations of experiences. 
    @args:
        User_given: "Few", "Median" or "Many"
    @returns:
        r: The parameter for the first binary closing
        r2: The parameter for the second binary closing
        k: How many clusters K-means need to classify
    """
    # Median situation setting
    r=1 
    r2=5 
    k=8
    if User_given=='few':
        r=1
        r2=3
        k=10
    elif User_given=="many":
        r=4
        r2=8
        k=6
    return r, r2, k

def preprocess_image(image):

    """
    @brief:
        Select specific channel of the image and preprocess it. 
    @args:
        image: The input image
    @returns:
        Proprocessed image.
    """
    
    # gray_image = color.rgb2gray(image)
    # After attemps, we find select the first channel is good for nuclei segmentation
    gray_image = img_as_float(image[:,:,0])

    # Denoise the image
    sigma_est = np.mean(estimate_sigma(gray_image))
    denoised_image = denoise_nl_means(gray_image, h=1.15 * sigma_est, fast_mode=True,
                                      patch_size=5, patch_distance=3)
    
    # Enhance contrast using CLAHE
    clahe = exposure.equalize_adapthist(denoised_image)
    
    # Resize the image in case that it is too large
    if gray_image.shape[0] > 500:
        clahe = resize(clahe, (500, int(clahe.shape[1] * 500 / clahe.shape[0])), anti_aliasing=True)
    
    return clahe

def segment_image(image, k):
    """
    @brief:
        Segment image using K-means
    @args:
        image: The input image
        k: The number of clusters
    @returns:
        Segmented image.
    """

    flat_image = image.flatten().reshape(-1, 1)
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0).fit(flat_image)
    labels = kmeans.labels_
    
    segmented_image = labels.reshape(image.shape)
    
    return segmented_image


def labeling_picture(processed_image,segmented):
    """
    @brief:
        Make segmented pictures into binary labeling
    @args:
        processed_image: The image that has only undergone preprocessing
        segmented: The image that has undergone preprocessing and K-means segmentation
    @returns:
        Binary segmented labeled image.
    """

    # Sort pixel to know the darkest pixel index (from the smallest to biggest)
    darkest_pixel_index=processed_image.flatten().argsort() # 从小到大 前x个 即为最小的x个 即最黑的x个
    darkest_label_list=set()
    # Find the first two darkest labels' name. The reason why to choose 2 is based on attempts on different images.
    for k in darkest_pixel_index:
        darkest_label_list.add(segmented.flatten()[k])
        if len(darkest_label_list)==2:
            break

    # darkest_label_list=list(darkest_label_list)
    # darkest_label_list=list(set(segmented.flatten()[darkest_pixel_index]))
    # Initialize a new image with the same shape as the segmented image, filled with black

    # Create a new one with all 0s
    binary_segmented = np.zeros_like(segmented)

    for darkest_label in darkest_label_list:
        label_0_mask = segmented == darkest_label
        # Set the pixels corresponding to target labels to 1
        binary_segmented[label_0_mask] = 1

    return binary_segmented


def filter_regions(label_image,user_given):

    """
    @brief:
        Filter smallest regions(actually not nucleus) and irregular shapes to remain normal nucleus.
    @args:
        label_image: The image that has only undergone preprocessing
        user_given: The image that has undergone preprocessing and K-means segmentation
    @returns:
        Binary segmented labeled image.
    """

    # Label the image 会得到一个圈为同一个数 下一个圈圈为另一个数...
    # label_image = measure.label(label_image)

    # Calculate properties of the labeled regions
    region_props = measure.regionprops(label_image)

    # Get the area of each region
    areas = [region.area for region in region_props]

    # This will give a relatively large value according to 
    thre_otsu=threshold_otsu(np.array(areas))
    # This will give a small value 
    thre_yen=threshold_yen(np.array(areas))

    # This is a adjustions from different evaluations
    thresh_new=np.mean(np.array([thre_yen,thre_otsu,np.mean(areas),np.median(areas)])) 


    if user_given=="few":
        thresh=thre_otsu
    elif user_given=="median":
        thresh=thresh_new
    else:
        # thresh=np.mean(np.array([thre_yen,thre_yen,thre_yen,thre_otsu])) # ,np.mean(areas),np.median(areas)
        thresh=thre_yen


    # Create a mask of the same shape as the label image, initialized to False
    mask = np.zeros_like(label_image, dtype=bool)

    # Iterate over each region to keep only those with area >= threshold and ratio < 6
    Count=0
    ratios=[]
    for region in region_props:
        try:
            # Get ratio to delete irregular shapes
            ratio=region.major_axis_length/region.minor_axis_length
        except ZeroDivisionError: 
            ratio=float("inf")
        ratios.append(ratio)
        if region.area >= thresh and ratio < 6:
            # Set all locations of the exsiting regions to True in the mask 
            mask[label_image == region.label] = True
            Count+=1

    # Apply the mask to the label_image to remove small objects
    label_image[~mask] = 0

    # Convert the label_image back to binary
    final_binary_image = label_image > 0

    return final_binary_image


def remove_inner_objects(image):

    """
    @brief:
        Some nucleus have inner objects not detected (looks like concentric circles), so we want to fill them so that they will be segmented as a whole.
    @args:
        image: The input image
    @returns:
        Image that has been removed inner objects.
    """
    # The image is inverted for further similar remove inner 
    inverted_image=np.logical_not(image)
    label_image = measure.label(inverted_image)
    # Calculate properties of the labeled regions
    region_props = measure.regionprops(label_image)

    # Get the area of each region
    areas = [region.area for region in region_props]

    # The largest one is the original background (but it can be somewhat segmented), so we can just set a otsu threshold to classify.
    thre_otsu=threshold_otsu(np.array(areas))


    mask = np.zeros_like(label_image, dtype=bool)

    # Iterate over each region to keep only those with area < thre_otsu
    for region in region_props:
        if region.area < thre_otsu:
            # Set all locations of the current region to True in the mask 该圈对应的label相应的位置全部都留下即设置为True
            mask[label_image == region.label] = True

    # Apply the mask to the final image to remove small objects
    final_image=image.copy()
    final_image[mask]=1

    
    return final_image

def Footprint_size(closed_image,level):

    """
    @brief:
        According to User given input estimation, we define a specific footprint size for subsequent instance segmentation.
    @args:
        closed_image: The input image
        level: User given input estimation ("median" and "many" except "few")
    @returns:
        Footprint size for instance segmentation.
    """

    label_image = measure.label(closed_image)
    region_props = measure.regionprops(label_image)

    # Obtain major axis length and minor axis length for each nuclei, and get the average. 
    radius = [(region.major_axis_length+region.minor_axis_length)/2 for region in region_props] 
    radius.sort()

    # if level=='few': 
    #     # Select Top 5 biggest one as a reference for footprint size (with attemps).
    #     size=radius[-5] 
    #     # The size need to be a integer. 
    #     return round(size)
    if level=='median':
        # Since there are overall 10~50 nucleus, we select Top25 as a reference for footprint size (with attemps).
        limit=-min(25,len(radius))
        size=radius[limit] 
        return round(size)
    else:
        # Since there are around > 50 nucleus, we select Top75 (if no 75 nucleus, set the smallest one) for footprint size.
        limit=-min(75,len(radius))
        size=radius[limit] 
        return round(size)



def instance_segmentation(image,User_given):
    """
    @brief:
        Many nucleus overlap with each other, it is very difficult to distinuguish them with semantic segmentation.
        In that case, instance segmentation (watershed algorithm) can help a little to relabel regions. 
    @args:
        image: The input image
        User_given: User given input estimation
    @returns:
        Labels after instance segmentation using watershed segmentation
    """
    # If nuclus are very few, the instance segmentation will not be processed, because we assume the nucleus don't overlap a lot.
    if User_given=='few': 
        # Directly return the normal labels
        return measure.label(image)
    
    # Given user_given, set different footprint size
    footprint_size=Footprint_size(image,User_given)
    distance = ndi.distance_transform_edt(image)

    # The minmum footprint size is (7,7)
    Size=max(7,footprint_size) 
    coords = peak_local_max(distance, footprint=np.ones((Size, Size)), labels=image)

    # Set mask
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)
    return labels

def get_edge(image, second_removed_image):
    """
    @brief:
        Label edges.
    @args:
        image: The origianl image without any processing
        second_removed_image: The image with all processings
    @returns:
        Image labeled edges
    """
    edges = feature.canny(second_removed_image)

    # 2x2 square to make the edge rough.
    selem = morphology.square(2) 
    dilated_edges = morphology.dilation(edges, selem)

    edged_image=image.copy()
    edged_image[dilated_edges, 0] = 0  
    # The second channel can make the edge more clearly in the image.
    edged_image[dilated_edges, 1] = 1 
    edged_image[dilated_edges, 2] = 0 
    return edged_image


def nuclei_property(image,Measure_cells=False,Measure_regions=False,Calculate_length=False,Measure_eccentricity=False):
    """
    @brief:
        According to users, we provide simple measurements of nuclei property.
    @args:
        image: The segmented image without being edged.
        Measure_cells: Measure cell numbers, True or False
        Measure_regions: Measure region areas (mean/median value) in pixels, True or False
        Calculate_length: Calculate length of major/minor axis of the ellipse nuclei (mean/median value), True or False
        Measure_eccentricity: The eccentricity of the nuclei (mean/median value), True or False
    @returns:
        Nothing, directly print results.
    """
    label_image = measure.label(image)
    region_props = measure.regionprops(label_image)

    
    areas = np.array([region.area for region in region_props])
    if Measure_cells:
        print(f"The number of cells (nuclei): {len(areas)}")
    if Measure_regions:
        print(f"The mean area of nuclei is: {round(np.mean(areas),2)}")
        print(f"The median area of nuclei is: {round(np.median(areas),2)}")

    if Calculate_length:
        majors=[region.major_axis_length for region in region_props]
        minors=[region.minor_axis_length for region in region_props]
        print(f"The mean length of the major axis of the ellipse nuclei is: {round(np.mean(majors),2)}; The median is: {round(np.median(majors),2)}")
        print(f"The mean length of the minor axis of the ellipse nuclei is: {round(np.mean(minors),2)}; The median is: {round(np.median(minors),2)}")

    if Measure_eccentricity:
        Eccentricities = np.array([region.eccentricity for region in region_props])
        print(f"The mean eccentricity of nuclei is: {round(np.mean(Eccentricities),2)}")
        print(f"The median eccentricity of nuclei is: {round(np.median(Eccentricities),2)}")




def nuclei_segmentation(image_path,User_given,Plot='one',save=False, save_path=None,Measure_cells=False,Measure_regions=False,Calculate_length=False,Measure_eccentricity=False):
    """
    @brief:
        Nuclei segmentation undergoes preprocessing, segmentation, binary labeling, closing, removing inner objects, instance segmentation, filter regions, and getting edges. 
        Although it is showed in this Segmentation.py file, we provide GUI at the last version of our software and this function is packged into the GUI Python file with Plot functions modified.
    @args:
        image_path: The image path.
        User_given: User given input estimation. "Few": <10 cells; "median": 10~50 cells; "many": > 50 cells
        Plot: Do you want to plot only the final edged image ("one"), or the whole processing steps ("all"). 
    @returns:
        Nothing, directly plot or print results.
    """
    r, r2, k = parameter_defining(User_given)

    image = io.imread(image_path)

    processed_image = preprocess_image(image)

    segmented = segment_image(processed_image, k=k)

    binary_segmented=labeling_picture(processed_image,segmented)

    closed_image = morphology.binary_closing(binary_segmented, footprint=morphology.disk(r))

    removed_image=remove_inner_objects(closed_image)

    # if User given == 'few', this step will not be processed
    labels=instance_segmentation(removed_image,User_given) 

    updated_regions=filter_regions(labels,User_given)

    Last_image = morphology.binary_closing(updated_regions, footprint=morphology.disk(r2))

    second_removed_image=remove_inner_objects(Last_image)

    edged_image=get_edge(image, second_removed_image)

    nuclei_property(second_removed_image,Measure_cells=True,Measure_regions=True,Calculate_length=True)

    if save:
        if save_path==None:
        # The file will be stored in where the original image stays with "_segmented" at the end.
            if image_savename == '':
                save_list=image_path.split(".")
                format=save_list[-1]
                save_list.pop()
                save_list.append("_segmented.")
                save_list.append(format)
                image_savename=''.join(save_list)
                image = Image.fromarray(edged_image)
                real_save_path = os.path.join(image_savename)
                image.save(real_save_path)



    if Plot == 'one':
        plt.imshow(edged_image) 
        if save:
            if save_path!=None:
                if image_savename == '':
                    print("Fail to save! Please enter the name of the saved file.")
                else:
                    image = Image.fromarray(edged_image)
                    real_save_path = os.path.join(save_path,image_savename)
                    image.save(real_save_path)
        plt.show()
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
            if save_path!=None:
                if image_savename == '':
                    print("Fail to save! Please enter the name of the saved file.")
                else:
                    image = Image.fromarray(edged_image)
                    real_save_path = os.path.join(save_path,image_savename)
                    image.save(real_save_path)
        plt.show()

