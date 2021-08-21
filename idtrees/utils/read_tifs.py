import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_im_paths(data_mode):
    # Returns list of all image paths 
    # Input:
    # data_mode: [string] - requested data mode. Valid values: [RGB, HSI, LAS]
    # Output:
    # im_paths: [string] - List of paths of all images wrt. root directory (idtrees) including file ending
    # Init paths of all image files

    train_path = "data/train/RemoteSensing/" # Path of training data wrt. root directory
    data_modes = ["RGB", "HSI", "LAS", "CHM"]
    mode_paths = ["RGB/", "HSI/", "LAS/", "CHM/"]
    mode_filetypes = [".tif", ".tif", ".las", ".tif"]
    site_paths = ["MLBS_", "OSBS_"] # Name of training site. TODO: get from file
    num_ims_site = [46, 39] # Number of images in each training site. TODO: get from file

    # Check if requested mode is specified
    # TODO: this error message didnt work.
    try:
        mode_id = data_modes.index(data_mode)
    except ValueError:
        print ("Requested data mode not specified; please specify in utils.load_data.get_im_paths")

    # Create list of all image paths
    im_paths = []
    for site_path, num_ims in zip(site_paths, num_ims_site):
        for i in range(num_ims):
            im_paths.append(train_path + mode_paths[mode_id] + site_path + str(i+1) + mode_filetypes[mode_id])

    return im_paths

def plot_bounding_box_of_idx(idx):
    """
    # Plots one bounding box 
    # Input:
    # uid - int: Index of bounding box that should be plotted
    """
    im, target = image_dataset.dataset.__getitem__(idx)
    im = im.numpy()
    print('Shape of bounding box: ', im.shape)
    # TODO: do not average over bands and make case distinction between rgb and hsi
    plt.imshow(im.mean(axis=0))

def get_hsi_pixels():
    """
    # Returns flattened array with class id and spectrum of all hyperspectral pixels that belong to a labelled bounding box
    # Input:
    # data - ndarray(1 + n_hsi_bands, n_pixels): Array that contains class_id (at idx=0) and spectrum of each pixel
    # TODO: move this function into other file than read_tifs.py 
    """
    import torch
    from dataset import TreeImagesDataset

    # Load hsi dataset of bounding boxes with torch dataloader
    csv_file ='data/data_train_mitree.csv'
    root_dir = '' # TODO: find out why we don't need to specicy root dir
    image_dataset = TreeImagesDataset(csv_file, root_dir, object_rec=False, datatype='hsi', instance_type = 'boxes')
    image_dataset = torch.utils.data.Subset(image_dataset, [0]) # TODO: find out what the [0] does

    # Get number of bounding boxes in dataset
    n_boxes = image_dataset.dataset.__len__()
    print('Number of trees, labeled with species and bounding box: ', n_boxes)

    # Iterate over each bounding box and
    spectra = [] # List of spectrum per pixel # TODO write as ndarray
    class_ids = [] # List of target per pixel 
    for idx in range(n_boxes): 
        # Get image and target of current bounding box 
        im, target = image_dataset.dataset.__getitem__(idx)

        # Append the spectra and class id of all pixels in bbox to a list
        n_px = np.prod(im.shape[1:])
        spectra.append(im.reshape(-1, n_px))
        class_ids.append(target['labels'] * np.ones(n_px))

    # Convert list into ndarray
    spectra = np.concatenate(spectra, axis=1)#.numpy())
    class_ids = np.concatenate(class_ids, axis=0)

    # Add class ids as zero'th row 
    data = np.vstack((class_ids[np.newaxis,:], spectra))
    
    return data

# Display one RGB image
def read_tif_to_ndarray(im_path, verbose=False):
    # Return one image
    """
    # Input 
    # im_paths - [string]: List of paths
    # Output
    # im - ndarray(n_channels, height, width): RGB or HSI Image normalized to [0,1]
    """
    from tifffile import imread
    # Read in image.tif and convert to float
    im_unnormalized = imread(im_path).astype(float)

    # Normalize imagery to [0,1] float
    im = np.zeros(im_unnormalized.shape)
    
    if im_unnormalized.ndim == 2: # CHM
        # not sure if normalizing is helpful
        im_unnormalized = np.expand_dims(im_unnormalized, axis=2)
        im_min, im_max = 0, 1
    elif im_unnormalized.shape[2] == 3: # RGB
        im_min, im_max = 0., 255.
    else: # Normalize HSI imagery
        im_min, im_max = 0., np.max(im_unnormalized)
    if verbose: print('min, max :', np.min(im_unnormalized), np.max(im_unnormalized))
    im = im_unnormalized / (im_max - im_min)
    
    # Roll from (height, width, n_ch) to (n_ch, height, width)
    im = np.rollaxis(im, 2) 
    
    # (In case tifffile doesn't work anymore and we want to use cv2)
    #im_unnormalized =cv2.imread(im_path + '.tif', cv2.IMREAD_UNCHANGED)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # convert BGR to RGB; necessary if read in with cv2
    return im


def read_all_tif_to_ndarray(im_paths):
    """
    # Input: 
    # im_paths - list(string): List of paths to images
    # Ouput:
    # ims - ndarray(n_images, n_channels, height, width): Tensor of RGB or HSI images
    """
    n_ims = len(im_paths)
    n_channels, height, width = read_tif_to_ndarray(im_paths[0]).shape # Get dimensions from Test image
    ims = np.empty((n_ims, n_channels, height, width))
    # Convert images from .tif into .png
    for i, im_path in enumerate(im_paths):
        #print('im_path', im_path)
        ims[i] = read_tif_to_ndarray(im_path)
    return ims


def plot_all_ims(ims, title='rgb'):
    """
    # Plots some RGB images
    """
    n_cls = 4
    n_ims_p_cl = 5# im_paths[0])
    fig, axs = plt.subplots(n_ims_p_cl, n_cls, figsize = (10, 12), sharex=True, sharey=True)
    for i, im in enumerate(ims):
        axs[int(round(int(round(i)/n_cls))),i%n_cls].imshow(np.squeeze(np.rollaxis(im,0,start=3)), interpolation='nearest')
    for ax, col in zip(axs[0], range(n_cls)):
        ax.set_title(col)
        fig.tight_layout()
    title = title
    # TODO: print plot on top of image: plt.title('Collection of ' + title + ' images')
    plt.draw()
    plt.savefig('figures/collection_'+ title + '.png')
    

def plot_spectral_curve(ims):
    """
    # Input:
    # ims - ndarray(n_images, n_channels, height, width): Tensor of RGB images
    """
    print(ims.shape)
    n_channels = ims.shape[1]
    spec_mean = np.mean(ims, axis=(0, 2, 3)) # Average over all images and pixel
    
    x_label = 'RGB band' if n_channels == 3 else 'Hyperspectral band ID'
    x = ["R","G","B"] if n_channels == 3 else range(n_channels)
    title = '_rgb' if n_channels == 3 else '_hsi'
    plt.xlabel(x_label)
    plt.ylabel('Intensity')
    plt.plot(x, spec_mean, )
    plt.savefig('figures/spectral_curve' + title + '.png')
    
def plot_hyperspectral_curve_per_species(data, class_ids, scale_std = 1.):
    """
    # Plot average hyperspectral curve per tree species
    # Input:
    # data - ndarray(1 + n_hsi_bands, n_pixels): Array that contains class_id (at idx=0) and spectrum of each pixel
    # class_ids - ndarray(n_classes,): Unique ids of the classes that should be plotted
    # scale_std - float: Scales the plotted standard deviation 
    """
    # Get scientific names for curve labels
    # TODO: outsource into function that gets sci names
    df_sci_name = pd.read_csv('data/train/Field/taxonID_ScientificName.csv')

    fig, axs = plt.subplots(1, 1, figsize = (26, 19))
    for c in class_ids:
        # Get spectra of all trees that belong to the class
        trees_in_c = np.argwhere([data[0,:] == c])
        spectra_in_c = data[1:,trees_in_c[:,1]]

        class_spectrum = spectra_in_c.mean(axis=1)
        class_std = spectra_in_c.std(axis=1)

        label = str(int(c)) +": "+ df_sci_name[df_sci_name.taxonCode==c].scientificName.iloc[0]
        axs.plot(range(class_spectrum.shape[0]), class_spectrum, label=label)
        axs.fill_between(range(class_spectrum.shape[0]), 
                         class_spectrum - class_std * scale_std, 
                         class_spectrum + class_std * scale_std,
                     alpha=0.2)

    axs.set_ylabel('Reflectance')
    axs.set_xlabel('HSI band')
    plt.title('Hyperspectral reflectance per species')
    plt.legend(loc='upper right')
    
def plot_histogram_per_species(data, class_ids, n_bins = 20):
    """
    # Plot histogram of each tree species (class)
    # Input:     
    # data - ndarray(1 + n_hsi_bands, n_pixels): Array that contains class_id (at idx=0) and spectrum of each pixel
    # class_ids - ndarray(n_classes,): Unique class ids of the plotted tree species
    # n_bins - int: Number of discrete bins
    """

    # Get scientific names for curve labels
    df_sci_name = pd.read_csv('data/train/Field/taxonID_ScientificName.csv')

    fig, axs = plt.subplots(1, 1, figsize = (26, 19))

    # Plot histogram of each tree species (class)
    for c in class_ids:
        # Get spectra of all trees that belong to the class
        trees_in_c = np.argwhere([data[0,:] == c]) 
        spectra_in_c = data[1:,trees_in_c[:,1]] 

        label = str(int(c)) +": "+ df_sci_name[df_sci_name.taxonCode==c].scientificName.iloc[0]

        # Create histogram of data, i.e., count how many bands have a value inside bins of reflectance
        hist = np.histogram(spectra_in_c, bins=n_bins)

        # Normalize histogram over number of trees per class
        hist = np.array([hist[0], hist[1]]) # Copy tuple into array
        hist[0] = hist[0] / trees_in_c.shape[0] 

        # Plot histogram 
        axs.plot(hist[1][1:], hist[0], '-', label=label)

    plt.rcParams.update({'font.size': 16})
    # X-label: Number of bands of one pixel that contain a value within the reflectance bin
    plt.xlabel('Avg value of reflectance bins')
    # Y-label: Number of bands with a value inside a reflectance bin, averaged over all pixels of a given class
    plt.ylabel('Number of bands')#Class ID')
    plt.title('Histogram: Frequency of bands per class')
    plt.legend()
    plt.savefig('figures/histogram_over_cls.png')
