import sys
sys.path.append('./')

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image
import io
import skimage.io as skio
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage import dtype_limits
from scipy import misc
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # <- deactivate image bomb error
import timeit

try:
    import cv2
    opencv_imported = True
except ImportError:
    print("Could not import opencv, using skimage instead")
    opencv_imported = False

# For loadng volumetric data
import napari
import SimpleITK as sitk
import nrrd
from BrainRender.Utils.image import image_to_surface


from BrainRender.Utils.paths_manager import Paths
from BrainRender.Utils.data_io import connected_to_internet, strip_path, listdir
from BrainRender.Utils.webqueries import request

"""
    The GeneExpressionAPI takes care of interacting with the image download API from the Allen ISH experiments
    do download images with gene expression data. Then, it extracts cell locations from these images and 
    alignes the found cells to the Allen CCF. The code for the cell location extractraction and alignment is 
    based on this GitHub repository: https://github.com/efferencecopy/ecallen. 
    More details on the ecallen code can be found here: https://efferencecopy.net/allen-brain-fast-image-registration/

"""


class GeneExpressionAPI(Paths):
    # URLs based on examples here: http://help.brain-map.org/display/api/Downloading+an+Image
    all_mouse_genes_url = ("http://api.brain-map.org/api/v2/data/Gene/query.json?criteria=products%5Bid$eq1%5D")

    gene_search_url =(
        "http://api.brain-map.org/api/v2/data/query.json?criteria="
        "model::SectionDataSet,"
        "rma::criteria,[failed$eq'false'],products[abbreviation$eq'Mouse'],"
        "plane_of_section[name$eq'PLANE'],genes[acronym$eq'GENE']"
    )

    exp_images_url = (
        "http://api.brain-map.org/api/v2/data/query.json?criteria="
        "model::SectionImage,"
        "rma::criteria,[data_set_id$eqEXPID]"
        )

    image_download_url = "http://api.brain-map.org/api/v2/image_download/IMAGEID"
    expression_image_download_url = "http://api.brain-map.org/api/v2/image_download/IMAGEID?view=expression"
    projection_image_download_url = "http://api.brain-map.org/api/v2/projection_image_download/IMAGEID?downsample=0&view=projection"

    gridded_expression_url = "http://api.brain-map.org/grid_data/download/EXPID?include=energy,intensity"

    # URLs based on https://github.com/efferencecopy/ecallen/blob/master/ecallen/images.py
    experiment_metadata_url = ('http://api.brain-map.org/api/v2/data/query.json?criteria='
                        'model::SectionDataSet'
                        ',rma::criteria,[id$eqEXPID]'
                        ',rma::include,genes,plane_of_section,probes'
                        ',products,reference_space,specimen,treatments'
                        )

    affine_3d_url = ('http://api.brain-map.org/api/v2/data/query.json?criteria='
                        'model::SectionDataSet'
                        ',rma::criteria,[id$eqEXPID],rma::include,alignment3d'
                    )

    affine_2d_url =  ('http://api.brain-map.org/api/v2/data/query.json?criteria='
                        'model::SectionImage'
                        ',rma::criteria,[id$eqIMGID],rma::include,alignment2d')


    def __init__(self, debug=False, use_opencv=True):
        # Check if we can use opencv
        if use_opencv and not opencv_imported:
            use_opencv = False 
        self.use_opencv = use_opencv 

        Paths.__init__(self)
        self.debug = debug

        self.test_dataset_id = 167643437

        self.nrrd_fld = os.path.join(self.gene_expression, "nrrd")
        if not os.path.isdir(self.nrrd_fld):
            os.mkdir(self.nrrd_fld)

        self.root_bounds = ([-17, 13193],  # used to reshape gene expression grid arrays
                            [134, 7564], 
                            [486, 10891])

        self.HCF_genes = [73520980, 70927813, 71924374, 77925097] 
        # list of experiments for genes highly expressed in HCF (hippocampus)
        # just coronal sections, for testing. 

    """
        ################## DATA IO ########################
    """
    def get_all_available_genes(self):
        url = self.all_mouse_genes_url
        res = pd.DataFrame(request(url, return_json=True)['msg'])

    def download_gridded_expression_data(self, expid):
        exp_dir = os.path.join(self.gene_expression, str(expid))
        if not os.path.isdir(exp_dir): os.mkdir(exp_dir)
        url = self.gridded_expression_url.replace("EXPID", str(expid))

        # res = request(url, return_json=False)
        print("\nThis is not automated yet, but please download the file going to the url: ")
        print(url)
        print("\nThen move the files to: {}\n\n".format(exp_dir))


    @staticmethod
    def imgid_from_imgpath(imgpath, image_type="expression"):
        if image_type is None or not image_type:
            img_id = int(strip_path(imgpath)[-1].split(".")[0])
        else:
            img_id = int(strip_path(imgpath)[-1].split("_")[0])
        return img_id

    def search_experiments_ids(self, plane, gene, fetch_images=False):
        """
            [Given a plane of section and a gene name, checks for ISH experiments that match the criteria. 
            Optionally, it takes care of downloading the corresponding images]

            Arguments:
                plane {[str]} -- ['sagittal' or 'coronal']
                gene {[str]} -- [name of the gene, you can lookup genes here: "http://mouse.brain-map.org"]

            Keyword arguments:
                fetch_images {[bool]} -- [If true the imges for the experiments found are downloaded]
        """
        url = self.gene_search_url.replace("PLANE", plane).replace("GENE", gene)
        res = pd.DataFrame(request(url, return_json=True)['msg'])

        if res.empty:
            print("Could not find any experiments that match the search criteria: {} - {}".format(plane, gene))
            return False

        if not fetch_images:
            return res.id.values
        else:
            for expid in res.id.values:
                self.fetch_images_for_exp(expid)

    def fetch_images_metadata(self, expid):
        """
            [Given an experiment id number, it downloads the image metadata for it.]
        """
        url = self.exp_images_url.replace("EXPID", str(expid))
        res = pd.DataFrame(request(url, return_json=True)['msg'])
        if res.empty:
            print("Could not find metadata for experiment id: {}".format(expid))
        return res

    def get_imaging_params(self, expid):
        url = self.experiment_metadata_url.replace("EXPID", str(expid))
        metadata = request(url, return_json=True)['msg'][0]

        if not metadata:
            print("Could not fetch imaging params for exp: {}".format(expid))
            return None

        # define the image params
        img_params = {
            'plane_of_section': metadata['plane_of_section']['name'],
            'red_channel': metadata['red_channel'],
            'green_channel': metadata['green_channel'],
            'blue_channel': metadata['blue_channel'],
            'is_FISH': metadata['treatments'][0]['name'].lower() == 'fish',
            'is_ISH': metadata['treatments'][0]['name'].lower() == 'ish',
            'probes': [dct['acronym'].lower() for dct in metadata['genes']],
            'section_thickness': metadata['section_thickness'],
            'genotype': metadata['specimen']['name']
        }
        return img_params

    def fetch_images_for_exp(self, expid, **kwargs):
        """
            [Given an experiment id number, it downloads the image data for it.]
        """
        res = self.fetch_images_metadata(expid)
        self.downlad_images(expid, res.id.values, **kwargs)

    def downlad_images(self, expid, image_ids, dest_folder=None, image_type=None):
        """
            [Downloads images as binary data and saves as .png. It's a bit slow.]
        """
        if dest_folder is None:
            dest_folder = self.gene_expression
        if not isinstance(image_ids, (list, np.ndarray)): image_ids = [image_ids]

        dest_folder = os.path.join(dest_folder, str(expid))
        if not os.path.isdir(dest_folder):
            try:
                os.mkdir(dest_folder)
            except:
                raise FileExistsError("Could not create target directory: {}".format(dest_folder))

        print("\nDownloading {} images for experiment: {}".format(len(image_ids), expid))
        for iid in tqdm(image_ids):
            if image_type is None or not image_type:
                url = self.image_download_url.replace("IMAGEID", str(iid))
                name_ext = ""
            elif image_type =="expression":
                url = self.expression_image_download_url.replace("IMAGEID", str(iid))
                name_ext = "_expression"
            elif image_type == "projection":
                url = self.projection_image_download_url.replace("IMAGEID", str(iid))
                name_ext = "_projection"

            savefile = os.path.join(dest_folder, str(iid)+name_ext+".png")
            if os.path.isfile(savefile): continue

            res = request(url)
            # save binary data to an image
            with open(savefile, "wb") as f:
                f.write(res.content)

    def get_affine_3d(self, expid) -> dict:
        """
        Get the coefficients for 3D affine transformation.
        get_affine_3d(expid)
        Query the Allen API to obtain the values for the TVR transform. This
        converts section_image 'volume' coordinates to 'reference' coordinates.
        Args:
            expid: int
                    Scalar that identifies the data set
        Returns:
            affine3: dict
                    With the following keys:
                    'A_mtx': 3x3 matrix of affine rotation coefficients
                    'traslation': 1x3 vector of translation coefficients
                    'section_thickness':  brain slice thickness in um (float)
        """
        url = self.affine_3d_url.replace("EXPID", str(expid))
        res = request(url, return_json=True)['msg'][0]
        align_info = res['alignment3d']

        coeffs = [align_info['tvr_{:0>2}'.format(x)] for x in range(12)]

        # construct the output dictionary
        affine3 = {'A_mtx': np.array(coeffs[0:9]).reshape((3, 3)),
                'translation': np.array(coeffs[9:]).reshape((3, 1)),
                'section_thickness': res['section_thickness']}

        return affine3

    def get_affine_2d(self, image_id) -> dict:
        """
        Get the coefficients for 2D affine transformation.
        get_affine_2d(section_data_set_id)
        Query the Allen API to obtain the values for the TVR transform. This
        converts section_image 'volume' coordinates to 'reference' coordinates.
        Args:
            image_id: (int) Scalar that identifies the data set
        Returns:
            affine2: dict
                    With the following keys:
                    'A_mtx': 2x2 matrix of affine rotation coefficients
                    'traslation': 1x2 vector of translation coefficients
                    'section_number': (int) used to determine distance between
                                    different slices
        """
        # send the query, extract the alignment information
        url = self.affine_2d_url.replace("IMGID", str(image_id))
        res = request(url, return_json=True)['msg'][0]
        align_info = res['alignment2d']
        coeffs = [align_info['tsv_{:0>2}'.format(x)] for x in range(6)]

        # construct the output dictionary
        affine2 = {'A_mtx': np.array(coeffs[0:4]).reshape((2, 2)),
                'translation': np.array(coeffs[4:]).reshape((2, 1)),
                'section_number': res['section_number']}

        return affine2

    """
        ########## IMAGE PROCESSING ############
    """
    # most of these functions are based on code from this repository: https://github.com/efferencecopy/ecallen/blob/master/ecallen/images.py
    def extract_region_props(self, img_path, params, probes, ish_minval=70) ->list:
        """Segment neuron cell bodies via thresholding.
        Accepts images from the Allen Brain Institute (ISH or FISH) and segments
        fluorescently labeled neuron cell bodies. Segmentation is accomplished by
        computing a label matrix on the thresholded image (via Otsu's method).

        Args:
            img_path (str): full path to the image.
            params (dict):  The experiment's parameters (see .get_imaging_params).
            probes (list): list of strings, specifying the RNA target of the
                ISH or FISH stain
            ish_minval (int): applies to ISH images only. Any value below
                this will be ignored by the thresholding algorithm.
                Default value is 70.
        Returns:
            rprops (list): each element is a dictionary of region properties
                as defined by scikit-image's regionprops function
        """

        # user must specify probe(s) (i.e., color channels) to analyze
        # if only one probe is specified, turn it into a list
        if type(probes) != list and type(probes) == str:
            probes = [probes]

        channels = ["red_channel", "green_channel", "blue_channel"]
        not_none_channels = [ch for ch in channels if params[ch] is not None]
        probe_ch = [params[ch].lower() in probes if ch in not_none_channels else False 
                        for ch in channels]

        # open the image
        # img = skio.imread(img_path)
        img = misc.imread(img_path)

        if params['is_FISH']:
            n_ch_correct = sum(probe_ch) > 0 and sum(probe_ch) <= 3
            assert n_ch_correct, "Did not identify the correct number of channels"
            img = np.array(img[:, :, probe_ch]).max(axis=2)  # max project

            # measure threshold
            thresh = threshold_otsu(img, nbins=256)

        elif params['is_ISH']:
            img = dtype_limits(img)[1] - img  # invert
            assert sum(probe_ch) == 3, "Not all ISH color channels identical"
            img = np.max(img, axis=2)  # max project inverted image

            # measure threshold
            thresh = threshold_otsu(img[img > ish_minval], nbins=256)
        else:
            raise ValueError('Image is neither FISH nor ISH')

        # apply the threshold to the image, which is now just a 2D matrix
        bw = img > thresh

        # label image regions with an integer. Each region gets a unique integer
        label_image = label(bw)
        rprops = regionprops(label_image)
        return rprops

    def get_cells_for_experiment(self, expid, ish_minval=85, image_type="expression", threshold=70, overwrite=False, **kwargs):
        # download the images for the experiment:
        self.fetch_images_for_exp(expid, image_type=image_type)

        # get the path to the images
        if image_type is not None and image_type:
            exp_images = [f for f in listdir(os.path.join(self.gene_expression, str(expid))) 
                if image_type in os.path.split(f)[-1]  and ".pkl" not in f]
        else:
            exp_images = [f for f in listdir(os.path.join(self.gene_expression, str(expid)))
                        if "expression" not in os.path.split(f)[-1] and "projection" not in os.path.split(f)[-1] and ".pkl" not in f]

        # exp folder
        exp_folder = os.path.join(self.gene_expression, str(expid))

        # Get cells aligned to ccf
        print("Extracting cells for experiment: {}".format(expid))
        data_files, cells_count = [], 0
        for img in tqdm(exp_images):
            if not ".png" in img: continue

            # Get image name and check if anlyzed already
            img_id = self.imgid_from_imgpath(img, image_type=image_type)

            img_data_file = os.path.join(exp_folder, str(img_id)+".pkl")
            if os.path.isfile(img_data_file) and not overwrite:
                data_files.append(img_data_file)
                cells_count += len(pd.read_pickle(img_data_file))
            else:
                cells = {"x":[], "y":[], "z":[]}
                if image_type is None and not image_type:
                    # Use the image processing functions from ecallen
                    img_cells = self.find_cells_in_image(img, expid, ish_minval=ish_minval)
                else:
                    # just get bright pixels
                    start = timeit.default_timer()
                    img_cells = self.analyze_expression_image(img, expid, image_type=image_type, threshold=threshold, **kwargs)
                    end = timeit.default_timer()
                    if self.debug:
                        print("     Extracting cells from image took: {}".format(end-start))
                
                cells['x'].extend(img_cells[0])
                cells['y'].extend(img_cells[1])
                cells['z'].extend(img_cells[2])
                cells_count += len(img_cells[0])

                cells = pd.DataFrame(cells)
                cells.to_pickle(img_data_file)
                data_files.append(img_data_file)
        print("Extracted {} cells\n\n".format(cells_count))
        return data_files
    
    def load_cells(self, expid=None, exp_data_path=None, image_type="expression", count_cells=True):
        """
            [Load the .pkl files with the cell location for each image in the dataset for experiment with id expid
            (or in folder exp_data_path).]
        """
        if exp_data_path is not None:
            if image_type is None:
                files = {os.path.split(f)[-1].split(".")[0]: pd.read_pickle(f) 
                            for f in listdir(exp_data_path) if ".pkl" in f}
            else:
                files = {os.path.split(f)[-1].split(".")[0]: pd.read_pickle(f) 
                            for f in listdir(exp_data_path) if ".pkl" in f and image_type in f}
        else:
            fld_path =  os.path.join(self.gene_expression, str(expid))
            if image_type is None:
                files = {os.path.split(f)[-1].split(".")[0]: pd.read_pickle(f)
                            for f in listdir(fld_path) if ".pkl" in f}
            else:
                files = {os.path.split(f)[-1].split(".")[0]: pd.read_pickle(f)
                            for f in listdir(fld_path) if ".pkl" in f and image_type in f}
        
        if count_cells:
            tot = 0
            for f, cells in files.items():
                tot += len(cells)
            print("Loaded {} cells across {} slices".format(tot, len(files.keys())))
        return files
    
    def analyze_expression_image(self, img_path, expid, threshold=60, image_type=None):
        start = timeit.default_timer()
        img = misc.imread(img_path)
        end = timeit.default_timer()
        if self.debug:
            print("     Loading image took: {}".format(end-start))

        if not self.use_opencv: # use skimage instead
            # threshold the image
            # thresh = threshold_otsu(img[img > threshold], nbins=256) #
            bw = img > threshold

            # label image regions with an integer. Each region gets a unique integer
            label_image = label(bw)
            rprops = regionprops(label_image)

            # grab the X and Y pixels at the center of each labeled region
            cell_x_pix = np.array([roi['centroid'][1] for roi in rprops])
            cell_y_pix = np.array([roi['centroid'][0] for roi in rprops])
        else:
            # Convert to gray and threshold
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

            # imgray = img[img > threshold]
            # imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # use adaptive thresholding:
            # th3 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

            # Extract centroids location from contours
            _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            centroids = [cv2.minEnclosingCircle(cnt) for cnt in contours]
            cell_x_pix = [x for (x,y),r in centroids]
            cell_y_pix = [y for (x,y),r in centroids]



        end = timeit.default_timer()
        if self.debug:
            print("     Image thresholding took: {}".format(end-start))


        # Align the cells coordinates to the Alle CCF v3
        start = timeit.default_timer()
        img_id = self.imgid_from_imgpath(img_path, image_type=image_type)
        aligned = self.align_cell_coords_to_ccf(cell_x_pix,
                      cell_y_pix, expid, img_id)
        end = timeit.default_timer()
        if self.debug:
            print("     Aligning {} cells took {}".format(len(cell_x_pix), end-start))

        return aligned

    def find_cells_in_image(self, img_path, expid, ish_minval=70):
        """
            [Processes an image to extract cell locations and align these coordinates to the CCF]
        """

        # grab the RNA probes from the imaging parameters
        img_params = self.get_imaging_params(expid)

        # run the segmentation function
        rprops = self.extract_region_props(img_path,
                                            img_params,
                                            img_params['probes'],
                                            ish_minval=ish_minval,
                                            )

        # grab the X and Y pixels at the center of each labeled region
        cell_x_pix = np.array([roi['centroid'][1] for roi in rprops])
        cell_y_pix = np.array([roi['centroid'][0] for roi in rprops])

        # Align the cells coordinates to the Alle CCF v3
        img_id = self.imgid_from_imgpath(img_path, image_type=image_type)
        aligned = self.align_cell_coords_to_ccf(cell_x_pix,
                      cell_y_pix, expid, img_id)
        return aligned

    def align_cell_coords_to_ccf(self, x_pix, y_pix, section_data_set_id, section_image_id) -> np.array:
        """
        Convert from [X,Y] to [P,I,R] coordinates.
        xy_to_pir(x_pix, y_pix, section_data_set_id, section_image_id)
        Implement 2D and 3D affine transformations to convert from
        "image" coordinates to section coordinates, and then to the
        Common Coordinate Framework units.
        Args:
            x_pix: numpy.array
                    vector of X coordinates of pixels in a section_image
            y_pix: numpy.array
                    vector of Y coordinates of pixels in a section_image
            section_data_set_id: int
            section_image_id: int
        Returns:
            pir: numpy.array
                [Posterior, Inferior, Right] of the corresponding location in the
                common coordinate framework (CCF) in units of micrometers
        """
        # implement the 2D affine transform for image_to_section coordinates
        start = timeit.default_timer()
        t_2d = self.get_affine_2d(section_image_id)
        tmtx_tsv = np.hstack((t_2d['A_mtx'], t_2d['translation']))
        tmtx_tsv = np.vstack((tmtx_tsv, [0, 0, 1]))  # T matrix for 2D affine
        data_mtx = np.vstack((x_pix, y_pix, np.ones_like(x_pix)))  # [3 x Npix]
        xy_2d_align = np.dot(tmtx_tsv, data_mtx)
        end = timeit.default_timer()
        if self.debug:
            print("         2D affine took: {}".format(end-start))
        

        # implement the 3D affine transform for section_to_CCF coordinates
        start = timeit.default_timer()
        t_3d = self.get_affine_3d(section_data_set_id)
        tmtx_tvr = np.hstack((t_3d['A_mtx'], t_3d['translation']))
        tmtx_tvr = np.vstack((tmtx_tvr, [0, 0, 0, 1]))


        data_mtx = np.vstack((xy_2d_align[0, :], xy_2d_align[1, :],
                            np.ones((1, xy_2d_align.shape[1])) * t_2d['section_number'] * t_3d['section_thickness'],
                            np.ones((1, xy_2d_align.shape[1]))))

        xyz_3d_align = np.dot(tmtx_tvr, data_mtx)
        pir = xyz_3d_align[0:3, :]
        end = timeit.default_timer()
        if self.debug:  
            print("         3D affine took: {}".format(end-start))
        return pir


    """
        ########## GRID VOLUME FUNCTIONS ############
    """

    def load_raw_grid_data(self, volume_file, expid=None, metadata_file=None, threshold=None, # threshold is in percentile
                                gridsize=25, visualize=False, save_to_obj=True, **kwargs):
        if metadata_file is not None:
            if not ".xml" in metadata_file: 
                raise ValueError("Unrecognized file format for metadata file.")
            pass
        else:
            # Load either a .raw or .mhd file and turn into a numpy array
            if ".raw" in volume_file:
                if gridsize == 25:
                    shape = (528, 320, -1) # 456
                elif gridsize == 200:
                    shape = (67, 41, -1) # 58
                else:
                    raise ValueError("Unrecognized gridsize value")

                # Load data to a numpy array
                f = open(volume_file, "rb")
                data = np.fromfile(f,dtype=np.uint8,count=shape[0]*shape[1]*shape[2])
                f.close()
                # data = np.frombuffer(data, dtype=np.uint8)
                data = data.reshape(*shape)
            elif ".mhd" in volume_file:
                data = sitk.ReadImage(volume_file)
                data = sitk.GetArrayFromImage(data)
            else:
                raise ValueError("Unrecognized file format for volume file")

        # Get N percentile and threshold
        if threshold is not None:
            th = np.percentile(data, threshold)
            data[data <= th] = 0
        else:
            th = 0

        # visualize the array interactively in napari
        if visualize:
            with napari.gui_qt():
                viewer = napari.view_image(data, rgb=False)

        # save 
        if save_to_obj:
            fname = os.path.split(volume_file)[-1].split(".")[0]
            if expid is None:
                expid = ""
            obj_path = os.path.join(self.nrrd_fld, fname+"_"+str(expid)+".obj")
            nrrd_path = os.path.join(self.nrrd_fld, fname+"_"+str(expid)+".nrrd")
            nrrd.write(nrrd_path, data)

            image_to_surface(nrrd_path, obj_path, voxel_size=float(gridsize), threshold=th, 
                    **kwargs)

            return obj_path

    def get_gene_expression_to_obj(self, expid, display="energy", **kwargs):
        exp_dir = os.path.join(self.gene_expression, str(expid))
        if not os.path.isdir(exp_dir): return

        relevant_files = [f for f in listdir(exp_dir) if ".mhd" in f or ".raw" in f or ".xml" in f]
        if not relevant_files:
            self.download_gridded_expression_data(expid)
            return

        volume_file = [f for f in relevant_files if display in f]
        if not volume_file:
            raise ValueError("Could not find a file to display: {}".format(display))
        if len(volume_file) == 1:
            volume_file = volume_file[0]
        else:
            volume_file = [f for f in volume_file if ".mhd" in f][0]
        return self.load_raw_grid_data(volume_file, expid=expid, gridsize=200, visualize=False, 
                    save_to_obj=True, orientation="coronal", **kwargs)

    """
        ########## DEBUG FUNCTIONS ############
    """
    @staticmethod
    def display_image(img, with_napari=True):
        if not with_napari:
            if isinstance(img, str):
                image = Image.open(img)
            elif isinstance(img, np.ndarray):
                image = Image.fromarray(img)
            elif isinstance(img, bytes):
                image = Image.frombytes(img)
            else:
                raise ValueError("image data type unknown")
            image.show()
        else:
            with napari.gui_qt():
                if len(img.shape == 2):
                    viewer = napari.view_image(img, rgb=False)
                else:
                    viewer = napari.view_image(img, rgb=True)

    

if __name__ == "__main__":
    api = GeneExpressionAPI(debug=False)
    # api.load_raw_grid_data("Data/ABA/gene_expression/71670687/energy.mhd", threshold=80, 
    #                         metadata_file=None, gridsize=200, visualize=True)
    # api.get_gene_expression_to_obj(71670687, threshold=99.99)


    for expid in api.HCF_genes:
        api.get_cells_for_experiment(expid, image_type="expression", overwrite=True, threshold=204)




    # api.load_cells(api.test_dataset_id)