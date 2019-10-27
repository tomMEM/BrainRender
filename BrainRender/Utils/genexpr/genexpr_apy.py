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
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None # <- deactivate image bomb error

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

    image_download_url = ("http://api.brain-map.org/api/v2/section_image_download/IMAGEID")

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


    def __init__(self):
        Paths.__init__(self)
        self.test_dataset_id = 167643437

    """
        ################## DATA IO ########################
    """
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


    def downlad_images(self, expid, image_ids, dest_folder=None):
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

        print("Downloading {} images for experiment: {}".format(len(image_ids), expid))
        for iid in tqdm(image_ids):
            savefile = os.path.join(dest_folder, str(iid)+".png")
            if os.path.isfile(savefile): continue

            url = self.image_download_url.replace("IMAGEID", str(iid))
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

    def get_cells_for_experiment(self, expid=None, exp_data_path=None):
        if exp_data_path is not None:
            exp_images = listdir(exp_data_path)
            if expid is None: raise ValueError("Need to pass a value for expid")
        elif expid is not None:
            # download the images for the experiment:
            self.fetch_images_for_exp(expid)

            # get the path to the images
            exp_images = listdir(os.path.join(self.gene_expression, str(expid)))
        else:
            raise ValueError("Need to pass either expid or exp_data_path to get_cells_for_experiment function")

        # Get cells aligned to ccf
        print("Extracting cells for experiment: {}".format(expid))
        cells = {"x":[], "y":[], "z":[]}
        for img in tqdm(exp_images):
            img_cells = self.find_cells_in_image(img, expid)
            cells['x'].extend(img_cells[0])
            cells['y'].extend(img_cells[1])
            cells['z'].extend(img_cells[2])
            break
        cells = pd.DataFrame(cells)
        savepath = os.path.join(self.gene_expression, str(expid))
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        
        cells.to_pickle(os.path.join(savepath, str(expid)+".pkl"))
        return cells
    
    def load_cells(self, expid=None, exp_data_path=None):
        if exp_data_path is not None:
            return pd.read_pickle(exp_data_path)
        else:
            return pd.read_pickle(os.path.join(self.gene_expression, str(expid),str(expid)+".pkl" ))


    def find_cells_in_image(self, img_path, expid):
        # grab the RNA probes from the imaging parameters
        img_params = self.get_imaging_params(expid)

        # run the segmentation function
        rprops = self.extract_region_props(img_path,
                                            img_params,
                                            img_params['probes'],
                                            )

        # grab the X and Y pixels at the center of each labeled region
        cell_x_pix = np.array([roi['centroid'][1] for roi in rprops])
        cell_y_pix = np.array([roi['centroid'][0] for roi in rprops])

        # Align the cells coordinates to the Alle CCF v3
        img_id = int(strip_path(img_path)[-1].split(".")[0])
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
        t_2d = self.get_affine_2d(section_image_id)
        tmtx_tsv = np.hstack((t_2d['A_mtx'], t_2d['translation']))
        tmtx_tsv = np.vstack((tmtx_tsv, [0, 0, 1]))  # T matrix for 2D affine
        data_mtx = np.vstack((x_pix, y_pix, np.ones_like(x_pix)))  # [3 x Npix]
        xy_2d_align = np.dot(tmtx_tsv, data_mtx)

        # implement the 3D affine transform for section_to_CCF coordinates
        t_3d = self.get_affine_3d(section_data_set_id)
        tmtx_tvr = np.hstack((t_3d['A_mtx'], t_3d['translation']))
        tmtx_tvr = np.vstack((tmtx_tvr, [0, 0, 0, 1]))

        data_mtx = np.vstack((xy_2d_align[0, :], xy_2d_align[1, :],
                            np.ones((1, xy_2d_align.shape[1])) * t_2d['section_number'] * t_3d['section_thickness'],
                            np.ones((1, xy_2d_align.shape[1]))))

        xyz_3d_align = np.dot(tmtx_tvr, data_mtx)
        pir = xyz_3d_align[0:3, :]
        return pir

if __name__ == "__main__":
    api = GeneExpressionAPI()
    # api.search_experiments_ids("sagittal", "Adora2a", fetch_images=True)
    # api.get_cells_for_experiment(api.test_dataset_id)
    api.load_cells(api.test_dataset_id)
