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

from BrainRender.Utils.paths_manager import Paths
from BrainRender.Utils.data_io import connected_to_internet
from BrainRender.Utils.webqueries import request




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

    image_download_url = ("http://api.brain-map.org/api/v2/image_download/IMAGEID")

    # URLs based on https://github.com/efferencecopy/ecallen/blob/master/ecallen/images.py
    experiment_metadata_url = ('http://api.brain-map.org/api/v2/data/query.json?criteria='
                        'model::SectionDataSet'
                        ',rma::criteria,[id$eqEXPID]'
                        ',rma::include,genes,plane_of_section,probes'
                        ',products,reference_space,specimen,treatments'
                        )

    def __init__(self):
        Paths.__init__(self)


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
            url = self.image_download_url.replace("IMAGEID", str(iid))
            res = request(url)
            # save binary data to an image
            with open(os.path.join(dest_folder, str(iid)+".png"), "wb") as f:
                f.write(res.content)

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

        probe_ch = [
            params['red_channel'].lower() in probes,
            params['green_channel'].lower() in probes,
            params['blue_channel'].lower() in probes,
        ]

        # open the image
        img = skio.imread(img_path)

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

    def find_cells_in_image(self, img_path, expid):
        # grab the RNA probes from the imaging parameters
        img_params = self.get_imaging_params(expid)

        # run the segmentation function
        rprops = self..extract_region_props(img_path,
                                            img_params,
                                            img_params['probes'],
                                            )

        # grab the X and Y pixels at the center of each labeled region
        cell_x_pix = np.array([roi['centroid'][1] for roi in rprops])
        cell_y_pix = np.array([roi['centroid'][0] for roi in rprops])

        # Align the cells coordinates to the Alle CCF v3
        img_id = os.path.split(img_path)[0]
        pir = self.align_cell_coords_to_ccf(cell_x_pix,
                      cell_y_pix,
                      expid,
                      section_image_id
                      )

if __name__ == "__main__":
    api = GeneExpressionAPI()
    # api.search_experiments_ids("sagittal", "Adora2a", fetch_images=True)
    api.get_imaging_params(70813257)
