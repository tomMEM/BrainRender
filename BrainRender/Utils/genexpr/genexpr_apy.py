import sys
sys.path.append('./')

import os
import pandas as pd
from tqdm import tqdm
import numpy as np

from BrainRender.Utils.paths_manager import Paths
from BrainRender.Utils.data_io import connected_to_internet
from BrainRender.Utils.webqueries import request




class GeneExpressionAPI(Paths):
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



if __name__ == "__main__":
    api = GeneExpressionAPI()
    api.search_experiments_ids("sagittal", "Adora2a", fetch_images=True)
