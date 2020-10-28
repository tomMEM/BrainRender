from rich.progress import track
import pandas as pd

try:
    from allensdk.api.queries.mouse_connectivity_api import (
        MouseConnectivityApi,
    )

    mca = MouseConnectivityApi()
    allen_sdk_installed = True
except ModuleNotFoundError:
    allen_sdk_installed = False


from brainrender import base_dir
from .._utils import listify
from .._io import request

streamlines_folder = base_dir / "streamlines"
streamlines_folder.mkdir(exist_ok=True)


def experiments_source_search(SOI):
    """
        Returns data about experiments whose injection was in the SOI, structure of interest
        :param SOI: str, structure of interest. Acronym of structure to use as seed for teh search
        :param source:  (Default value = True)
        """

    transgenic_id = 0  # id = 0 means use only wild type
    primary_structure_only = True

    return pd.DataFrame(
        mca.experiment_source_search(
            injection_structures=listify(SOI),
            target_domain=None,
            transgenic_lines=transgenic_id,
            primary_structure_only=primary_structure_only,
        )
    )


def get_streamlines_data(eids, force_download=False):
    """
        Given a list of expeirmental IDs, it downloads the streamline data from the https://neuroinformatics.nl cache and saves them as
        json files. 

        :param eids: list of integers with experiments IDs


    """
    data = []
    for eid in track(eids, total=len(eids), description="downloading"):
        url = "https://neuroinformatics.nl/HBP/allen-connectivity-viewer/json/streamlines_{}.json.gz".format(
            eid
        )

        jsonpath = streamlines_folder / f"{eid}.json"

        if not jsonpath.exists() or force_download:
            response = request(url)

            # Write the response content as a temporary compressed file
            temp_path = streamlines_folder / "temp.gz"
            with open(str(temp_path), "wb") as temp:
                temp.write(response.content)

            # Open in pandas and delete temp
            url_data = pd.read_json(
                str(temp_path), lines=True, compression="gzip"
            )
            temp_path.unlink()

            # save json
            url_data.to_json(str(jsonpath))

            # append to lists and return
            data.append(url_data)
        else:
            data.append(pd.read_json(str(jsonpath)))
    return data
