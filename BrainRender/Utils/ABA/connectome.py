import sys
sys.path.append("./")

import numpy as np
from vtkplotter import Lines, Spheres, Line, Tube, Assembly
from tqdm import tqdm

from BrainRender.scene import Scene
from BrainRender.colors import *

class Connectome(Scene):
    def __init__(self, *args, **kwargs):
        Scene.__init__(self, *args, **kwargs)
        self.connections = None

    def get_connectome_for_region(self, region, *args, **kwargs):
        self.seed = region
        mtx = self.get_projection_matrix_for_region(region, *args, **kwargs)

        # Average across experiments
        expr = np.mean(mtx['matrix'], 0)
        target_regions = [self.structure_tree.get_structures_by_id([r['structure_id']])[0]['acronym'] for r in mtx['columns']]
        self.connections =  {reg:val for reg, val in zip(target_regions, expr)}
        return self.connections
    
    def print_structures(self):
        acronyms, names = self.structures.acronym.values, self.structures['name'].values
        sort_idx = np.argsort(acronyms)
        acronyms, names = acronyms[sort_idx], names[sort_idx]
        [print("({}) - {}".format(a, n)) for a,n in zip(acronyms, names)]

    def experiments_source_search(self, SOI, *args, source=True,  **kwargs):
        """
            [Returns data about experiments whose injection was in the SOI, structure of interest]
            Arguments:
                SOI {[str]} -- [acronym of the structure of interest to look at]
        """
        """
            list of possible kwargs
                injection_structures : list of integers or strings
                    Integer Structure.id or String Structure.acronym.
                target_domain : list of integers or strings, optional
                    Integer Structure.id or String Structure.acronym.
                injection_hemisphere : string, optional
                    'right' or 'left', Defaults to both hemispheres.
                target_hemisphere : string, optional
                    'right' or 'left', Defaults to both hemispheres.
                transgenic_lines : list of integers or strings, optional
                    Integer TransgenicLine.id or String TransgenicLine.name. Specify ID 0 to exclude all TransgenicLines.
                injection_domain : list of integers or strings, optional
                    Integer Structure.id or String Structure.acronym.
                primary_structure_only : boolean, optional
                product_ids : list of integers, optional
                    Integer Product.id
                start_row : integer, optional
                    For paging purposes. Defaults to 0.
                num_rows : integer, optional
                    For paging purposes. Defaults to 2000.

        """
        transgenic_id = kwargs.pop('transgenic_id', 0) # id = 0 means use only wild type
        primary_structure_only = kwargs.pop('primary_structure_only', True)

        if not isinstance(SOI, list): SOI = [SOI]

        if source:
            injection_structures=SOI
            target_domain = None
        else:
            injection_structures = None
            target_domain = SOI

        return pd.DataFrame(self.mca.experiment_source_search(injection_structures=injection_structures,
                                            target_domain = target_domain,
                                            transgenic_lines=transgenic_id,
                                            primary_structure_only=primary_structure_only))

    def experiments_target_search(self, *args, **kwargs):
        return self.experiments_source_search(*args, source=False, **kwargs)

    def fetch_experiments_data(self, experiments_id, *args, average_experiments=False, base_structures=True, **kwargs):
        if isinstance(experiments_id, np.ndarray):
            experiments_id = [int(x) for x in experiments_id]
        elif not isinstance(experiments_id, list): 
            experiments_id = [experiments_id]
        if [x for x in experiments_id if not isinstance(x, int)]:
            raise ValueError("Invalid experiments_id argument: {}".format(experiments_id))

        default_structures_ids = self.structures.id.values


        is_injection = kwargs.pop('is_injection', False) # Include only structures that are not injection
        structure_ids = kwargs.pop('structure_ids', default_structures_ids) # Pass IDs of structures of interest 
        hemisphere_ids= kwargs.pop('hemisphere_ids', None) # 1 left, 2 right, 3 both

        if not average_experiments:
            return pd.DataFrame(self.mca.get_structure_unionizes(experiments_id,
                                                is_injection = is_injection,
                                                structure_ids = structure_ids,
                                                hemisphere_ids = hemisphere_ids))
        else:
            raise NotImplementedError("Need to find a way to average across experiments")
            unionized = pd.DataFrame(self.mca.get_structure_unionizes(experiments_id,
                                                is_injection = is_injection,
                                                structure_ids = structure_ids,
                                                hemisphere_ids = hemisphere_ids))

        for regionid in list(set(unionized.structure_id)):
            region_avg = unionized.loc[unionized.structure_id == regionid].mean(axis=1)

    ####### ANALYSIS ON EXPERIMENTAL DATA
    def analyze_efferents(self, SOI, projection_metric = None):
        """[Loads the experiments on SOI and looks at average statistics of efferent projections]
        
        Arguments:
            SOI {[str]} -- [acronym of the structure of interest to look at]
        """
        if projection_metric is None: 
            projection_metric = self.projection_metric

        experiment_data = pd.read_pickle(os.path.join(self.output_data, "{}.pkl".format(SOI)))
        experiment_data = experiment_data.loc[experiment_data.volume > self.volume_threshold]

        # Loop over all structures and get the injection density
        results = {"left":[], "right":[], "both":[], "id":[], "acronym":[], "name":[]}
        for target in self.structures.id.values:
            target_acronym = self.structures.loc[self.structures.id == target].acronym.values[0]
            target_name = self.structures.loc[self.structures.id == target].name.values[0]

            exp_target = experiment_data.loc[experiment_data.structure_id == target]

            exp_target_hemi = self.hemispheres(exp_target.loc[exp_target.hemisphere_id == 1], 
                                                exp_target.loc[exp_target.hemisphere_id == 2], 
                                                exp_target.loc[exp_target.hemisphere_id == 3])
            proj_energy = self.hemispheres(np.nanmean(exp_target_hemi.left[projection_metric].values),
                                            np.nanmean(exp_target_hemi.right[projection_metric].values),
                                            np.nanmean(exp_target_hemi.both[projection_metric].values)
            )


            for hemi in self.hemispheres_names:
                results[hemi].append(proj_energy._asdict()[hemi])
            results["id"].append(target)
            results["acronym"].append(target_acronym)
            results["name"].append(target_name)

        results = pd.DataFrame.from_dict(results).sort_values("right", na_position = "first")
        return results

    def analyze_afferents(self, SOI, projection_metric = None):
        """[Loads the experiments on SOI and looks at average statistics of afferent projections]
        
        Arguments:
            SOI {[str]} -- [structure of intereset]
        """
        if projection_metric is None: 
            projection_metric = self.projection_metric
        SOI_id = self.structure_tree.get_structures_by_acronym([SOI])[0]["id"]

        # Loop over all strctures and get projection towards SOI
        results = {"left":[], "right":[], "both":[], "id":[], "acronym":[], "name":[]}

        for origin in self.structures.id.values:
            origin_acronym = self.structures.loc[self.structures.id == origin].acronym.values[0]
            origin_name = self.structures.loc[self.structures.id == origin].name.values[0]

            experiment_data = pd.read_pickle(os.path.join(self.output_data, "{}.pkl".format(origin_acronym)))
            experiment_data = experiment_data.loc[experiment_data.volume > self.volume_threshold]

            exp_target = experiment_data.loc[experiment_data.structure_id == SOI_id]
            exp_target_hemi = self.hemispheres(exp_target.loc[exp_target.hemisphere_id == 1], exp_target.loc[exp_target.hemisphere_id == 2], exp_target.loc[exp_target.hemisphere_id == 3])
            proj_energy = self.hemispheres(np.nanmean(exp_target_hemi.left[projection_metric].values),
                                            np.nanmean(exp_target_hemi.right[projection_metric].values),
                                            np.nanmean(exp_target_hemi.both[projection_metric].values)
            )
            for hemi in self.hemispheres_names:
                results[hemi].append(proj_energy._asdict()[hemi])
            results["id"].append(origin)
            results["acronym"].append(origin_acronym)
            results["name"].append(origin_name)

        results = pd.DataFrame.from_dict(results).sort_values("right", na_position = "first")
        return results

    ####### GET TRACTOGRAPHY AND SPATIAL DATA
    def get_projection_tracts_to_target(self, p0=None, **kwargs):
        """[Gets tractography data for all experiments whose projections reach the brain region or location of iterest.]
        
        Keyword Arguments:
            p0 {[list]} -- [list of 3 floats with XYZ coordinates of point to be used as seed] (default: {None})
        
        Raises:
            ValueError: [description]
            ValueError: [description]
        
        Returns:
            [type] -- [description]
        """

        # check args
        if p0 is None:
            raise ValueError("Please pass coordinates")
        elif isinstance(p0, np.ndarray):
            p0 = list(p0)
        elif not isinstance(p0, (list, tuple)):
            raise ValueError("Invalid argument passed (p0): {}".format(p0))

        tract = self.mca.experiment_spatial_search(seed_point=p0, **kwargs)

        if isinstance(tract, str): 
            raise ValueError('Something went wrong with query, query error message:\n{}'.format(tract))
        else:
            return tract

    ### OPERATIONS ON STRUCTURE TREES
    def get_structure_ancestors(self, regions, ancestors=True, descendants=False):
        """
            [Get's the ancestors of the region(s) passed as arguments]
        
        Arguments:
            regions {[str, list]} -- [List of acronyms of brain regions]
        """

        if not isinstance(regions, list):
            struct_id = self.structure_tree.get_structures_by_acronym([regions])[0]['id']
            return pd.DataFrame(self.tree_search.get_tree('Structure', struct_id, ancestors=ancestors, descendants=descendants))
        else:
            ancestors = []
            for region in regions:
                struct_id = self.structure_tree.get_structures_by_acronym([region])[0]['id']
                ancestors.append(pd.DataFrame(self.tree_search.get_tree('Structure', struct_id, ancestors=ancestors, descendants=descendants)))
            return ancestors

    def get_structure_descendants(self, regions):
        return self.get_structure_ancestors(regions, ancestors=False, descendants=True)

    def visualize_connectome(self, cutoff_perc=25, show_mesh=False):
        if self.connections is None:
            print("You need to run 'get_connectome_for_region' first!")

        # Add seed brain region
        self.add_brain_regions(self.seed, use_original_color=True, alpha=.8)
        self.edit_actors(self.actors['regions'][self.seed], wireframe=True)
        self.seed_coords = self.get_region_CenterOfMass(self.seed)

        # Get colors for each target region
        colors = self.get_region_color(list(self.connections.keys()))

        # Get variables for sphere and edges
        max_strength = np.percentile(list(self.connections.values()), 90)
        threshold = np.percentile(list(self.connections.values()), cutoff_perc)
        target_coords, sphere_colors, tubes = [], [], []
        for (region, strength), region_color in tqdm(zip(self.connections.items(), colors)):
            coords = self.get_region_CenterOfMass(region)
            target_coords.append(coords)
            sphere_colors.append(region_color)

            if strength > threshold:
                link_color = colorMap(strength, name='jet', vmin=0, vmax=max_strength)
                tubes.append(Tube([self.seed_coords, coords], c=link_color, alpha=.8, r=10, res=4))

                if show_mesh:
                    self.add_brain_regions(region, use_original_color=True, alpha=.4)
        self.actors['others'].append(Assembly(tubes))

        # Create spheres actors
        start_coords = [self.seed_coords for i in range(len(target_coords))]
        self.actors['others'].append(Spheres(target_coords, r=50, c=sphere_colors, alpha=.8, res=8))


if __name__ == '__main__':
    cc = Connectome()
    cc.get_connectome_for_region("MOs", hemisphere="right", parameter='projection_density')
    cc.visualize_connectome(cutoff_perc=90, show_mesh=True)
    cc.render()
