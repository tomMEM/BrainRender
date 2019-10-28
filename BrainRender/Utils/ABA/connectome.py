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