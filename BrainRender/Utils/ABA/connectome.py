import sys
sys.path.append("./")

import numpy as np
from vtkplotter import Lines, Spheres
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
    
    def visualize_connectome(self, cutoff_perc=25):
        if self.connections is None:
            print("You need to run 'get_connectome_for_region' first!")

        # Add seed brain region
        self.add_brain_regions(self.seed, use_original_color=True, alpha=.8)
        self.edit_actors(self.actors['regions'][self.seed], wireframe=True)
        self.seed_coords = self.get_region_CenterOfMass(self.seed)

        # Get colors for each target region
        colors = self.get_region_color(list(self.connections.keys()))

        # Get variables for sphere and edges
        max_strength = np.max(list(self.connections.values()))
        threshold = np.percentile(list(self.connections.values()), cutoff_perc)
        target_coords, line_colors, line_radiuses, sphere_colors = [], [], [], []
        for (region, strength), region_color in tqdm(zip(self.connections.items(), colors)):
            if strength <= threshold: continue
            link_color = colorMap(strength, name='jet', vmin=0, vmax=max_strength)
            coords = self.get_region_CenterOfMass(region)

            target_coords.append(coords)
            sphere_colors.append(region_color)
            line_colors.append(link_color)
            line_radiuses.append(10*strength)
        start_coords = [self.seed_coords for i in range(len(line_radiuses))]
        
        # Create actors
        self.actors['others'].append(Lines(start_coords, endPoints=target_coords, c="red", 
                                                alpha=.8, lw=2))
        self.actors['others'].append(Spheres(target_coords, r=50, c=sphere_colors, alpha=.8, res=8))


if __name__ == '__main__':
    cc = Connectome()
    cc.get_connectome_for_region("PAG")
    cc.visualize_connectome()
    cc.render()