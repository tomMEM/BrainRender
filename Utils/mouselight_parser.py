import sys
sys.path.append('./')

import os
import json
from vtkplotter import *
import pandas as pd
from tqdm import tqdm
import numpy as np

from Utils.data_io import load_json
from Utils.data_manipulation import get_coords
from colors import *
from variables import *

# TODO fix axons initial segment missing from renderes

def neurites_parser(neurites, neurite_radius, color):
    """[Given a dataframe with all the samples for some neurites, create "Tube" actors that render each neurite segment.]
    
    Arguments:
        neurites {[DataFrame]} -- [dataframe with each sample for the neurites]
        neurite_radius {[float]} -- [radius of the Tube actors]
        color {[color object]} -- [color to be assigned to the Tube actor]
    

    Returns:
        actors {[list]} -- [list of VTK actors]

    ----------------------------------------------------------------
    This function works by first identifyingt the branching points of a neurite structure. Then each segment between either two branchin points
    or between a branching point and a terminal is modelled as a Tube. This minimizes the number of actors needed to represent the neurites
    while stil accurately modelling the neuron. 

    Known issue: the axon initial segment is missing from renderings. 
    """

    # get branching points
    parent_counts = neurites["parentNumber"].value_counts()
    branching_points = parent_counts.loc[parent_counts > 1]

    # loop over each branching point
    actors = []
    for idx, bp in branching_points.iteritems():
        # get neurites after the branching point
        bp = neurites.loc[neurites.sampleNumber == idx]
        post_bp = neurites.loc[neurites.parentNumber == idx]
        
        # loop on each branch after the branching point
        for bi, branch in post_bp.iterrows():

            parent = neurites.loc[neurites.sampleNumber == branch.parentNumber]
            branch_points = [get_coords(parent), get_coords(bp), get_coords(branch)] # this list stores all the samples that  are part of a branch

            # loop over all following points along the branch, until you meet either a terminal or another branching point. store the points
            idx = branch.sampleNumber
            while True:
                nxt = neurites.loc[neurites.parentNumber == idx]
                if len(nxt) != 1: 
                    break
                else:
                    branch_points.append(get_coords(nxt))
                    idx += 1

            # if the branch is too short for a tube, create a sphere instead
            if len(branch_points) < 2: # plot either a line between two branch_points or  a spheere
                actors.append(Sphere(branch_points[0], c="g", r=100))
                continue 
            
            # create tube actor
            actors.append(shapes.Tube(branch_points, r=neurite_radius, c=color, alpha=1, res=NEURON_RESOLUTION))
        
    return actors


def render_neurons(ml_file, render_neurites = True,
                neurite_radius=None, 
                color_neurites=True, axon_color=None, soma_color=None, dendrites_color=None, random_color=False):
    
    """[Given a file with JSON data about neuronal structures downloaded from the Mouse Light neurons browser website, 
       this function creates VTKplotter actors that can be used to render the neurons, returns them as nested dictionaries]

    Arguments:
        ml_file {[string]} -- [path to the JSON MouseLight file]
        render_neurites {[boolean]} -- [If false neurites are not rendered, just the soma]
        neurite_radius {[float]} -- [radius of the "Tube" used to render neurites, it's also used to scale the sphere used for the soma. If set to None the default is used]
        color_neurites {[Bool]} -- [default: True. If true, soma axons and dendrites are colored differently, if false each neuron has a single color (the soma color)]
        axon_color, soma_color, dendrites_color {[String, array, list]} -- [if list it needs to have the same length as the number of neurons being rendered to specify the colors for each neuron. 
                                            colors can be either strings (e.g. "red"), arrays (e.g.[.5, .5,. 5]) or variables (e.g see colors.py)]
        random_color {[Bool]} -- [if True each neuron will have one color picked at random among those defined in colors.py]

    Returns:
        actors [list] -- [list of dictionaries, each dictionary contains the VTK actors of one neuron]
    """
    # Check neurite radius
    if neurite_radius is None:
        neurite_radius = DEFAULT_NEURITE_RADIUS
    
    # Load the data
    data = load_json(ml_file)
    data = data["neurons"]
    print("Found {} neurons".format(len(data)))

    # Loop over neurons
    actors = []
    for neuron in tqdm(data):
        # Define colors of different components
        if random_color:
            color = get_random_colors(n_colors=1)
            axon_color = soma_color = dendrites_color = color
        else:
            if soma_color is None:
                print("No soma color is provided, picking a random one")
                soma_color = get_random_colors(n_colors=1)

            if not color_neurites:
                axon_color = dendrites_color = soma_color
            else:
                if axon_color is None:
                    print("No axon color provided, using soma color")
                    axon_color = soma_color
                if dendrites_color is None:
                    print("No dendrites color provided, using soma color")
                    dendrites_color = soma_color

        if not check_colors([soma_color, axon_color, dendrites_color]):
            raise ValueError("The colors chosen are not valid: soma - {}, dendrites {}, axon {}".format(soma_color, dendrites_color, axon_color))


        # create soma actor
        neuron_actors = {}

        soma_coords = get_coords(neuron["soma"])
        soma = Sphere(pos=soma_coords, c=soma_color, r=SOMA_RADIUS)
        neuron_actors['soma'] = soma

        # Draw dendrites and axons
        if render_neurites:
            neuron_actors['dendrites'] = neurites_parser(pd.DataFrame(neuron["dendrite"]), neurite_radius, dendrites_color)
            neuron_actors['axon'] = neurites_parser(pd.DataFrame(neuron["axon"]), neurite_radius, axon_color)
        else:
            neuron_actors['dendrites'] = []
            neuron_actors['axon'] = []

        actors.append(neuron_actors)
    return actors


def test():
    """
        Small function used to test the render_neurons function above. Specify a file path and run it
    """
    res = render_neurons(NEURONS_FILE,
                render_neurites = True,
                neurite_radius=None, 
                color_neurites=False, axon_color="red", soma_color="red", dendrites_color="blue", 
                random_color=True)

    vp = Plotter(title='first example')
    for neuron in res:
        vp.show(neuron['soma'], *neuron['dendrites'], *neuron['axon'])
    

if __name__ == "__main__":
    test()