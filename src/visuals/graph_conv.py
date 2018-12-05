import os
import sys
import numpy as np
from vispy import app, scene, visuals
import argparse

def parse_args():
    '''
    '''
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser('graph_conv.py')
    add_arg = parser.add_argument
    add_arg('data_path', nargs='?', default='/')
    add_arg('-v', '--verbose', action='store_true')

    return parser.parse_args()

def plot_edges(view, root_node, neighbor_nodes, active, d=1):
    '''
    '''
    tpos = []
    tcolors = []
    for _ in neighbor_nodes:
        pos = np.concatenate([np.expand_dims(root_node, axis=0),
                                np.expand_dims(_, axis=0),
                                np.expand_dims(root_node, axis=0)], axis=0)
        dist = np.linalg.norm(root_node-_)
        c = 120/np.log(40)
        #e_dist = np.exp((dist*np.log(0.025))/d)
        e_dist = 1 / np.exp(dist/c)
        if active: color = np.array([[e_dist,0,0,e_dist] for j in range(3)])
        else:
            color = np.array([[1-e_dist,1-e_dist,1-e_dist,e_dist] for j in range(3)])
        tpos.append(pos)
        tcolors.append(color)

    l = scene.visuals.Line(pos=np.concatenate(tpos,axis=0), color=np.concatenate(tcolors,axis=0),width=0.1)
    view.add(l)

def parse_graph(path):
    '''
    '''
    # Parse Protein Graph
    v = []
    c = []
    with open(path, 'r')as f:
        for i, _ in enumerate(f):
            row = _[:-1].split()
            res = [0 for _ in range(23)]
            res[int(row[3])] = 1
            v.append(res)
            c.append(row[-3:])
    v = np.array(v, dtype=float)
    c = np.array(c, dtype=float)
    c = c - c.mean(axis=0) # Center on origin

    data_ = [v,c]

    return data_

if __name__ == '__main__':

    # Parse the command line
    args = parse_args()
    data_path = args.data_path
    verbose = args.verbose

    # Parse data
    v,c = parse_graph(data_path)
    colors = np.ones((len(c),3))

    # Render data
    canvas = scene.SceneCanvas(size=(800, 600), show=True, keys='interactive', bgcolor='white', title=data_path.split('/')[-1])
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'

    p1 = scene.visuals.Markers(pos=c, face_color=colors)
    view.add(p1)

    ps = scene.visuals.Line(pos=c, color=[0,0,1],width=1.0)
    view.add(ps)

    plot_edges(view, c[0], c, True)
    for cc in c:
        plot_edges(view, cc, c, False, d=np.max(np.abs(c)))

    canvas.show()
    if sys.flags.interactive == 0:
        app.run()
