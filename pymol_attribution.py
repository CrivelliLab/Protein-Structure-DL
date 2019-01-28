from pymol import cmd
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
cmap = cm.seismic
norm = Normalize(vmin=-1, vmax=1)

pdb_id = '4m21_a'
attributions_path = 'out/krashras_graph_new/interpret/attributions.npz'
data_path = 'data/KrasHras/pdb/'


# Load Attribution
data = np.load(attributions_path)
attributions = data['data']
labels = data['labels']
ind = np.where(labels == pdb_id)
attribution = attributions[ind][0][:,-4]

# Load PDB
cmd.reinitialize()
cmd.bg_color('white')
cmd.load(data_path+pdb_id[:-2]+'.pdb')
cmd.split_chains()
for name in cmd.get_names('objects', 0, '(all)'):
    if not name.endswith(pdb_id[-1].upper()):
        cmd.delete(name)
cmd.reset()

for i, _ in enumerate(attribution):
    cmd.select('toBecolored', 'res ' + str(i))
    cmd.set_color('saliency'+str(i), list(cmap(norm(_)))[:3])
    cmd.color('saliency'+str(i), 'toBecolored')

cmd.select('selected','chain '+pdb_id[-1].upper())
cmd.show('mesh', 'selected')
cmd.deselect()
