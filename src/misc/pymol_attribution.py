from pymol import cmd, stored
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
cmap = cm.seismic
norm = Normalize(vmin=-1, vmax=1)

def zero_residues(sel1,offset=0,chains=0):
        """
        """
        offset = int(offset)

        # variable to store the offset
        stored.first = None
        # get the names of the proteins in the selection

        names = ['(model %s and (%s))' % (p, sel1)
                        for p in cmd.get_object_list('(' + sel1 + ')')]

        if int (chains):
                names = ['(%s and chain %s)' % (p, chain)
                                for p in names
                                for chain in cmd.get_chains(p)]

        # for each name shown
        for p in names:
                # get this offset
                ok = cmd.iterate("first %s and polymer and n. CA" % p,"stored.first=resv")
                # don't waste time if we don't have to
                if not ok or stored.first == offset:
                        continue;
                # reassign the residue numbers
                cmd.alter("%s" % p, "resi=str(int(resi)-%s)" % str(int(stored.first)-offset))
                # update pymol

        cmd.rebuild()

def stucture_attribution(pdb_id, attributions_path, pdb_path, flag=False):
    '''
    '''
    data = np.load(attributions_path)
    attributions = data['data']
    labels = data['labels']
    offsets = data['offsets']
    ind = np.where(labels == pdb_id)
    attribution = attributions[ind][0][:,-1]

    # Load PDB
    cmd.bg_color('black')
    cmd.load(pdb_path+pdb_id[:-2]+'.pdb')
    cmd.split_chains(pdb_id[:-2])
    for name in cmd.get_names('objects', 0, '(all)'):
        if not name.endswith(pdb_id[-1].upper()) and name.startswith(pdb_id[:4]):
            cmd.delete(name)
        else: zero_residues(name)
    cmd.reset()

    cmd.color('white', pdb_id)
    for i, _ in enumerate(attribution):
        if flag: _ = _ *-1
        cmd.select('toBecolored', pdb_id+ ' and res ' + str(i+offsets[ind][0]))
        cmd.set_color('saliency'+str(i)+pdb_id, list(cmap(norm(_)))[:3])
        cmd.color('saliency'+str(i)+pdb_id, 'toBecolored')

    cmd.select('selected', pdb_id)
    cmd.show('mesh', 'selected')
    cmd.deselect()

def kernel_attribution(pdb_id, attributions_path, pdb_path, flag=False):
    '''
    '''
    data = np.load(attributions_path)
    attributions = data['kernels']
    labels = data['labels']
    offsets = data['offsets']
    ind = np.where(labels == pdb_id)
    attribution = attributions[ind][0][:,-1]

    # Load PDB
    cmd.bg_color('black')
    cmd.load(pdb_path+pdb_id[:-2]+'.pdb')
    cmd.split_chains(pdb_id[:-2])
    for name in cmd.get_names('objects', 0, '(all)'):
        if not name.endswith(pdb_id[-1].upper()) and name.startswith(pdb_id[:4]):
            cmd.delete(name)
        else: zero_residues(name)
    cmd.reset()

    pdb_id2 = pdb_id+'copy'
    cmd.create(pdb_id2, pdb_id)

    cmd.hide('everything',pdb_id)
    cmd.show_as('lines', pdb_id + ' and (name ca or name c or name n)')
    cmd.set('line_width', 5)
    cmd.set_bond('line_width', 5,  pdb_id + ' and (name ca or name c or name n)')
    cmd.show('spheres', pdb_id + ' and name ca')
    cmd.set('sphere_transparency', 0.0, pdb_id + ' and name ca')
    cmd.set('sphere_scale', 0.5, pdb_id + ' and name ca')

    cmd.hide('everything',pdb_id2)
    cmd.show('spheres', pdb_id2 + ' and name ca')
    cmd.set('sphere_transparency', 0.8, pdb_id2 + ' and name ca')
    cmd.set('sphere_scale', code2[elem], elem+'&'+selection)

    cmd.color('white', pdb_id)
    for i, _ in enumerate(attribution):
        if flag: _ = _ *-1
        cmd.select('toBecolored', pdb_id+ ' and res ' + str(i+offsets[ind][0]))
        cmd.set_color('saliency'+str(i)+pdb_id, list(cmap(norm(_)))[:3])
        cmd.color('saliency'+str(i)+pdb_id, 'toBecolored')

    cmd.select('selected', pdb_id)
    cmd.show('mesh', 'selected')
    cmd.deselect()

################################################################################

# Paths
attributions_path = '../../out/krashras_graph_new/interpret/attributions.npz'
pdb_path = '../../data/KrasHras/pdb/'


# Load Attribution
#kras = ['5uqw_a']
#hras= ['3lo5_c']
kras = ['3gft_d', '4dsn_a', '5uqw_b', '4lyh_b']
hras = ['4us2_r', '3kkn_a', '2quz_a', '3kud_a']

cmd.reinitialize()
for _ in kras:
    stucture_attribution(_, attributions_path, pdb_path)
for _ in hras:
    stucture_attribution(_, attributions_path, pdb_path, True)

# Align and Translate
for _ in kras[1:]:
    cmd.align(_,  kras[0])
for _ in hras:
    cmd.align(_,  kras[0])
cmd.orient()

for i, _ in enumerate(kras):
    cmd.translate([0,50*i,0],selection=_[:-1]+_[-1].upper())

for i, _ in enumerate(hras):
    cmd.translate([60,50*i,00],selection=_[:-1]+_[-1].upper())

cmd.reset()
