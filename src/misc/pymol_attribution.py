from pymol import cmd, stored
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
cmap = cm.seismic
norm = Normalize(vmin=-1, vmax=1)
cmap2 = cm.PRGn
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
    kernels = data['kernels']
    labels = data['labels']
    offsets = data['offsets']
    ind = np.where(labels == pdb_id)
    #attribution = attributions[ind][0][:,-1]
    if flag:
        a_ = data["all_"][0]
        a_[a_<=0] = 0.0
        attribution = data['all_'][0] + data['all_'][1]
        attribution[attribution<=0] = 0.0
        attribution = a_ * attribution
    else:
        a_ = data["all_"][1]
        a_[a_<=0] = 0.0
        attribution = data['all_'][0] + data['all_'][1]
        attribution[attribution<=0] = 0.0
        attribution = a_ * attribution

    # Load PDB
    cmd.load(pdb_path+pdb_id[:-2]+'.pdb')
    cmd.split_chains(pdb_id[:-2])
    for name in cmd.get_names('objects', 0, '(all)'):
        if not name.endswith(pdb_id[-1].upper()) and name.startswith(pdb_id[:4]):
            cmd.delete(name)
        else: zero_residues(name)
    cmd.reset()

    cmd.color('white', pdb_id)
    for i, _ in enumerate(attribution):
        #if flag: _ = _ *-1
        cmd.select('toBecolored', pdb_id+ ' and res ' + str(i+offsets[ind][0]))
        cmd.set_color('saliency'+str(i)+pdb_id, list(cmap(norm(_)))[:3])
        #else: cmd.set_color('saliency'+str(i)+pdb_id, list(cmap2(norm(_)))[:3])
        cmd.color('saliency'+str(i)+pdb_id, 'toBecolored')

    cmd.select('selected', pdb_id)
    #cmd.show('mesh', 'selected')
    #cmd.show('sticks', 'selected')
    cmd.deselect()

def kernels(pdb_id, data_path, pdb_path, flag=False):
    '''
    '''
    data = np.load(data_path)
    kernels = data['kernels']
    labels = data['labels']
    offsets = data['offsets']
    ind = np.where(labels == pdb_id)
    kernels = kernels[ind][0][0]

    # Load PDB
    cmd.load(pdb_path+pdb_id[:-2]+'.pdb')
    cmd.split_chains(pdb_id[:-2])
    for name in cmd.get_names('objects', 0, '(all)'):
        if not name.endswith(pdb_id[-1].upper()) and name.startswith(pdb_id[:4]):
            cmd.delete(name)
        else: zero_residues(name)
    cmd.reset()

    pdb_id2 = pdb_id+'kernel'
    cmd.create(pdb_id2, pdb_id)
    cmd.delete(pdb_id)

    '''
    cmd.hide('everything',pdb_id)
    cmd.show_as('lines', pdb_id + ' and (name ca or name c or name n)')
    cmd.set('line_width', 5)
    cmd.set_bond('line_width', 5,  pdb_id + ' and (name ca or name c or name n)')
    cmd.show('spheres', pdb_id + ' and name ca')
    cmd.set('sphere_transparency', 0.0, pdb_id + ' and name ca')
    cmd.set('sphere_scale', 0.5, pdb_id + ' and name ca')
    '''

    cmd.hide('everything',pdb_id2)
    cmd.show('spheres', pdb_id2 + ' and name ca')
    cmd.set('sphere_transparency', 0.9, pdb_id2 + ' and name ca')

    for i, _ in enumerate(kernels):
        print(_)
        cmd.set('sphere_scale', _/(1.7*2), pdb_id2+ ' and res ' + str(i+offsets[ind][0])+ ' and name ca')
    cmd.deselect()

################################################################################

# Paths
data_path = '../../out/krashras_graph_new/interpret/attributions.npz'
pdb_path = '../../data/KrasHras/pdb/'

# Load Attribution
kras = ['5tb5_c', '4dsn_a', '3gft_f', '5v6v_b', '4luc_b',
             '4m22_a', '4lv6_b', '4epy_a', '4lrw_b', '4epx_a']
             #'4m1w_a', '5uqw_a', '4pzy_b', '4m21_a', '5us4_a',
             #'5f2e_a', '4q03_a', '4lyh_c', '4m1o_b', '4pzz_a']
hras = ['1p2v_a', '4urz_r', '2x1v_a', '3lo5_c', '2quz_a',
            '3kud_a', '1aa9_a', '1plk_a', '4k81_d', '5wdq_a']
            #'1iaq_a', '1xd2_a', '3i3s_r', '4efl_a', '4l9w_a',
            #'3lo5_a', '5b2z_a', '1nvv_q', '4efm_a', '3l8z_a']

cmd.reinitialize()
cmd.bg_color('black')
for _ in kras:
    #kernels(_, data_path, pdb_path)
    stucture_attribution(_, data_path, pdb_path)
for _ in hras:
    #kernels(_, data_path, pdb_path)
    stucture_attribution(_, data_path, pdb_path, True)

# Align and Translate
for _ in kras[1:]:
    cmd.align(_,  kras[0])
for _ in hras:
    cmd.align(_,  kras[0])
cmd.orient()

for i, _ in enumerate(kras):
    cmd.rotate([1.0,0.0,0.0],60,selection=_[:-1]+_[-1].upper())
    cmd.translate([0,50*0,0],selection=_[:-1]+_[-1].upper())

for i, _ in enumerate(hras):
    cmd.rotate([1.0,0.0,0.0],60,selection=_[:-1]+_[-1].upper())
    cmd.translate([-60,50*0,00],selection=_[:-1]+_[-1].upper())


cmd.reset()
