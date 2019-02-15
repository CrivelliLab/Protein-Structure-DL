from pymol import cmd, stored
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
cmap = cm.seismic
norm = Normalize(vmin=-1, vmax=1)

def resicolor(selection='all', scale=False):

    '''USAGE: resicolor <selection>
    colors all or the given selection with arbitrary
    coloring scheme.
    '''
    cmd.select ('calcium','resn ca or resn cal')
    cmd.select ('acid','resn asp or resn glu or resn cgu')
    cmd.select ('basic','resn arg or resn lys or resn his')
    cmd.select ('nonpolar','resn met or resn phe or resn pro or resn trp or resn val or resn leu or resn ile or resn ala')
    cmd.select ('polar','resn ser or resn thr or resn asn or resn gln or resn tyr')
    cmd.select ('cys','resn cys or resn cyx')
    cmd.select ('backbone','name ca or name n or name c or name o')
    cmd.select ('none')

    code={'acid'    :  'red'    ,
          'basic'   :  'blue'   ,
          'nonpolar':  'orange' ,
          'polar'   :  'green'  ,
          'cys'     :  'yellow'}

    code2={'acid'    :  2    ,
          'basic'   :  2   ,
          'nonpolar':  4 ,
          'polar'   :  2.5  ,
          'cys'     :  3 }

    for elem in code:
        line='color '+code[elem]+','+elem+'&'+selection
        cmd.do (line)
        if scale:
            print(elem)
            cmd.set('sphere_scale', code2[elem], elem+'&'+selection)
    cmd.hide ('everything','resn HOH')

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

def display_graphkernel(pdb_id, pdb_path):
    '''
    '''
    # Load PDB
    cmd.bg_color('white')
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
    #cmd.set('sphere_scale', 2, pdb_id2 + ' and name ca')

    data  = cmd.get_coords(selection=pdb_id + ' and name ca', state=1)

    j = 55
    cmd.set('dash_width', 1.0)
    cmd.set('dash_color', 'marine')
    for i in range(len(data)):
        cmd.distance('d'+str(i)+str(j), pdb_id2 + ' and name ca and res ' + str(i), pdb_id2+ ' and name ca and res ' + str(j))
        cmd.hide('labels', 'd'+str(i)+str(j))
    resicolor(pdb_id)

    resicolor(pdb_id)
    resicolor(pdb_id2, True)

def display_graphpool(pdb_id, pdb_path):
    '''
    '''
    # Load PDB
    cmd.bg_color('white')
    cmd.load(pdb_path+pdb_id[:-2]+'.pdb')
    cmd.split_chains(pdb_id[:-2])
    for name in cmd.get_names('objects', 0, '(all)'):
        if not name.endswith(pdb_id[-1].upper()) and name.startswith(pdb_id[:4]):
            cmd.delete(name)
        else: zero_residues(name)
    cmd.reset()

    pdb_id2 = pdb_id+'copy'
    cmd.create(pdb_id2, pdb_id)

    cmd.color('grey', pdb_id)
    cmd.hide('everything',pdb_id)
    cmd.show_as('lines', pdb_id + ' and (name ca)')
    cmd.set('line_width', 5)
    cmd.set_bond('line_width', 5,  pdb_id + ' and (name ca)')
    cmd.show('spheres', pdb_id + ' and name ca')
    cmd.set('sphere_transparency', 0.0, pdb_id + ' and name ca')
    cmd.set('sphere_scale', 0.5, pdb_id + ' and name ca')
    data  = cmd.get_coords(selection=pdb_id + ' and name ca', state=1)
    data_ = []
    cmd.set('dash_color', 'marine')
    cmd.set('dash_width', 1.0)
    j = 55
    for i in range(len(data)):
        if i % 2 == 0 and i+1 <len(data):
            data[i] = np.mean(data[i:i+2],axis=0)
            data[i+1] = np.array([10000,10000,10000])

        #cmd.distance('d'+str(i)+str(j), pdb_id + ' and name ca and res ' + str(i), pdb_id + ' and name ca and res ' + str(j))
        #cmd.hide('labels', 'd'+str(i)+str(j))

    cmd.color('red', pdb_id2)
    cmd.hide('everything',pdb_id2)
    cmd.show_as('lines', pdb_id2 + ' and (name ca)')
    cmd.set('line_width', 5)
    cmd.set_bond('line_width', 5,  pdb_id2 + ' and (name ca)')
    cmd.show('spheres', pdb_id2 + ' and name ca')
    cmd.set('sphere_transparency', 0.0, pdb_id2 + ' and name ca')
    cmd.set('sphere_scale', 0.5, pdb_id2 + ' and name ca')
    for i in range(len(data)):
        if i % 2 == 0 and i+1 <len(data):
            cmd.alter_state(1,pdb_id2 + ' and name ca and res ' + str(i),'(x,y,z)='+str(tuple(data[i])))
        else:
            cmd.hide('spheres', pdb_id2 + ' and name ca and res ' + str(i))

    pdb_id3 = pdb_id+'copy2'
    cmd.create(pdb_id3, pdb_id2)
    cmd.color('red', pdb_id3)
    cmd.hide('everything',pdb_id3)
    cmd.show_as('lines', pdb_id3 + ' and (name ca)')
    cmd.set('line_width', 5)
    cmd.set_bond('line_width', 5,  pdb_id3 + ' and (name ca)')
    cmd.show('spheres', pdb_id3 + ' and name ca')
    cmd.set('sphere_transparency', 0.8, pdb_id3 + ' and name ca')
    for i in range(len(data)):
        if i % 2 == 0 and i+1 <len(data):
            cmd.set('sphere_scale', np.random.uniform(1.0, 4.5), pdb_id3 + ' and name ca and res ' + str(i))
        else:
            cmd.hide('spheres', pdb_id2 + ' and name ca and res ' + str(i))




def display_graphconv(pdb_id, pdb_path):
    '''
    '''
    cmd.bg_color('white')
    cmd.load(pdb_path+pdb_id[:-2]+'.pdb')
    cmd.split_chains(pdb_id[:-2])
    for name in cmd.get_names('objects', 0, '(all)'):
        if not name.endswith(pdb_id[-1].upper()) and name.startswith(pdb_id[:4]):
            cmd.delete(name)
        else: zero_residues(name)
    cmd.reset()

    pdb_id2 = pdb_id+'copy'
    cmd.create(pdb_id2, pdb_id)

    cmd.color('white', pdb_id)
    cmd.hide('everything',pdb_id)
    cmd.show('spheres', pdb_id + ' and name ca')
    cmd.set('sphere_transparency', 0.0, pdb_id + ' and name ca')
    cmd.set('sphere_scale', 0.5, pdb_id + ' and name ca')
    data  = cmd.get_coords(selection=pdb_id + ' and name ca', state=1)



    j = 55
    cmd.set('dash_width', 1.0)
    cmd.set('dash_color', 'marine')
    data = np.random.uniform(0,0.25,len(data))
    for i in range(len(data)):
        cmd.distance('d'+str(i)+str(j), pdb_id + ' and name ca and res ' + str(i), pdb_id + ' and name ca and res ' + str(j))
        cmd.hide('labels', 'd'+str(i)+str(j))
        cmd.select('toBecolored', pdb_id + ' and name ca and res ' + str(i))
        if i == j: cmd.set_color('saliency'+str(i)+pdb_id, list(cmap(norm(1)))[:3])
        else: cmd.set_color('saliency'+str(i)+pdb_id, list(cmap(norm(data[i])))[:3])
        cmd.color('saliency'+str(i)+pdb_id, 'toBecolored')


    cmd.color('white', pdb_id2)
    cmd.hide('everything',pdb_id2)
    cmd.show('spheres', pdb_id2 + ' and name ca')
    cmd.set('sphere_transparency', 0.0, pdb_id2 + ' and name ca')
    cmd.set('sphere_scale', 0.5, pdb_id2 + ' and name ca')

    j = 55
    cmd.set('dash_width', 1.0)
    cmd.set('dash_color', 'marine')
    #data = np.random.uniform(0,0.25,len(data))
    for i in range(len(data)):
        cmd.distance('d'+str(i)+str(j), pdb_id2 + ' and name ca and res ' + str(i), pdb_id2+ ' and name ca and res ' + str(j))
        cmd.hide('labels', 'd'+str(i)+str(j))
        cmd.select('toBecolored', pdb_id2 + ' and name ca and res ' + str(i))
        if i == j: cmd.set_color('saliency'+str(i)+pdb_id2, list(cmap(norm(0)))[:3])
        else: cmd.set_color('saliency'+str(i)+pdb_id2, list(cmap(norm(data[i])))[:3])
        cmd.color('saliency'+str(i)+pdb_id2, 'toBecolored')

################################################################################

# Paths
pdb_path = '../../data/KrasHras/pdb/'

# Load Attribution
pdb_id = '4epr_a'

cmd.reinitialize()
#display_graphkernel(pdb_id, pdb_path)
display_graphpool(pdb_id, pdb_path)
#display_graphconv(pdb_id, pdb_path)
display_filter()
