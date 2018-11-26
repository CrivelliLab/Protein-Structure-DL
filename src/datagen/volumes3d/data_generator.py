'''
data_generator.py
Updated: 1/28/18

Script defines data_generator object which parses PDB files and generates 3D, 2D
and 1D representations of the proteins defined in the PDB file.

'''
import os
import scipy
import numpy as np
from itertools import product
from sklearn.decomposition import PCA

################################################################################

class data_generator(object):
    """
    Class creates a data generator which takes raw PDBs and creates volumetric
    representations of the atomic data.

    """
    # Hard Coded Values

    van_der_waal_radii = {  'H' : 1.2, 'C' : 1.7, 'N' : 1.55, 'O' : 1.52,
    'S' : 1.8, 'D' : 1.2, 'F' : 1.47, 'CL' : 1.75, 'BR' : 1.85, 'P' : 1.8,
    'I' : 1.98, 'E' : 1.0, 'X':1.0 , '': 0.0}
    # Source:https://physlab.lums.edu.pk/images/f/f6/Franck_ref2.pdf

    max_radius = 2.0

    def __init__(self, size=64, center=[0,0,0], resolution=1.0, thresh=1.0, nb_rots=0,
                 channels=None, seed=9999):
        # Random Seed
        self.seed = seed

        # Channels
        if channels: self.channels = channels
        else: print("Error: No Channels Defined."); exit()

        # Window Parameters
        self.size = size
        self.center = center
        self.resolution = resolution
        self.thresh = thresh

        # Window bounds and tolerances
        self.bounds = [ -(size * resolution)/2, -(size * resolution)/2,
                        -(size * resolution)/2, (size * resolution)/2,
                        (size * resolution)/2,  (size * resolution)/2]
        self.tolerance = int(self.max_radius/resolution)
        self.tolerance_perms = np.array([x for x in product(*[[z for z in
                range(-self.tolerance, self.tolerance+1)] for y in range(3)])])

        # Random Rotation Parameters
        self.nb_rots = nb_rots
        if self.nb_rots > 0:
            self.random_rotations = self.__gen_random_rotations(nb_rots)

    def generate_data(self, path, chain, rot, res_i=None, all_chains=False):
        '''
        Method generates 3D, 2D and 1D representations of PDB.

        '''
        # Parse PBD Atomic Data
        pdb_data = self.__parse_pdb(path, chain, res_i, all_chains)

        if len(pdb_data) == 0: return []

        # Apply Rotation To Data
        if rot > 0:
            pdb_data = self.__apply_rotation(pdb_data, self.random_rotations[rot-1])
        l1 = len(pdb_data)

        # Remove Outlier Atoms
        pdb_data = self.__remove_outlier_atoms(pdb_data)
        l2 = len(pdb_data)
        #if (l1-l2)/float(l1) > (1.0 - self.thresh): return []

        # Calculate Distances From Voxels
        data, distances, indexes_ = self.__calc_distances_from_voxels(pdb_data)
        del pdb_data

        # Calculate Indexes and Values of Occuppied Voxels
        indexes, values = self.__apply_channels(data, distances, indexes_)
        del distances, indexes_

        # Generate Representations
        array_3d = self.__generate_voxel_3d(indexes, values)
        #array_2d = self.__generate_voxel_2d(indexes, values, self.mapping)
        del indexes, values

        #array_1d = self.__generate_voxel_1d(array_3d)

        return [array_3d,]

    def __parse_pdb(self, path, chain, res_i, all_chains=False):
        '''
        Method parses atomic coordinate data from PDB. Coordinates are center
        around the centroid of the protein and then translated to the center
        coordinate defined for the DataGenerator.

        Params:
            path - str; PDB file path
            chain - str; chain identifier
            res_i - list[int]; list of residue indexes
            all_chains - boolean; whether all chains of PDB are used

        Returns:
            data - np.array; PDB atomic coordinate data

        '''
        residues = ['ALA', 'ARG', 'ASN', 'ASP', 'ASX', 'CYS', 'GLN',
                    'GLU', 'GLX', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
                    'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR',
                    'UNK', 'VAL']

        # Parse Coordinates
        data = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for row in lines:
                if row[:4] == 'ATOM' and row[21] == chain.upper():
                    if res_i != None:
                        if int(row[22:26]) in res_i:
                            if row[17:20] not in residues:
                                print("error")
                                ress = 'UNK'
                            else: ress = row[17:20]
                            parsed_data = [ress, row[12:16].strip(), self.van_der_waal_radii[row[77].strip()], row[30:38], row[38:46], row[47:54]]
                    else:
                        if row[17:20] not in residues:
                            print("error")
                            ress = 'UNK'
                        else: ress = row[17:20]
                        parsed_data = [row[17:20], row[12:16].strip(), self.van_der_waal_radii[row[77].strip()], row[30:38], row[38:46], row[47:54]]
                    data.append(parsed_data)
                elif row[:4] == 'ATOM' and all_chains:
                    if res_i != None:
                        if int(row[22:26]) in res_i:
                            if row[17:20] not in residues: ress = 'UNK'
                            else: ress = row[17:20]
                            parsed_data = [ress, row[12:16].strip(), self.van_der_waal_radii[row[77].strip()], row[30:38], row[38:46], row[47:54]]
                    else:
                        if row[17:20] not in residues:
                            print("error")
                            ress = 'UNK'
                        else: ress = row[17:20]
                        parsed_data = [row[17:20], row[12:16].strip(), self.van_der_waal_radii[row[77].strip()], row[30:38], row[38:46], row[47:54]]
                    data.append(parsed_data)

        data = np.array(data)
        if len(data) == 0:

            return []

        # Center Coordinates Around Centroid
        coords = data[:,3:].astype('float')
        centroid = np.mean(coords, axis=0)
        centered_coord = coords - centroid - self.center

        # Orient along prime axis
        pca = PCA(n_components=3)
        pca.fit(centered_coord)
        c = pca.components_[np.argmax(pca.explained_variance_)]
        angle = np.arctan(c[2]/np.sqrt((c[0]**2)+(c[1]**2)))
        axis = np.dot(np.array([[0,1],[-1,0]]), np.array([c[0],c[1]]))
        rot1 = self.__get_rotation_matrix([axis[0],axis[1],0],angle)
        if c[0] < 0 and c[1] < 0 or c[0] < 0 and c[1] > 0 :
            rot2 = self.__get_rotation_matrix([0,0,1], np.arctan(c[1]/c[0]) + np.pi)
        else: rot2 = self.__get_rotation_matrix([0,0,1], np.arctan(c[1]/c[0]))
        rot = np.dot(rot1, rot2)
        centered_coord = np.dot(centered_coord, rot)

        data = np.concatenate([data[:,:3], centered_coord], axis=1)

        del centroid, centered_coord, coords

        return data

    def __remove_outlier_atoms(self, data):
        '''
        Method removes atoms outside of the window defined by the size of the
        voxel map, center of window and resolution of window.

        Param:
            data - np.array; PDB atomic coordinates

        Return:
            data - np.array; PDB atomic coordinates with outlier removed

        '''
        # Get Atomic Coordinates
        coords = data[:,3:].astype('float')

        # Get Indexes of Outlier Atoms
        i = np.concatenate([np.where(coords[:,0] < self.bounds[0] - self.tolerance)[0],
                            np.where(coords[:,1] < self.bounds[1] - self.tolerance)[0],
                            np.where(coords[:,2] < self.bounds[2] - self.tolerance)[0],
                            np.where(coords[:,0] > self.bounds[3] + self.tolerance)[0],
                            np.where(coords[:,1] > self.bounds[4] + self.tolerance)[0],
                            np.where(coords[:,2] > self.bounds[5] + self.tolerance)[0]], axis=0)

        # Delete Outliers
        if len(i) > 0: data = np.delete(data, i, axis=0)

        del i

        return data

    def __calc_distances_from_voxels(self, data):
        '''
        Method calculates the distances from atoms to voxel centers for all atoms and
        voxels within the window.

        '''
        # Calculate Distances
        coords = data[:,3:].astype('float')
        nearest_voxels = np.rint(((coords - self.bounds[:3] + (self.resolution/2.0)) / self.resolution) - 1).astype('int')

        nearest_voxels_repeat = np.repeat(nearest_voxels, len(self.tolerance_perms), axis=0)
        tolerance_perms_repeat = np.tile(self.tolerance_perms, (len(nearest_voxels),1))
        del nearest_voxels

        nearest_voxels_with_tolerance = nearest_voxels_repeat + tolerance_perms_repeat
        del tolerance_perms_repeat, nearest_voxels_repeat

        nearest_voxels_coords = ((nearest_voxels_with_tolerance + (self.resolution/2.0)) * self.resolution) + self.bounds[:3]
        coords_repeat = np.repeat(coords, len(self.tolerance_perms), axis=0)
        del coords

        distances = np.linalg.norm(coords_repeat -  nearest_voxels_coords, axis=1)
        data = np.repeat(data, len(self.tolerance_perms), axis=0)

        del coords_repeat, nearest_voxels_coords

        # Get Outlier Indexes
        i = np.concatenate([np.where(np.min(nearest_voxels_with_tolerance, axis=1) < 0)[0],
            np.where(np.max(nearest_voxels_with_tolerance, axis=1) > self.size-1)[0]], axis=0)

        # Delete outlier indexes
        if len(i) > 0:
            data = np.delete(data, i, axis=0)
            distances = np.delete(distances, i, axis=0)
            nearest_voxels_with_tolerance = np.delete(nearest_voxels_with_tolerance, i, axis=0)

        del i

        return data, distances, nearest_voxels_with_tolerance

    def __apply_channels(self, data, distances, nearest_voxels_indexes):
        '''
        Method extracts channel information from data and returns voxel
        indexes with corresponding values.

        '''
        # Remove Voxels Outside Atom Radius
        i = np.where((distances-data[:,2].astype('float')) > 0)
        if len(i[0]) > 0:
            data = np.delete(data, i, axis=0)
            distances = np.delete(distances, i, axis=0)
            nearest_voxels_indexes = np.delete(nearest_voxels_indexes, i, axis=0)
        del i

        # Split Channels
        chans = np.zeros((len(nearest_voxels_indexes),1)).astype('int')
        for i in range(len(self.channels)):
            x = ['1'] + ['0' for j in range(i)]
            x = ''.join(x)
            indexes = self.channels[i](data)
            print(self.channels[i])
            print(indexes)
            chans[indexes] += int(x, 2)

        # Get Occupancy Values and Voxel Indexes
        values = np.ones(len(data))
        #values = np.exp((-4*(np.square(distances)))/np.square(data[:,2].astype('float')))
        voxel_indexes = np.concatenate([nearest_voxels_indexes, chans], axis=1)

        del data, distances, nearest_voxels_indexes

        return voxel_indexes, values

    def __generate_voxel_3d(self, voxel_indexes, values):
        '''
        Method generates 3D representation of data.

        Param:
            voxel_indexes - np.array ; indexes of occupied voxels
            values - np.array ; channel values of occupied voxels

        Returns:
            vox_3d - np.array; 3D representation of data, shape (N,N,N,C)

        '''
        vox_3d = np.zeros((self.size, self.size, self.size, len(self.channels)))
        for i in range(len(voxel_indexes)):
            ind = voxel_indexes[i,:3]
            chans = list(bin(voxel_indexes[i,3])[2:])
            for j in range(len(chans)):
                z = int(chans[j])
                if z == 1: vox_3d[ind[0],ind[1],ind[2],len(chans)-1 - j] = values[i]
        return vox_3d


    def __gen_random_rotations(self, nb_rot):
        '''
        Method generates random rotations by sampling hypersphere.

        For more information see this link, the last example of which inspired this
        approach:
            http://mathworld.wolfram.com/SpherePointPicking.html

        '''
        # For consistant random coordinate generation.
        np.random.seed(self.seed)

        # Sample rotations
        vector = np.random.randn(3, nb_rot)
        vector /= np.linalg.norm(vector, axis=0)
        coordinate_arry = np.transpose(vector, (1,0))

        # Convert to Rotation Matrix
        rotations = []
        for c in coordinate_arry:
            angle = np.arctan(c[2]/np.sqrt((c[0]**2)+(c[1]**2)))
            axis = np.dot(np.array([[0,1],[-1,0]]), np.array([c[0],c[1]]))
            rot1 = self.__get_rotation_matrix([axis[0],axis[1],0],angle)
            if c[0] < 0 and c[1] < 0 or c[0] < 0 and c[1] > 0 :
                rot2 = self.__get_rotation_matrix([0,0,1], np.arctan(c[1]/c[0]) + np.pi)
            else: rot2 = self.__get_rotation_matrix([0,0,1], np.arctan(c[1]/c[0]))
            rot = np.dot(rot1, rot2)
            rotations.append(rot)
        rotations = np.array(rotations)

        return(rotations)

    def __get_rotation_matrix(self, axis, theta):
        '''
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.

        Param:
            axis - list ; (x, y, z) axis coordinates
            theta - float ; angle of rotaion in radians

        Return:
            rotation_matrix - np.array

        '''
        axis = np.asarray(axis)
        axis = axis/np.sqrt(np.dot(axis, axis))
        a = np.cos(theta/2.0)
        b, c, d = -axis*np.sin(theta/2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

        rotation_matrix = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                                    [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                                    [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

        return rotation_matrix

    def __apply_rotation(self, data, rotation):
        '''
        Method applies rotation to pdb_data defined as list of rotation matricies.

        Param:
            data - np.array ; parsed pdb data
            rotation - np.array; rotation matrix

        Returns:
            rotated_data - np.array ; pdb data with rotation applied

        '''
        # Get Atomic Coordinates
        coords = data[:,3:].astype('float')

        # Apply Rotation
        coords = np.dot(coords, rotation)

        # Update with Rotated Coordinates
        rotated_data = np.concatenate([data[:,:3], coords],axis=1)

        return rotated_data
