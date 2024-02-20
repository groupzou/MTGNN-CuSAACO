print('Start!')

from tqdm import tqdm
from pathlib import Path
import lmdb
import errno
import pickle
import numpy as np
import pandas as pd
import torch
import torch_geometric
import torch.nn.functional as F
from torch.utils.data import Dataset
from ase.io import read, vasp
from pymatgen.io.ase import AseAtomsAdaptor
from ocpmodels.datasets import SinglePointLmdbDataset


""" 

input file structure: 
    - /{lmdb_dir}
        - /csv
            - {csv_file}.csv
        - /POSCAR (required when generating structures using POSCAR/CONTCAR files)
            - {element}
                - {no}.vasp
        - /CONCAR (required when generating structures using POSCAR/CONTCAR files 
                   & judging fixed atoms based on CONTCAR information)
            - {element}
                - cont_{no}.vasp
        - /vasp_data (required when generating structures using OUTCAR files)
            - {element}
                - {no}
                    - OUTCAR 

output files:
    - /{lmdb_dir}
        - outcar_content.txt (when generating structures using OUTCAR files)
        - init_data_list.pt (when generating structures using OUTCAR files)
        - relax_data_list.pt (when generating structures using OUTCAR files)
        - pos-contcar_content.txt (when generating structures using POSCAR/CONTCAR files)
        - init_data_list-pos.pt (when generating structures using POSCAR/CONTCAR files)
        - relax_data_list-cont.pt (when generating structures using POSCAR/CONTCAR files)
        - {task}
            - train{name}
                - data.lmdb
                - data.lmdb-lock
            - val{name}
                - data.lmdb
                - data.lmdb-lock
            - test{name}
                - data.lmdb
                - data.lmdb-lock
            - split_idx_for_csv{name}.npz
            - train{name}_norm.npz (when normalization information output is needed)
            
            {name} == {seed} when datasets are generated based on outcar files 
                      & fixing atoms are determined only based on their coordinates in the z-axis
            {name} == _newtags{seed} when datasets are generated based on outcar files 
                      & fixing atoms are dependent on contcar information
            {name} == _poscont{seed} when datasets are generated based on poscar & contcar files 
                      & fixing atoms are determined only based on their coordinates in the z-axis
            {name} == _poscont_newtags{seed} when datasets are generated based on poscar & contcar files 
                      & fixing atoms are dependent on contcar information

"""


# doping_lmdb_dir = 'fill in the directory of your OUTCAR/CONTCAR/POSCAR here ' \
#                   '(not including "OUTCAR"/"CONTCAR"/"POSCAR")'
# doping_lmdb_dir = r'../data/is2re/doping'
# doping_lmdb_dir = r'G:/ocp-data/raw_data'
doping_lmdb_dir = r'/mnt/e/raw_data'
# doping_lmdb_dir = r'/Volumes/One Touch/ocp-data/raw_data'

prime_data_dire = doping_lmdb_dir + r'/csv'
# csv_file = 'fill in your .csv file name here'
csv_file = 'main_coord_expnFalse_full'


def get_data(energy_filter=False):
    """ especially for the SAA dataset (for CO adsorption energy prediction);
    may not be useful for your dataset """

    file_name = csv_file
    data = pd.read_csv(prime_data_dire + f'/{file_name}.csv')
    e_mask = (data['element'] != 'C') & (data['element'] != 'O')
    if energy_filter:
        e_mask = e_mask & (data['E'] < 2.5) & (data['E'] > -5.0)
    data = data[e_mask]
    return data


def struct_surface_dict():
    """ especially for the SAA dataset (constructed based on 5 Cu surfaces);
    may not be useful for your dataset """

    return {3: 100, 4: 100, 5: 100, 6: 111, 7: 111, 8: 111, 9: 111,
            18: 110, 19: 110, 20: 110, 21: 110, 31: 210, 32: 210, 33: 210,
            34: 210, 35: 210, 36: 210, 37: 210, 38: 210, 39: 210, 40: 210,
            41: 210, 42: 210, 43: 210, 44: 210, 45: 210, 46: 210, 47: 210,
            48: 210, 49: 210, 50: 210, 51: 210, 52: 210, 53: 210, 54: 210,
            55: 210, 56: 210, 57: 210, 59: 411, 60: 411, 61: 411, 62: 411,
            63: 411, 64: 411, 65: 411, 67: 411, 68: 411, 69: 411, 70: 411,
            71: 411, 72: 411, 73: 411, 74: 411, 76: 411, 77: 411, 78: 411,
            79: 411, 80: 411, 82: 411, 83: 411, 84: 411, 85: 411, 86: 411,
            87: 411, 89: 100, 90: 100, 91: 100, 93: 111, 94: 111, 95: 111,
            96: 111, 98: 110, 99: 110, 100: 110, 101: 110}, \
           {100: 0, 111: 1, 110: 2, 210: 3, 411: 4}


def get_top_from_atoms(atoms):
    """ 0 - subsurface, 1 - surface, 2 - adsorbate;
    especially for catalytic performance prediction tasks """

    atom_idx = np.arange(atoms.get_global_number_of_atoms())
    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    co_mask = (atomic_numbers == 6) | (atomic_numbers == 8)
    co_idx = atom_idx[co_mask]

    crystal = positions[~co_mask]
    crystal_idx = np.arange(len(crystal))
    top = np.max(crystal[:, 2])
    bottom = np.min(crystal[:, 2])
    # when constructing the surfaces, we piled three layers of atoms on the z-axis
    thickness = np.true_divide((top - bottom), 3)
    sub_surf_h = top - thickness

    top_atoms_mask = crystal[:, 2] >= sub_surf_h  # get the top layer
    top_idx = crystal_idx[top_atoms_mask]

    tags = atoms.get_tags()
    tags[co_idx] = 2
    tags[top_idx] = 1
    return tags


def get_top_from_contcar(struct_no, element, data_dire):
    """ 0 - subsurface, 1 - surface, 2 - adsorbate;
    especially for catalytic performance prediction tasks """

    from ase.io.vasp import read_vasp
    poscar = read_vasp(f'{data_dire}/{element}/cont_{struct_no}.vasp')
    tags = np.ones_like(poscar.numbers)
    # depended on how the .vasp files are saved for your dataset

    try:
        fixed_ids = poscar.constraints[0].index
    except IndexError:
        raise Exception(f'The {element} {struct_no} contcar add no constraints on atoms.')

    atom_idx = np.arange(poscar.get_global_number_of_atoms())
    atomic_numbers = poscar.numbers
    co_mask = (atomic_numbers == 6) | (atomic_numbers == 8)
    co_idx = atom_idx[co_mask]

    tags[co_idx] = 2
    tags[fixed_ids] = 0
    return tags


class Doping2ASE:
    def __init__(self, rcut, max_nbr, task_type, tag_from_cont, use_pos_and_cont):
        assert task_type in ['i2i', 'r2i', 'r2r', 'combined']

        self._load_doping_dire()

        self.data_type = task_type

        """ following ocp scheme of building crystal graphs (6 properties) """
        self.radius = rcut
        self.max_neigh = max_nbr
        self.r_forces = True
        self.r_distances = True
        self.r_edges = True
        self.r_fixed = True

        self.tag_from_cont = tag_from_cont

        self._get_data()
        self.dop_species = self.df.loc[:, ['No.', 'element', 'E']]

        self.use_pos_and_cont = use_pos_and_cont
        self._load_outcar()
        self._load_pos_and_cont()
        if self.use_pos_and_cont:
            self.struct_content_name = 'pos-contcar_content.txt'
        else:
            self.struct_content_name = 'outcar_content.txt'

        self.atoms_collection = self._outcar2ase(save=True, load_cache=True)
        self.surf_dict, self.surf_sid = struct_surface_dict()
        self._load_size()

        return

    def _load_doping_dire(self):
        self.doping_lmdb_dir = doping_lmdb_dir
        return

    def _get_data(self):
        self.df = get_data(energy_filter=False)
        return

    def _load_outcar(self):
        self.outcar_path = Path(self.doping_lmdb_dir + r'/vasp_data')
        return

    def _load_pos_and_cont(self):
        self.poscar_path = Path(self.doping_lmdb_dir + r'/POSCAR')
        self.concar_path = Path(self.doping_lmdb_dir + r'/CONTCAR')
        return

    def _outcar2ase(self, save=False, load_cache=True):
        """ Extract structure information from OUTCARs / POSCARs & CONTCARs"""

        outcar_content_dire = Path(self.doping_lmdb_dir) / self.struct_content_name
        if load_cache:
            if outcar_content_dire.exists():
                with open(outcar_content_dire, 'rb') as f:
                    return pickle.load(f)

        atom_cluster = []
        for i in range(len(self.dop_species)):
            struct_no, element, e = self.dop_species.iloc[i]['No.'], \
                                    self.dop_species.iloc[i]['element'], \
                                    self.dop_species.iloc[i]['E']
            print(f'Dealing with: {element}, {struct_no}')
            if not self.use_pos_and_cont:
                cluster_init = vasp.read_vasp_out(
                    self.outcar_path / str(element) / str(struct_no) / 'OUTCAR', index=0)
                # cluster_init.pbc = [1, 1, 0]
                cluster_relax = vasp.read_vasp_out(
                    self.outcar_path / str(element) / str(struct_no) / 'OUTCAR', index=-1)
                # cluster_relax.pbc = [1, 1, 0]
            else:
                cluster_init = vasp.read_vasp(
                    self.poscar_path / str(element) / f'{str(struct_no)}.vasp')
                # cluster_init.pbc = [1, 1, 0]
                cluster_relax = vasp.read_vasp(
                    self.concar_path / str(element) / f'cont_{str(struct_no)}.vasp')
                # cluster_relax.pbc = [1, 1, 0]
            # print(f'Finish {element} {struct_no}')
            atom_cluster.append([cluster_init, cluster_relax])
        print('Finish Structure Analysis!')

        if save:
            with open(outcar_content_dire, 'wb') as f:
                pickle.dump(atom_cluster, f, protocol=-1)

        return atom_cluster

    def _get_neighbors_pymatgen(self, atoms):
        """
        Source: https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/preprocessing/atoms_to_graphs.py
        """

        struct = AseAtomsAdaptor.get_structure(atoms)
        _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
            r=self.radius, numerical_tol=0, exclude_self=True
        )

        _nonmax_idx = []
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            # sort neighbors by distance, remove edges larger than max_neighbors
            idx_sorted = np.argsort(n_distance[idx_i])[: self.max_neigh]
            _nonmax_idx.append(idx_i[idx_sorted])
        _nonmax_idx = np.concatenate(_nonmax_idx)

        _c_index = _c_index[_nonmax_idx]
        _n_index = _n_index[_nonmax_idx]
        n_distance = n_distance[_nonmax_idx]
        _offsets = _offsets[_nonmax_idx]

        return _c_index, _n_index, n_distance, _offsets

    @staticmethod
    def _reshape_features(c_index, n_index, n_distance, offsets):
        """
        Source: https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/preprocessing/atoms_to_graphs.py
        """

        edge_index = torch.LongTensor(np.vstack((n_index, c_index)))
        edge_distances = torch.FloatTensor(n_distance)
        cell_offsets = torch.LongTensor(offsets)

        # remove distances smaller than a tolerance ~ 0. The small tolerance is
        # needed to correct for pymatgen's neighbor_list returning self atoms
        # in a few edge cases.
        nonzero = torch.where(edge_distances >= 1e-8)[0]
        edge_index = edge_index[:, nonzero]
        edge_distances = edge_distances[nonzero]
        cell_offsets = cell_offsets[nonzero]

        return edge_index, edge_distances, cell_offsets

    def direct_convert(self, atoms):
        """
        Source: https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/preprocessing/atoms_to_graphs.py
        """

        atomic_numbers = torch.Tensor(atoms.get_atomic_numbers())
        positions = torch.Tensor(atoms.get_positions())
        cell = torch.Tensor(atoms.get_cell()).view(1, 3, 3)
        natoms = int(positions.shape[0])

        data = torch_geometric.data.Data(
            cell=cell,
            pos=positions,
            atomic_numbers=atomic_numbers,
            natoms=natoms)
        try:
            data.y = atoms.get_potential_energy(apply_constraint=True)
        except RuntimeError:
            data.y = 9999  # we cannot obtain force related information from pos/contcar

        data.tags = torch.LongTensor(get_top_from_atoms(atoms))

        # optionally include other properties
        if self.r_edges:
            # run internal functions to get padded indices and distances
            split_idx_dist = self._get_neighbors_pymatgen(atoms)
            edge_index, edge_distances, cell_offsets = self._reshape_features(
                *split_idx_dist
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            if self.r_distances:
                data.distances = edge_distances
        if self.r_forces:
            try:
                forces = torch.Tensor(atoms.get_forces(apply_constraint=True))
                data.force = forces
            except RuntimeError:
                data.force = 9999
        if self.r_fixed:
            fixed_idx = torch.zeros(natoms)
            if hasattr(atoms, "constraints"):
                from ase.constraints import FixAtoms

                for constraint in atoms.constraints:
                    if isinstance(constraint, FixAtoms):
                        fixed_idx[constraint.index] = 1
            data.fixed = fixed_idx

        return data

    def direct_convert_all(self, atoms_collection, save=False, load_cache=True):

        init_data_dire = Path(self.doping_lmdb_dir) / 'init_data_list.pt'
        relax_data_dire = Path(self.doping_lmdb_dir) / 'relax_data_list.pt'
        if self.use_pos_and_cont:
            init_data_dire = Path(self.doping_lmdb_dir) / 'init_data_list-pos.pt'
            relax_data_dire = Path(self.doping_lmdb_dir) / 'relax_data_list-cont.pt'
        if load_cache:
            if init_data_dire.exists() & relax_data_dire.exists():
                return torch.load(init_data_dire), \
                       torch.load(relax_data_dire)

        init_data_list = []
        relax_data_list = []
        for atoms in tqdm(atoms_collection,
                          desc="converting ASE atoms collection into graphs",
                          total=len(atoms_collection),
                          unit=" systems"
                          ):
            init_data_list.append(self.direct_convert(atoms[0]))
            relax_data_list.append(self.direct_convert(atoms[1]))

        if save:
            torch.save(init_data_list, init_data_dire)
            torch.save(relax_data_list, relax_data_dire)

        return init_data_list, relax_data_list

    def _ase_info_aggr(self, data, i, init_data_list, relax_data_list):

        struct_no, element, e = self.dop_species.iloc[i]['No.'], \
                                self.dop_species.iloc[i]['element'], \
                                self.dop_species.iloc[i]['E']

        absolute_init_energy = init_data_list[i].y
        absolute_relax_energy = relax_data_list[i].y
        reference_energy = absolute_relax_energy - e
        e_init = absolute_init_energy - reference_energy

        data.y = e_init
        data.y_relaxed = e

        data.sfid = self.surf_sid[self.surf_dict[struct_no]]  # surface index
        data.sid = i  # system/cluster index

        if self.tag_from_cont:
            data.tags = torch.from_numpy(
                get_top_from_contcar(struct_no, element, data_dire=self.concar_path))
        return data

    def _combine_ase_data(self, i, init_data_list, relax_data_list):
        combined_data = init_data_list[i]
        combined_data = self._ase_info_aggr(combined_data, i, init_data_list, relax_data_list)
        combined_data.pos_relaxed = relax_data_list[i].pos
        combined_data.force_relaxed = relax_data_list[i].force

        # get relaxed distances based on edges in init structures
        init_src_node, init_dst_node = combined_data.edge_index
        relax_src_pos, relax_dst_pos = combined_data.pos_relaxed[init_src_node], \
                                       combined_data.pos_relaxed[init_dst_node]
        distances_relaxed = torch.sqrt(F.relu(torch.sum((relax_src_pos - relax_dst_pos) ** 2, -1)))
        combined_data.distances_relaxed = distances_relaxed
        # print(f'Finish {i}-th sample!')
        return combined_data

    def _single_task_init_ase_data(self, i, init_data_list, relax_data_list):
        st_data = init_data_list[i]
        st_data = self._ase_info_aggr(st_data, i, init_data_list, relax_data_list)
        # print(f'Finish {i}-th sample!')
        return st_data

    def _single_task_relax_ase_data(self, i, init_data_list, relax_data_list):
        st_data = relax_data_list[i]
        st_data = self._ase_info_aggr(st_data, i, init_data_list, relax_data_list)
        # print(f'Finish {i}-th sample!')
        return st_data

    def _load_size(self):
        self.tr_size, self.val_size, self.te_size = 0.0003, 0.0001, 0.0001
        return

    def outcar2lmdb(self, building_type, idx_list, lmdb_data_name,
                    init_data_list, relax_data_list, size=0.0001):

        db = lmdb.open(
            lmdb_data_name + '/' + "data.lmdb",
            map_size=1099511627776 * size,
            subdir=False,
            meminit=False,
            map_async=True,
        )

        data_list = []
        for i, idx in enumerate(
                tqdm(idx_list, desc="saving graphs as a lmdb file", total=len(idx_list), unit="systems")):
            if building_type == 'combined':
                data = self._combine_ase_data(idx, init_data_list, relax_data_list)
            elif building_type == 'init':
                data = self._single_task_init_ase_data(idx, init_data_list, relax_data_list)
            elif building_type == 'relaxed':
                data = self._single_task_relax_ase_data(idx, init_data_list, relax_data_list)
            else:
                raise Exception('Data type options: combined, init, relaxed')
            data_list.append(data)

            # Write to LMDB
            txn = db.begin(write=True)
            txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))
            txn.commit()
            db.sync()
        db.close()
        return data_list

    @staticmethod
    def _get_label_mean_std(lmdb_dataset):
        label_list = [data.y_relaxed for data in lmdb_dataset]
        target_mean, target_std = np.mean(label_list), np.std(label_list)
        print(f'target_mean: {target_mean}')
        print(f'target_std: {target_std}')
        return target_mean, target_std

    def _data_dir(self, random_seed):
        train_dire = self.doping_lmdb_dir + '/' + self.data_type + '/' + "train"
        val_dire = self.doping_lmdb_dir + '/' + self.data_type + '/' + "val"
        test_dire = self.doping_lmdb_dir + '/' + self.data_type + '/' + "test"
        slice_dire = self.doping_lmdb_dir + '/' + self.data_type + '/' + "split_idx_for_csv"
        if self.use_pos_and_cont:
            train_dire, val_dire, test_dire, slice_dire = \
                train_dire + '_poscont', val_dire + '_poscont', \
                test_dire + '_poscont', slice_dire + '_poscont'
        if self.tag_from_cont:
            train_dire, val_dire, test_dire, slice_dire = \
                train_dire + '_newtags', val_dire + '_newtags', \
                test_dire + '_newtags', slice_dire + '_newtags'

        if random_seed is not None:
            train_dire += str(random_seed)
            val_dire += str(random_seed)
            test_dire += str(random_seed)
            slice_dire += str(random_seed)
        return train_dire, val_dire, test_dire, slice_dire

    def _split(self, random_seed):
        num_rows = len(self.dop_species)
        full_id = np.arange(num_rows)
        np.random.seed(random_seed)
        np.random.shuffle(full_id)
        train_num = int(0.6 * num_rows)
        val_num = (num_rows - train_num) // 2
        train_ids = full_id[:train_num]
        val_ids = full_id[train_num:train_num + val_num]
        test_ids = full_id[train_num + val_num:]
        return train_ids, val_ids, test_ids

    def __call__(self, random_seed=None, load_cache=True, normalization=True):

        train_dire, val_dire, test_dire, slice_dire = self._data_dir(random_seed=random_seed)

        if load_cache:
            if ((Path(train_dire + '/data.lmdb').exists() &
                 Path(val_dire + '/data.lmdb').exists() &
                 Path(test_dire + '/data.lmdb').exists() &
                 Path(slice_dire + '.npz').exists()) & ~normalization):
                return SinglePointLmdbDataset({"src": str(train_dire + '/data.lmdb')}), \
                       SinglePointLmdbDataset({"src": str(val_dire + '/data.lmdb')}), \
                       SinglePointLmdbDataset({"src": str(test_dire + '/data.lmdb')}), \
                       np.load(slice_dire + '.npz')
            elif ((Path(train_dire + '/data.lmdb').exists() &
                   Path(val_dire + '/data.lmdb').exists() &
                   Path(test_dire + '/data.lmdb').exists() &
                   Path(slice_dire + '.npz').exists()) & normalization):
                train_dataset = SinglePointLmdbDataset({"src": str(train_dire + '/data.lmdb')})
                target_mean, target_std = self._get_label_mean_std(train_dataset)
                np.savez(train_dire + '_norm.npz', target_mean=target_mean, target_std=target_std)
                return train_dataset, \
                       SinglePointLmdbDataset({"src": str(val_dire + '/data.lmdb')}), \
                       SinglePointLmdbDataset({"src": str(test_dire + '/data.lmdb')}), \
                       np.load(slice_dire + '.npz'), \
                       target_mean, target_std

        init_data_list, relax_data_list = self.direct_convert_all(
            self.atoms_collection, save=True, load_cache=load_cache)

        train_ids, val_ids, test_ids = self._split(random_seed=random_seed)

        Path(train_dire).mkdir(parents=True, exist_ok=True)
        Path(val_dire).mkdir(parents=True, exist_ok=True)
        Path(test_dire).mkdir(parents=True, exist_ok=True)
        if self.data_type == 'i2i':
            tr_data = self.outcar2lmdb('init', train_ids, train_dire, init_data_list, relax_data_list,
                                       size=self.tr_size)
            val_data = self.outcar2lmdb('init', val_ids, val_dire, init_data_list, relax_data_list,
                                        size=self.val_size)
            te_data = self.outcar2lmdb('init', test_ids, test_dire, init_data_list, relax_data_list,
                                       size=self.te_size)
        elif self.data_type == 'r2i':
            tr_data = self.outcar2lmdb('relaxed', train_ids, train_dire, init_data_list, relax_data_list,
                                       size=self.tr_size)
            val_data = self.outcar2lmdb('init', val_ids, val_dire, init_data_list, relax_data_list,
                                        size=self.val_size)
            te_data = self.outcar2lmdb('init', test_ids, test_dire, init_data_list, relax_data_list,
                                       size=self.te_size)
        elif self.data_type == 'r2r':
            tr_data = self.outcar2lmdb('relaxed', train_ids, train_dire, init_data_list, relax_data_list,
                                       size=self.tr_size)
            val_data = self.outcar2lmdb('relaxed', val_ids, val_dire, init_data_list, relax_data_list,
                                        size=self.val_size)
            te_data = self.outcar2lmdb('relaxed', test_ids, test_dire, init_data_list, relax_data_list,
                                       size=self.te_size)
        else:
            tr_data = self.outcar2lmdb('combined', train_ids, train_dire, init_data_list, relax_data_list,
                                       size=self.tr_size)
            val_data = self.outcar2lmdb('combined', val_ids, val_dire, init_data_list, relax_data_list,
                                        size=self.val_size)
            te_data = self.outcar2lmdb('combined', test_ids, test_dire, init_data_list, relax_data_list,
                                       size=self.te_size)

        np.savez(slice_dire + '.npz', train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)
        print('End')
        if normalization:
            target_mean, target_std = self._get_label_mean_std(tr_data)
            np.savez(train_dire + '_norm.npz', target_mean=target_mean, target_std=target_std)
            return tr_data, val_data, te_data, np.load(slice_dire + '.npz'), target_mean, target_std
        else:
            return tr_data, val_data, te_data, np.load(slice_dire + '.npz')


if __name__ == '__main__':
    for i in range(10):
        dop1 = Doping2ASE(rcut=4.0, max_nbr=50, task_type='combined',
                          tag_from_cont=True, use_pos_and_cont=False)
        dop2 = Doping2ASE(rcut=4.0, max_nbr=50, task_type='combined',
                          tag_from_cont=True, use_pos_and_cont=True)
        tr_1, val_1, te_1, slc1, t_m1, t_s1 = dop1(
            random_seed=i, load_cache=True, normalization=True)
        tr_2, val_2, te_2, slc2, t_m2, t_s2 = dop2(
            random_seed=i, load_cache=True, normalization=True)
