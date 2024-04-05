# MTGNN-CuSAACO

MTGNN is a graph neural network framework for predicting chemical and physical properties of materials with relaxed structures, with their unrelaxed structure counterparts as input. An [arXiv](https://arxiv.org/abs/2209.07300) article has been published to illustrate architectures of the models and their performances in predicting Cu-based single-atom alloys (SAAs) catalysts for the CO2 reduction reaction (CO2RR), and ```IS2RE``` task in ```open catalyst project (ocp)```.

This project incorporates implementation of MT DimeNet++ architecture (including MT-MD DimeNet++ and MT-MLP DimeNet++), a trainer script to train the model, and a dataset generation script, in which a method of constructing .lmdb datasets based on [VASP](https://www.vasp.at/) output files (OUTCARs, or POSCARs + CONTCARs) is provided.

Addtionally, dopant information-modified GNN models illustrated in the Dicussion section of the [arXiv](https://arxiv.org/abs/2209.07300) article have also attached.

The project is implemented based on [open catalyst project](https://github.com/Open-Catalyst-Project/ocp).

## Installation

Please complete the following steps to install the package:

- Install the python environment of ```ocp```, following the [instruction](https://github.com/Open-Catalyst-Project/ocp/blob/main/INSTALL.md).
- Put the ```dpp-csmd.py```, ```dpp-csmlp.py``` and ```mt_utils.py``` scripts in this project into this directory: ```ocpmodels/models```.
- Put the ```mt_energy_trainer.py``` script in this project into this directory: ```ocpmodels/trainers```.
- Modify the ```trainer``` variable in the ```new_trainer_context``` method in the ```ocpmodels/common/utils.py``` script as:
```
        trainer = trainer_cls(
            task=config["task"],
            model=config["model"],
            dataset=config["dataset"],
            optimizer=config["optim"],
            identifier=config["identifier"],

            mdn=config["mdn"],
            alpha=config["alpha"],

            timestamp_id=config.get("timestamp_id", None),
            run_dir=config.get("run_dir", "./"),
            is_debug=config.get("is_debug", False),
            print_every=config.get("print_every", 10),
            seed=config.get("seed", 0),
            logger=config.get("logger", "tensorboard"),
            local_rank=config["local_rank"],
            amp=config.get("amp", False),
            cpu=config.get("cpu", False),
            slurm=config.get("slurm", {}),
            noddp=config.get("noddp", False),
        )

```
Then, the installation procedure is finished.

To train the model, run the following command in the main folder of ```ocp```:
```
python3 main.py --config-yml configs/0_newtags/dpp-csmd.yml --mode train --seed 0 --identifier dpp-csmd_0_newtags
```
Sample .yml files are provided in ```configs``` directory of this project. Choose the name of the model as ```dimenetpluspluscsmd``` to train MT-MD DimeNet++, which is implemented in ```dpp-csmd.py```, ```dimenetpluspluscsmlp``` to train MT-MLP DimeNet++ in ```dpp-csmlp.py```.

## Dataset generation

```struct2dataset.py``` is provided for users to generate custom datasets based on VASP output files for the package, which is also applicable for other models in the ```ocp``` environment. Node-, edge- and graph-level features in the generated graph data are introduced in the [ocp tutorial](https://github.com/Open-Catalyst-Project/ocp/blob/main/tutorials/data_preprocessing.ipynb).

Example datasets, which have been applied in articles ([1](https://arxiv.org/abs/2209.07300), [2](https://arxiv.org/abs/2303.02875)), are available in [this link](https://doi.org/10.5281/zenodo.10679574).

To generate datasets, path of input source data should be arranged in the following structure:
```
- /{lmdb_dir}
        - /csv
                - {csv_file}.csv
        - /POSCAR (required when generating structures using POSCAR/CONTCAR files)
                - {element}
                        - {no}.vasp
        - /CONCAR (required when generating structures using POSCAR/CONTCAR files & judging fixed atoms based on CONTCAR information)
                - {element}
                        - cont_{no}.vasp
        - /vasp_data (required when generating structures using OUTCAR files)
                - {element}
                        - {no}
                                - OUTCAR 
```
in which ```{lmdb_dir}``` represents the directory where source data is saved, as well as where .lmdb datasets are generated.

The VASP data files are seperated into ```{element}``` catagories, including materials related to that element, and ```{no}``` denotes a specific material of corresponding index.
- In the SAA example, ```{element}``` indicate the doping element species of the material, and therefore, various SAAs doped by the same element are saved in the same catagory. If there is not a clear relationship between some specific element with materials in your dataset, all materials can be saved in one catagory with an arbitary name for the ```{element}``` file name, so that the dataset can still be generated without modifying the code.

The ```{csv_file}.csv``` file saves index of materials, element related to each material, and the desired property of the relaxed material.
- In the implementation, the three catagories are saved with the column names as 'No.', 'element', 'E', respectively.

After running the code, dataset will be generated and saved in the following directory structure:
```
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
```
The ```outcar_content.txt``` and ```pos-contcar_content.txt``` save coordinates information derived from source data. ```init_data_list(-pos).pt``` and ```relax_data_list(-cont).pt``` save graph datasets with the same features defined in ```ocp```, in which ```pos``` of each atom is unrelaxed and relaxed one, respectively. The files are needed for generating .lmdb datasets saved in ```{task}``` directory, and they will not be generated again (and directly used for generation) if they are detected to be located in ```{lmdb_dir}``` directory.

The variable ```{task}``` indicates types of the dataset, with possible values as ```I2I```, ```R2I```, ```R2R``` and ```combined```. Introduction of the tasks are illustrated in "Benchmarks on basic adsorption energy prediction tasks" section in the [arXiv](https://arxiv.org/abs/2209.07300) article, and ```combined``` dataset is for the ```MT I2I``` task, incorporating both unrelaxed and relaxed structures.

The variable ```{name}``` differentiates datasets generated with different random seeds, VASP source data. Besides, the scripts also provides a function to judge whether an atom is fixed based on the CONTCAR file of the material, the information of which is saved in ```data.fixed``` (implemented in ```get_top_from_contcar```). If not, an atom is regarded as fixed if the z-axis of the coordinate of the atom is smaller than 2/3 of the thickness of the substrate (highest z-coordinate - lowest z-coordinate, implemented in ```get_top_from_atoms```).
- The implementation is based on CO-adsorbed surface materials. Therefore, C and O atoms are ignored when deciding the thickness. It is possible to modify the implementation for constructing other custom datasets.

Possible values of ```{name}``` are listed as follows:
```
{name} == {seed} when datasets are generated based on outcar files 
      & fixing atoms are determined only based on their coordinates in the z-axis
{name} == _newtags{seed} when datasets are generated based on outcar files 
      & fixing atoms are dependent on contcar information
{name} == _poscont{seed} when datasets are generated based on poscar & contcar files 
      & fixing atoms are determined only based on their coordinates in the z-axis
{name} == _poscont_newtags{seed} when datasets are generated based on poscar & contcar files 
      & fixing atoms are dependent on contcar information
```

## Citing

If you find this project useful for your research, please consider to cite:
```
Multi-Task Mixture Density Graph Neural Networks for Predicting Cu-based Single-Atom Alloy Catalysts for CO2 Reduction
Chen Liang#, Bowen Wang#, Shaogang Hao, Guangyong Chen*, Pheng-Ann Heng, Xiaolong Zou*
arXiv preprint arXiv:2209.07300, 2022. https://arxiv.org/abs/2209.07300
```
