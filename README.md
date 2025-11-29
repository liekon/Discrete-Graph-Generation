# DMol
## Environment installation
This code was tested with PyTorch 2.0.1, cuda 11.8 and torch_geometrics 2.3.1

  - Download anaconda/miniconda if needed
  - Create a rdkit environment that directly contains rdkit:
    
    ```conda create -c conda-forge -n dmol rdkit=2023.03.2 python=3.9```
  - `conda activate dmol`
  - Check that this line does not return an error:
    
    ``` python3 -c 'from rdkit import Chem' ```
  - Install graph-tool (https://graph-tool.skewed.de/): 
    
    ```conda install -c conda-forge graph-tool=2.45```
  - Check that this line does not return an error:
    
    ```python3 -c 'import graph_tool as gt' ```
  - Install the nvcc drivers for your cuda version. For example:
    
    ```conda install -c "nvidia/label/cuda-11.8.0" cuda```
  - Install a corresponding version of pytorch, for example: 
    
    ```pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118```
  - Install other packages using the requirement file: 
    
    ```pip install -r requirements.txt```

  - Navigate to the ./src/analysis/orca directory and compile orca.cpp: 
    
     ```g++ -O2 -std=c++11 -o orca orca.cpp```

Note: graph_tool and torch_geometric currently seem to conflict on MacOS, I have not solved this issue yet.

## Run the code
  
  - All code is currently launched through `python3 main.py`. Check hydra documentation (https://hydra.cc/) for overriding default parameters.
  
  - To run the discrete model with the default QM9 configuration:
    
    ```bash
    python3 main.py
    ```
  - To select a different dataset, override the `dataset.name` parameter with Hydra. For example, to use Guacamol:
    
    ```bash
    python3 main.py dataset=guacamol
    ```
    See the YAML files under `configs/dataset` for the list of supported datasets and their options.

## Prepare datasets and ring compression configuration

Before running any experiment, you must:

- **1. Download raw datasets**
  - **QM9**  
    - The code will automatically download from the PyG/DeepChem URLs if files are missing (see `QM9Dataset.raw_url` and `QM9Dataset.raw_url2` in `src/datasets/qm9_dataset.py`):  
      - `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip`  
      - `https://ndownloader.figshare.com/files/3195404`  
    - If you prefer manual download, place the extracted files under  
      `data/qm9/raw/` (e.g. `gdb9.sdf`, `gdb9.sdf.csv`, `uncharacterized.txt`, `train.csv`, `val.csv`, `test.csv`).
  - **MOSES**  
    - Automatic download URLs are defined in `MOSESDataset` (`src/datasets/moses_dataset.py`):  
      - Train: `https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv`  
      - Test:  `https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv`  
      - Scaffolds: `https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv`  
    - To download manually, save them as `train_moses.csv`, `val_moses.csv`, `test_moses.csv` under  
      `data/moses/raw/`.
  - **Guacamol**  
    - Download URLs are defined in `GuacamolDataset` (`src/datasets/guacamol_dataset.py`):  
      - Train: `https://figshare.com/ndownloader/files/13612760`  
      - Test:  `https://figshare.com/ndownloader/files/13612757`  
      - Valid: `https://figshare.com/ndownloader/files/13612766`  
      - (All: `https://figshare.com/ndownloader/files/13612745`, used only if you need the full set)  
    - Save them as `guacamol_v1_train.smiles`, `guacamol_v1_valid.smiles`, `guacamol_v1_test.smiles` under  
      `data/guacamol/raw/`.
  - Make sure the folder structure matches what is expected in the corresponding dataset classes under `src/datasets/`.

- **2. Collect ring statistics (optional but recommended)**
  - Use `ring_statistics.py` at the project root to analyze which ring structures are most frequent in your dataset:
    
    ```bash
    # Example: compute ring statistics for a SMILES file
    python3 ring_statistics.py data/qm9/raw/qm9_train.smiles
    ```
  - The script will output a text file (e.g. `*_ring_stats.txt`) containing all observed rings and their frequencies.
  - Inspect this file and decide which ring SMILES you want to compress into supernodes (e.g. `C1CCC1`, `C1CC1`, `N1CCC1`).

- **3. Configure ring compression in the dataset config**
  - Open the corresponding YAML file under `configs/dataset/`, for example:
    - QM9: `configs/dataset/qm9.yaml`
    - MOSES: `configs/dataset/moses.yaml`
    - Guacamol: `configs/dataset/guacamol.yaml`
  - In each of these files there is a **ring compression section** with the keys:
    - `ring_types`: list of ring SMILES that will be compressed into supernodes.
    - `ring_weights`: list of molecular weights (one per entry in `ring_types`), used to compute total molecular mass after compression.
    - `statistics_after_4` (or `statistics_after_compression`): path to the JSON file where compressed-dataset statistics are stored.
  - Replace the default `ring_types` and `ring_weights` with the ring structures you selected from the statistics file, making sure the lengths of both lists match.
  - After changing these parameters, re-run the dataset preprocessing (by running `python3 main.py` once) so that the compressed graphs and updated statistics are recomputed.

In summary, the full workflow is:

1. Download the raw dataset files into `data/<dataset_name>/raw/`.
2. (Optional but recommended) Run `ring_statistics.py` on a SMILES file from that dataset to obtain ring frequency statistics.
3. Choose a small set of ring SMILES to compress and put them into the `ring_types` and `ring_weights` fields of the corresponding YAML config under `configs/dataset/`.
4. Launch experiments with `python3 main.py` (and override `dataset=...` if you want a dataset other than the default QM9).