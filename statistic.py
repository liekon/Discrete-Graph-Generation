from rdkit import Chem
from collections import defaultdict
from tqdm import tqdm
def process_smiles_file(file_path):
    # 用于统计不同非H原子数量的分子个数
    heavy_atom_count = defaultdict(int)
    # 用于统计所有分子中各种键的出现次数
    bond_type_count = defaultdict(int)
    max_atoms = 0

    with open(file_path, 'r') as f:
        for line in tqdm(f):
            smi = line.strip()
            if not smi:
                continue  # 跳过空行
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"无法解析的 SMILES: {smi}")
                continue
            # 删除氢原子
            mol = Chem.RemoveHs(mol)
            
            # 获取非H原子数量
            n_atoms = mol.GetNumAtoms()
            heavy_atom_count[n_atoms] += 1
            if n_atoms > max_atoms:
                max_atoms = n_atoms

            # 累计键的种类和数量（将键类型转换为字符串作为字典的键）
            for bond in mol.GetBonds():
                bt = str(bond.GetBondType())
                bond_type_count[bt] += 1

    # 构造列表，索引i表示非H原子数量为i的分子数量
    heavy_atoms_array = [heavy_atom_count[i] for i in range(max_atoms + 1)]

    # 返回的结果数组，下标0为重原子数量的直方图，下标1为各键类型计数字典
    return [heavy_atoms_array, dict(bond_type_count)]

# 示例调用
if __name__ == "__main__":
    file_path = "/data/shared/vishal/junction_trees/icml18-jtnn/data/moses/train.smiles"
    results = process_smiles_file(file_path)
    print("非H原子数量直方图 (索引为原子数量，对应的值为分子数量):")
    print(results[0])
    print("\n各键类型在所有分子中的数量:")
    print(results[1])
