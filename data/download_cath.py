import os
import json
import argparse
from tqdm import tqdm
from Bio.PDB import PDBParser, PDBIO, Select


def get_pdb(pdb_code=""):
    os.system(f"wget -qnc -P cath_download/all/ https://files.rcsb.org/view/{pdb_code}.pdb")
    return f"cath_download/all/{pdb_code}.pdb"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download CATH dataset")
    parser.add_argument('--cath_version', required=True, type=str, help="CATH version")
    args = parser.parse_args()

    if args.cath_version == '4.2':
        with open('source/chain_set_splits_cath_4_2.json', 'r') as f:
            data = json.load(f)
    elif args.cath_version == '4.3':
        with open('source/chain_set_splits_cath_4_3.json', 'r') as f:
            data = json.load(f)
    else:
        raise ValueError("Invalid CATH version")

    if not os.path.exists('cath_download'):
        os.mkdir('cath_download')
        os.mkdir('cath_download/all')
        os.mkdir('cath_download/test')
        os.mkdir('cath_download/train')
        os.mkdir('cath_download/validation')

    exits_file = os.listdir('cath_download/all/')
    for key in data.keys():
        for pdb_code in data[key]:
            pdb_code = pdb_code[:4]
            if pdb_code + '.pdb' in exits_file:
                print(pdb_code, 'exist')
            else:
                get_pdb(pdb_code)
                print(pdb_code)

    err_file = []
    all_processed_file = os.listdir('cath_download/test/') + os.listdir('cath_download/train/') + os.listdir(
        'cath_download/validation/')
    for key in data.keys():
        if key not in ['cath_nodes']:
            for pdb_code in tqdm(data[key]):
                if pdb_code + '.pdb' not in all_processed_file:
                    pdb_file = f'cath_download/all/{pdb_code[:4]}' + '.pdb'
                    chain_id = pdb_code[5]

                    parser = PDBParser(QUIET=True)
                    try:
                        structure = parser.get_structure("name", pdb_file)

                        io = PDBIO()


                        class ChainSelector(Select):
                            def accept_chain(self, chain):
                                return chain.get_id() == chain_id

                            def accept_residue(self, residue):
                                return residue.id[0] == " "


                        io.set_structure(structure)
                        io.save(f"cath_download/{key}/{pdb_code[:4]}.{chain_id}.pdb", ChainSelector())
                    except FileNotFoundError:
                        err_file.append(pdb_code)

    print(err_file)
