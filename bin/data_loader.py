import torch
import pandas as pd

def load_data(data_path = "./data/data.csv", set_length = 72):
    df = pd.read_csv(data_path)

    df = df[df["len"] == set_length]
    df = df[df["has_nonstd_aa"] == False]
    df = df.drop(columns=["pdb_id", "chain_code", "sst3", "len", "has_nonstd_aa"])

    AminoAcids = sorted(list(set("".join(df["seq"]))))
    SecondaryStructures = sorted(list(set("".join(df["sst8"]))))

    aa_stoi = {s: i for i, s in enumerate(AminoAcids, start=1)}
    aa_itos = {i: s for i, s in enumerate(AminoAcids, start=1)}

    aa_encode = lambda s: [aa_stoi[c] for c in s]
    aa_decode = lambda l: "".join([aa_itos[i] for i in l])

    ss_stoi = {s: i for i, s in enumerate(SecondaryStructures)}
    ss_itos = {i: s for i, s in enumerate(SecondaryStructures)}

    PAD_LABEL = -100

    ss_encode = lambda s: [ss_stoi[c] for c in s]
    ss_decode = lambda l: "".join([ss_itos[i] for i in l if i != PAD_LABEL])

    X = torch.stack([
        torch.tensor(aa_encode(seq), dtype=torch.long)
        for seq in df["seq"]
    ])

    y = torch.stack([
        torch.tensor(ss_encode(sst8), dtype=torch.long)
        for sst8 in df["sst8"]
    ])
    
    return X, y, AminoAcids, SecondaryStructures, aa_stoi, aa_itos, aa_encode, aa_decode, ss_stoi, ss_itos, ss_encode, ss_decode, PAD_LABEL, set_length
