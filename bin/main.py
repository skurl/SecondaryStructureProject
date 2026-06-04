import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from data_loader import load_data
from train import train_model
from model import Transformer

<<<<<<< HEAD
<<<<<<< HEAD
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # when running locally with newer PyTorch versions with torch.accelerator support

# device = "cuda" if torch.cuda.is_available() else "cpu" # alternative for older PyTorch versions without torch.accelerator, for use on the cluster 

print(f"Using: {device} \n")

X, y, AminoAcids, SecondaryStructures, aa_stoi, aa_itos, aa_encode, aa_decode, ss_stoi, ss_itos, ss_encode, ss_decode, set_length = load_data(set_length=72)
=======
=======
>>>>>>> 26dfcf78dc672b2c1db45bf12b1651805314ce44
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # when running locally with newer PyTorch versions with torch.accelerator support

device = "cuda" if torch.cuda.is_available() else "cpu" # alternative for older PyTorch versions without torch.accelerator, for use on the cluster 

print(f"Using: {device} \n")

X, y, AminoAcids, SecondaryStructures, aa_stoi, aa_itos, aa_encode, aa_decode, ss_stoi, ss_itos, ss_encode, ss_decode, PAD_LABEL, set_length = load_data(set_length=72)
<<<<<<< HEAD
>>>>>>> 26dfcf78dc672b2c1db45bf12b1651805314ce44
=======
>>>>>>> 26dfcf78dc672b2c1db45bf12b1651805314ce44
dataset = TensorDataset(X, y)

traindata, valdata, testdata = torch.utils.data.random_split(dataset, [0.80, 0.10, 0.10])

print(f"Train size: {len(traindata)}, Val size: {len(valdata)}, Test size: {len(testdata)} \n")

train_loader = DataLoader(traindata, batch_size=32, shuffle=True)
val_loader = DataLoader(valdata, batch_size=32, shuffle=False)
test_loader = DataLoader(testdata, batch_size=32, shuffle=False)

src_vocab_size = len(AminoAcids) + 1
<<<<<<< HEAD
<<<<<<< HEAD
tgt_vocab_size = len(SecondaryStructures)
d_model = 64
num_heads = 2
num_layers = 3
d_ff = 4*d_model
max_seq_length = set_length
dropout = 0.1
=======
=======
>>>>>>> 26dfcf78dc672b2c1db45bf12b1651805314ce44
tgt_vocab_size = len(SecondaryStructures) + 1
d_model = 24
num_heads = 4
num_layers = 1
d_ff = 4*d_model
max_seq_length = set_length
dropout = 0.2
<<<<<<< HEAD
>>>>>>> 26dfcf78dc672b2c1db45bf12b1651805314ce44
=======
>>>>>>> 26dfcf78dc672b2c1db45bf12b1651805314ce44

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

transformer = train_model(
    model=transformer,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    tgt_vocab_size=tgt_vocab_size,
<<<<<<< HEAD
<<<<<<< HEAD
    num_epochs=300,
    learning_rate=1e-4,
=======
    num_epochs=100,
    learning_rate=0.0001,
>>>>>>> 26dfcf78dc672b2c1db45bf12b1651805314ce44
=======
    num_epochs=100,
    learning_rate=0.0001,
>>>>>>> 26dfcf78dc672b2c1db45bf12b1651805314ce44
    save_path="./outputs/weights.pth"
    # save_path=save_path
)

from testing import test_model

criterion = nn.CrossEntropyLoss(ignore_index=-100)

test_loss, test_accuracy = test_model(
    model=transformer,
    test_loader=test_loader,
    device=device,
    tgt_vocab_size=tgt_vocab_size,
    criterion=criterion
)