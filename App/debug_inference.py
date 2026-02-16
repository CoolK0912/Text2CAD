"""Debug script to inspect raw model output and find why CAD generation fails."""
import os, sys
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-1]))
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))

import torch
import yaml
import numpy as np
from Cad_VLM.models.text2cad import Text2CAD
from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH, N_BIT, END_TOKEN
from CadSeqProc.cad_sequence import CADSequence
from CadSeqProc.utility.utils import split_array

def parse_config_file(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)

config_path = "../Cad_VLM/config/inference_user_input.yaml"
config = parse_config_file(config_path)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}")

# Load model
cad_config = config["cad_decoder"]
cad_config["cad_seq_len"] = MAX_CAD_SEQUENCE_LENGTH
model = Text2CAD(text_config=config["text_encoder"], cad_config=cad_config).to(device)

checkpoint_file = config["test"]["checkpoint_path"]
print(f"Loading checkpoint: {checkpoint_file}")
checkpoint = torch.load(checkpoint_file, map_location=device)

# Check what keys are in the checkpoint
ckpt_keys = set(checkpoint["model_state_dict"].keys())
print(f"\nCheckpoint has {len(ckpt_keys)} keys")

pretrained_dict = {}
for key, value in checkpoint["model_state_dict"].items():
    if key.split(".")[0] == "module":
        pretrained_dict[".".join(key.split(".")[1:])] = value
    else:
        pretrained_dict[key] = value

# Check for key mismatches
model_keys = set(model.state_dict().keys())
missing = model_keys - set(pretrained_dict.keys())
unexpected = set(pretrained_dict.keys()) - model_keys
print(f"Missing keys (in model but not checkpoint): {missing}")
print(f"Unexpected keys (in checkpoint but not model): {unexpected}")

load_result = model.load_state_dict(pretrained_dict, strict=False)
print(f"Load result - missing: {load_result.missing_keys}")
print(f"Load result - unexpected: {load_result.unexpected_keys}")

model.eval()

# Run inference
text = "A ring."
print(f"\n{'='*60}")
print(f"Testing with: '{text}'")
print(f"{'='*60}")

with torch.no_grad():
    pred = model.test_decode(
        texts=[text],
        maxlen=MAX_CAD_SEQUENCE_LENGTH,
        nucleus_prob=0,
        topk_index=1,
        device=device,
    )

cad_vec = pred["cad_vec"][0].cpu().numpy()
print(f"\nRaw cad_vec shape: {cad_vec.shape}")
print(f"cad_vec[:20]:\n{cad_vec[:20]}")
print(f"\nUnique first-column values: {np.unique(cad_vec[:, 0])}")
print(f"Value counts (col 0):")
unique, counts = np.unique(cad_vec[:, 0], return_counts=True)
for v, c in zip(unique, counts):
    token_name = END_TOKEN[v] if v < len(END_TOKEN) else f"data({v})"
    print(f"  {v} ({token_name}): {c}")

# Check for end tokens
print(f"\nEND_TOKEN mapping: {dict(enumerate(END_TOKEN))}")

# Try from_vec processing
print(f"\n{'='*60}")
print("Attempting CADSequence.from_vec...")
try:
    cad_seq = CADSequence.from_vec(
        cad_vec,
        bit=N_BIT,
        post_processing=True,
    )
    print(f"from_vec succeeded!")
    print(f"Number of sketches: {len(cad_seq.sketch_seq)}")
    print(f"Number of extrusions: {len(cad_seq.extrude_seq)}")

    for i, (skt, ext) in enumerate(zip(cad_seq.sketch_seq, cad_seq.extrude_seq)):
        print(f"\n  Sketch {i}: {skt}")
        print(f"  Extrude {i} metadata: {ext.metadata}")

    print("\nAttempting create_cad_model...")
    cad_seq.create_cad_model()
    print("create_cad_model succeeded!")

    print("\nAttempting create_mesh...")
    cad_seq.create_mesh()
    print(f"create_mesh succeeded! Mesh: {cad_seq.mesh}")

except Exception as e:
    import traceback
    print(f"FAILED: {e}")
    traceback.print_exc()
