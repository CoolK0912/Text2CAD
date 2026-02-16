"""Compare CPU vs MPS output - dump first-step logits."""
import os, sys
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-1]))
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))

import torch
import yaml
import numpy as np
from Cad_VLM.models.text2cad import Text2CAD
from Cad_VLM.models.layers import text_embed, embedder
from Cad_VLM.models.decoder import CADDecoder, CADDecoderLayer
from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH, END_TOKEN
from CadSeqProc.utility.utils import generate_attention_mask

def parse_config_file(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)

def override_device(device_str):
    """Patch all hardcoded MPS references to use a specific device."""
    # Patch TextEmbedder
    orig_te_init = text_embed.TextEmbedder.__init__
    def new_te_init(self, model_name, cache_dir, max_seq_len):
        super(text_embed.TextEmbedder, self).__init__()
        self.device = device_str
        self.max_seq_len = max_seq_len
        self.model_name = text_embed.MODEL_NAME_DICT.get(model_name, "bert_large_uncased")
        self.tokenizer = text_embed.BertTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)
        self.model = text_embed.BertModel.from_pretrained(
            self.model_name, cache_dir=cache_dir, max_position_embeddings=max_seq_len
        ).to(self.device)
    text_embed.TextEmbedder.__init__ = new_te_init

    # Patch PositionalEncodingSinCos
    orig_pe_init = embedder.PositionalEncodingSinCos.__init__
    def new_pe_init(self, embedding_size, max_seq_len, device):
        orig_pe_init(self, embedding_size, max_seq_len, device)
        self.device = torch.device(device_str)
    embedder.PositionalEncodingSinCos.__init__ = new_pe_init

    # Patch CADDecoder.from_config
    def new_from_config(config):
        from CadSeqProc.utility.macro import CAD_CLASS_INFO
        return CADDecoder(
            cad_class_info=CAD_CLASS_INFO,
            tdim=config["tdim"], cdim=config["cdim"],
            num_layers=config["num_layers"], num_heads=config["num_heads"],
            dropout=config["dropout"], ca_level_start=config["ca_level_start"],
            device=device_str,
        )
    CADDecoder.from_config = staticmethod(new_from_config)

def load_and_test(device_str, text="A ring."):
    config = parse_config_file("../Cad_VLM/config/inference_user_input.yaml")
    device = torch.device(device_str)
    print(f"\n{'='*60}")
    print(f"Testing on: {device_str}")
    print(f"{'='*60}")

    override_device(device_str)

    cad_config = config["cad_decoder"]
    cad_config["cad_seq_len"] = MAX_CAD_SEQUENCE_LENGTH
    model = Text2CAD(text_config=config["text_encoder"], cad_config=cad_config).to(device)

    checkpoint = torch.load(config["test"]["checkpoint_path"], map_location=device)
    pretrained_dict = {}
    for key, value in checkpoint["model_state_dict"].items():
        if key.split(".")[0] == "module":
            pretrained_dict[".".join(key.split(".")[1:])] = value
        else:
            pretrained_dict[key] = value

    result = model.load_state_dict(pretrained_dict, strict=False)
    # Only show non-BERT missing keys
    non_bert_missing = [k for k in result.missing_keys if 'base_text_embedder' not in k]
    print(f"Non-BERT missing keys: {non_bert_missing}")
    model.eval()

    # Manual first decode step to inspect logits
    with torch.no_grad():
        ZE, key_padding_mask = model.base_text_embedder.get_embedding([text])
        print(f"ZE shape: {ZE.shape}, device: {ZE.device}")
        print(f"ZE stats: min={ZE.min():.4f}, max={ZE.max():.4f}, mean={ZE.mean():.4f}")

        ca_mask = {
            "attn_mask": text_embed.prepare_cross_attention_mask_batch(key_padding_mask, cad_seq_len=1),
            "key_padding_mask": key_padding_mask
        }

        ZE_adapted, _ = model.adaptive_layer(ZE, {
            "attn_mask": None,
            "key_padding_mask": ca_mask["key_padding_mask"],
        }, False)
        print(f"ZE_adapted stats: min={ZE_adapted.min():.4f}, max={ZE_adapted.max():.4f}, mean={ZE_adapted.mean():.4f}")

        # First token
        new_cad_seq_dict = {
            "cad_vec": torch.tensor([[[1, 0]]]).to(device),
            "flag_vec": torch.zeros(1, 1).int().to(device),
            "index_vec": torch.zeros(1, 1).int().to(device),
        }

        cad_mask = {
            "attn_mask": ca_mask["attn_mask"].repeat(1, 1, 1),
            "key_padding_mask": ca_mask["key_padding_mask"],
        }

        cad_pred, _ = model.cad_decoder(
            new_cad_seq_dict,
            ZE_adapted,
            {
                "attn_mask": generate_attention_mask(1, 1, device=device),
                "key_padding_mask": (new_cad_seq_dict["cad_vec"] == 0),
            },
            cad_mask,
            False,
        )

        print(f"\ncad_pred shape: {cad_pred.shape}")
        print(f"cad_pred stats: min={cad_pred.min():.4f}, max={cad_pred.max():.4f}")

        # First token predictions (x and y)
        logits_x = cad_pred[0, 0, 0]  # (267,)
        logits_y = cad_pred[0, 0, 1]  # (267,)
        print(f"\nX logits top-5: values={torch.topk(logits_x, 5).values.cpu().numpy()}, indices={torch.topk(logits_x, 5).indices.cpu().numpy()}")
        print(f"Y logits top-5: values={torch.topk(logits_y, 5).values.cpu().numpy()}, indices={torch.topk(logits_y, 5).indices.cpu().numpy()}")
        print(f"X argmax: {torch.argmax(logits_x).item()}, Y argmax: {torch.argmax(logits_y).item()}")

        # For top-k sampling with topk_index=1, the code does:
        # torch.topk(cad_pred, topk_index, dim=-1).indices[:, t-1:t, :, -1]
        topk_result = torch.topk(cad_pred, 1, dim=-1)
        new_token = topk_result.indices[:, 0:1, :, -1]
        print(f"First predicted token (topk_index=1): {new_token.cpu().numpy()}")

    # Now run full decode (short)
    with torch.no_grad():
        pred = model.test_decode(
            texts=[text], maxlen=30,
            nucleus_prob=0, topk_index=1, device=device,
        )

    cad_vec = pred["cad_vec"][0].cpu().numpy()
    print(f"\nFull decode first 30: {cad_vec[:, 0]}")
    non_pad = np.sum(cad_vec[:, 0] != 0)
    print(f"Non-padding tokens: {non_pad}")
    return cad_vec

# Test on CPU
cpu_vec = load_and_test("cpu")

# Test on MPS
if torch.backends.mps.is_available():
    mps_vec = load_and_test("mps")
