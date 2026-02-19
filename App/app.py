import os, sys
import traceback

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-1]))
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))
from Cad_VLM.models.text2cad import Text2CAD
from CadSeqProc.utility.macro import MAX_CAD_SEQUENCE_LENGTH, N_BIT
from CadSeqProc.cad_sequence import CADSequence
import gradio as gr
import yaml
import torch


def load_model(config, device):
    # -------------------------------- Load Model -------------------------------- #
    cad_config = config["cad_decoder"]
    cad_config["cad_seq_len"] = MAX_CAD_SEQUENCE_LENGTH
    text2cad = Text2CAD(text_config=config["text_encoder"], cad_config=cad_config).to(
        device
    )

    if config["test"]["checkpoint_path"] is not None:
        checkpoint_file = config["test"]["checkpoint_path"]

        checkpoint = torch.load(checkpoint_file, map_location=device)
        pretrained_dict = {}
        for key, value in checkpoint["model_state_dict"].items():
            if key.split(".")[0] == "module":
                pretrained_dict[".".join(key.split(".")[1:])] = value
            else:
                pretrained_dict[key] = value

        text2cad.load_state_dict(pretrained_dict, strict=False)
    text2cad.eval()
    return text2cad

def test_model(model, text, config, device, max_retries=4):
    import traceback
    
    if not isinstance(text, list):
        text = [text]

    # The Auto-Retry Loop: Try up to 4 times to get a valid OCC geometry
    for attempt in range(max_retries):
        print(f"\n--- [Attempt {attempt + 1}/{max_retries}] Generating geometry for: '{text[0]}' ---")
        
        try:
            # 1. Ask the AI to predict the tokens (using Top-P sampling for variance)
            pred_cad_seq_dict = model.test_decode(
                texts=text,
                maxlen=MAX_CAD_SEQUENCE_LENGTH,
                nucleus_prob=0.9,  # Critical for retries: ensures we get a slightly different sequence each time
                topk_index=10,
                device=device,
            )
        except Exception as e:
            print(f"[test_model] Error during model decoding: {e}")
            if attempt == max_retries - 1:
                return None, str(e)
            continue # Try again

        try:
            # 2. Hand the tokens to OpenCASCADE to build the B-Rep and Mesh
            pred_cad = CADSequence.from_vec(
                pred_cad_seq_dict["cad_vec"][0].cpu().numpy(),
                bit=N_BIT,
                post_processing=True,
            ).create_mesh()

            # If we make it to this line, OpenCASCADE accepted the topology!
            print(f"✅ Attempt {attempt + 1} succeeded! Valid topology generated.")
            return pred_cad.mesh, pred_cad
            
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ Attempt {attempt + 1} failed. OpenCASCADE rejected the geometry: {error_msg}")
            
            # If this was our last attempt, log the failure and give up
            if attempt == max_retries - 1:
                print(f"❌ All {max_retries} attempts failed to produce a valid B-Rep.")
                traceback.print_exc()
                return None, error_msg
            
            # Otherwise, the loop automatically restarts for the next attempt

def parse_config_file(config_file):
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data



config_path = "../Cad_VLM/config/inference_user_input.yaml"
config = parse_config_file(config_path)
device = torch.device("cpu")  # MPS produces NaN in the CAD decoder
model = load_model(config, device)
OUTPUT_DIR="output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def genrate_cad_model_from_text(text):
    global model, config
    result = test_model(model=model, text=text, config=config, device=device)
    mesh, extra = result[0], result[1]
    if mesh is not None:
        cad_seq = extra  # CADSequence object with .cad_model set
        output_path = os.path.join(OUTPUT_DIR, "output.stl")
        mesh.export(output_path)
        # Export STEP file
        step_path = os.path.join(OUTPUT_DIR, "output.step")
        cad_seq.save_stp(filename="output", output_folder=OUTPUT_DIR, type="step")
        print(f"STEP file saved to: {os.path.abspath(step_path)}")
        return output_path, step_path
    else:
        error_detail = extra if isinstance(extra, str) else "Unknown error"
        raise gr.Error(f"Could not generate a valid CAD model. Detail: {error_detail}. Try a simpler prompt like 'A ring.' or 'A rectangular prism.'")


examples = [
    "A ring.",
    "A rectangular prism.",
    "A 3D star shape with 5 points.",
    "The CAD model features a cylindrical object with a cylindrical hole in the center.",
    "The CAD model features a rectangular metal plate with four holes along its length."
]

title = "Text2CAD: Generating Sequential CAD Designs from Beginner-to-Expert Level Text Prompts"
description = """
Generate 3D CAD models from text prompts of varying complexity, from beginner-level descriptions to expert-level specifications.

**Note:** Generation takes ~1-2 minutes on Apple Silicon. Simple geometric prompts work best.

<div style="display: flex; justify-content: center; gap: 10px; align-items: center;">

<a href="https://arxiv.org/abs/2409.17106">
  <img src="https://img.shields.io/badge/Arxiv-3498db?style=for-the-badge&logoWidth=40&logoColor=white&labelColor=2c3e50&borderRadius=10" alt="Arxiv" />
</a>
<a href="https://sadilkhan.github.io/text2cad-project/">
  <img src="https://img.shields.io/badge/Project-2ecc71?style=for-the-badge&logoWidth=40&logoColor=white&labelColor=27ae60&borderRadius=10" alt="Project" />
</a>
<a href="https://huggingface.co/datasets/SadilKhan/Text2CAD">
  <img src="https://img.shields.io/badge/Dataset-7D5BA6?style=for-the-badge&logoWidth=40&logoColor=white&labelColor=27ae60&borderRadius=10" alt="Dataset" />
</a>

</div>
"""

# Create the Gradio interface
demo = gr.Interface(
    fn=genrate_cad_model_from_text,
    inputs=gr.Textbox(label="Text", placeholder="Enter a text prompt here"),
    outputs=[
        gr.Model3D(clear_color=[0.678, 0.847, 0.902, 1.0], label="3D CAD Model"),
        gr.File(label="Download STEP File"),
    ],
    examples=examples,
    title=title,
    description=description,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.queue().launch(share=True)
