import os
import time
import torch

# Import your existing pipeline directly from app.py
from app import load_model, test_model, parse_config_file

def run_batch():
    config_path = "../Cad_VLM/config/inference_user_input.yaml"
    config = parse_config_file(config_path)
    device = torch.device("cpu")
    
    print("Loading Text2CAD Model...")
    model = load_model(config, device)
    
    # Create a dedicated folder for the batch run
    OUTPUT_DIR = "batch_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ---------------------------------------------------------
    # TEST BANK: Mix of semantic concepts and geometric operations
    # ---------------------------------------------------------
    prompts = [
        "A chair.",
        "A wooden chair consisting of four separate cylindrical legs, a flat square seat, and a tall rectangular backrest.",
        "A thick square base plate. Four solid cylindrical legs are joined to the bottom of the plate.",
        "A rectangular metal structural beam with three circular holes cut through the center.",
        "A solid cylindrical pipe with a thick flange on one end.",
        "A table with a circular top and a single cylindrical pillar for a base."
    ]
    
    print(f"Starting batch evaluation for {len(prompts)} prompts...\n")
    print("-" * 50)
    
    success_count = 0
    
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] Prompt: '{prompt}'")
        start_time = time.time()
        
        # Run the generation
        result = test_model(model=model, text=prompt, config=config, device=device)
        mesh, extra = result[0], result[1]
        
        if mesh is not None:
            # Create a clean filename from the prompt
            safe_name = "".join([c if c.isalnum() else "_" for c in prompt])[:25]
            output_path = os.path.join(OUTPUT_DIR, f"mesh_{i:02d}_{safe_name}.stl")
            
            # Save the valid B-Rep/Mesh
            mesh.export(output_path)
            elapsed = time.time() - start_time
            
            print(f"  ✅ SUCCESS: Saved to {output_path} ({elapsed:.1f}s)\n")
            success_count += 1
        else:
            print(f"  ❌ FAILED: {extra}\n")
            
    print("-" * 50)
    print(f"Batch complete! {success_count}/{len(prompts)} FEA-ready meshes generated successfully.")

if __name__ == "__main__":
    run_batch()