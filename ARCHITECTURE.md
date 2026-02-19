# Text2CAD — Architecture Deep Dive

A comprehensive guide to every component in this repository: how text becomes a parametric 3D CAD model, what each file does, and how the pieces connect.

---

## Table of Contents

1. [End-to-End Pipeline](#end-to-end-pipeline)
2. [Repository Map](#repository-map)
3. [Model Architecture (Cad_VLM/)](#model-architecture-cad_vlm)
   - [Text Encoder](#text-encoder)
   - [CAD Decoder](#cad-decoder)
   - [Model Orchestration](#model-orchestration)
   - [Data Loading](#data-loading)
   - [Training and Inference Scripts](#training-and-inference-scripts)
4. [CAD Sequence Processing (CadSeqProc/)](#cad-sequence-processing-cadseqproc)
   - [Core: CADSequence](#core-cadsequence)
   - [Sketch Representation](#sketch-representation)
   - [Extrusion Representation](#extrusion-representation)
   - [Geometric Primitives](#geometric-primitives)
   - [OpenCASCADE Utilities](#opencascade-utilities)
   - [Dataset Preprocessing Scripts](#dataset-preprocessing-scripts)
   - [Constants and Utilities](#constants-and-utilities)
5. [Web Application (App/)](#web-application-app)
6. [Evaluation (Evaluation/)](#evaluation-evaluation)
7. [CAD Representation and Tokenization](#cad-representation-and-tokenization)
   - [Quantization Scheme](#quantization-scheme)
   - [Token Vocabulary](#token-vocabulary)
   - [Sequence Format](#sequence-format)
8. [Training Pipeline](#training-pipeline)
9. [Inference Pipeline](#inference-pipeline)
10. [Configuration Reference](#configuration-reference)
11. [Setup and Installation](#setup-and-installation)
12. [Running the System](#running-the-system)

---

## End-to-End Pipeline

Text2CAD converts a natural language description into a parametric 3D CAD model through three stages:

```
Text Prompt
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: Text Encoding                                 │
│                                                         │
│  BERT-large-uncased (frozen)                            │
│    └─► 1024-dim contextual embeddings                   │
│         └─► TextAdaptiveLayer (self-attention, FFN)     │
│              └─► Refined text features + attention mask  │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: Autoregressive CAD Token Generation           │
│                                                         │
│  CADDecoder (8-layer transformer)                       │
│    • Layers 1–2: self-attention only                    │
│    • Layers 3–8: self-attention + cross-attention       │
│                   to text features                      │
│    • Generates (x, y) token pairs one step at a time   │
│    • Vocabulary: 267 classes per coordinate             │
│      (256 quantized values + 7 structural + 4 boolean)  │
│    • Max sequence length: 272 tokens                    │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: B-Rep Reconstruction (OpenCASCADE)            │
│                                                         │
│  CADSequence.from_vec(tokens)                           │
│    └─► Parse into SketchSequence + ExtrudeSequence      │
│         └─► Build 2D wires (lines, arcs, circles)       │
│              └─► Extrude into 3D solids                  │
│                   └─► Boolean operations (join/cut/intersect) │
│                        └─► Final B-Rep solid             │
│                             └─► Export STEP / STL        │
└─────────────────────────────────────────────────────────┘
```

The key insight: CAD models are represented as **flat token sequences** — the same representation used by language models. This means a standard transformer architecture can learn to "speak CAD" the same way it learns to speak English.

---

## Repository Map

```
Text2CAD/
│
├── Cad_VLM/                        # The neural network
│   ├── models/
│   │   ├── text2cad.py             #   Top-level model: Text2CAD
│   │   ├── decoder.py              #   CADDecoder transformer
│   │   ├── loss.py                 #   Cross-entropy loss (x + y)
│   │   ├── metrics.py              #   Token accuracy with tolerance
│   │   ├── utils.py                #   Model utilities
│   │   └── layers/
│   │       ├── text_embed.py       #     BERT wrapper (TextEmbedder)
│   │       ├── adaptive_layer.py   #     Text refinement (TextAdaptiveLayer)
│   │       ├── embedder.py         #     CAD token embeddings + positional encoding
│   │       ├── attention.py        #     Multi-head and cross-attention
│   │       ├── functional.py       #     Feed-forward sublayer
│   │       └── utils_decode.py     #     Sampling, masks, mesh utilities
│   │
│   ├── config/
│   │   ├── trainer.yaml            #   Training hyperparameters
│   │   ├── inference.yaml          #   Batch test inference config
│   │   └── inference_user_input.yaml #   Interactive inference config
│   │
│   ├── dataprep/
│   │   └── t2c_dataset.py          #   Text2CAD_Dataset + DataLoader
│   │
│   ├── checkpoints/                #   Trained weights (Text2CAD_1.0.pth)
│   ├── train.py                    #   Training loop entry point
│   ├── test.py                     #   Batch inference on test set
│   └── test_user_input.py          #   Single-prompt inference
│
├── CadSeqProc/                     # CAD geometry engine
│   ├── cad_sequence.py             #   CADSequence: the central data structure
│   │
│   ├── sequence/
│   │   ├── sketch/
│   │   │   ├── sketchsequence.py   #     SketchSequence (2D profile)
│   │   │   ├── face.py             #     FaceSequence (bounded region)
│   │   │   ├── loop.py             #     Loop (closed wire of curves)
│   │   │   └── coord_system.py     #     CoordinateSystem (3D plane)
│   │   └── transformation/
│   │       └── extrude_sequence.py #     ExtrudeSequence (3D operation)
│   │
│   ├── geometry/
│   │   ├── line.py                 #     Line segment primitive
│   │   ├── arc.py                  #     Circular arc primitive
│   │   ├── circle.py               #     Full circle primitive
│   │   └── curve.py                #     Base curve class
│   │
│   ├── OCCUtils/                   #   OpenCASCADE wrapper library
│   │   ├── Topology.py             #     Shape traversal (Topo)
│   │   ├── Construct.py            #     B-Rep construction helpers
│   │   ├── Common.py               #     Bounding box, shared utilities
│   │   ├── edge.py, face.py, ...   #     Per-shape-type utilities
│   │   └── base.py                 #     Base classes
│   │
│   ├── utility/
│   │   ├── macro.py                #     Constants: N_BIT, tokens, vocab sizes
│   │   ├── utils.py                #     STEP I/O, mesh sampling, LR schedulers
│   │   ├── logger.py               #     Loguru logging config
│   │   └── decorator.py            #     Timing decorators
│   │
│   ├── json2vec.py                 #   Preprocessing: DeepCAD JSON → vectors
│   ├── json2step.py                #   Convert JSON → STEP files
│   ├── json2stl_skt3d.py           #   Convert JSON → STL meshes
│   ├── split_json.py               #   Train/val/test splitting
│   ├── merge_vlm_minimal.py        #   Merge VLM annotations
│   └── eda.py                      #   Exploratory data analysis
│
├── App/                            # Web interface
│   ├── app.py                      #   Gradio app (3D viewer + STEP download)
│   ├── batch_eval.py               #   Batch evaluation runner
│   ├── debug_cpu_vs_mps.py         #   Device comparison diagnostic
│   └── debug_inference.py          #   Step-by-step debug tool
│
├── Evaluation/
│   └── eval_seq.py                 #   Chamfer distance, curve F1, invalidity ratio
│
├── environment.yml                 #   Conda env (Linux/CUDA)
├── environment-mac.yml             #   Conda env (macOS/CPU)
└── LICENSE
```

---

## Model Architecture (Cad_VLM/)

### Text Encoder

The text encoder converts a natural language prompt into a dense feature representation the CAD decoder can attend to.

**TextEmbedder** (`models/layers/text_embed.py`)
- Wraps a pretrained **BERT-large-uncased** (340M parameters, entirely frozen during training)
- Tokenizes input text up to 512 tokens
- Outputs: `(batch, seq_len, 1024)` contextual embeddings + attention masks
- The BERT weights are never updated — this gives a stable text representation while keeping trainable parameter count manageable

**TextAdaptiveLayer** (`models/layers/adaptive_layer.py`)
- A lightweight, learnable module stacked on top of the frozen BERT output
- Architecture: multi-head self-attention (8 heads) → layer norm → feed-forward network → dropout
- Purpose: adapts the general-purpose BERT features to be more useful for the specific task of grounding geometry in text
- This is one of the **trainable** components (~100K parameters)

### CAD Decoder

The decoder is where the core generation happens. It's an autoregressive transformer that predicts one CAD token at a time.

**CADDecoder** (`models/decoder.py`)
- 8 transformer decoder layers, each with:
  - **Self-attention**: the growing CAD sequence attends to itself (with a causal mask so token _t_ can only see tokens 1 through _t-1_)
  - **Cross-attention** (layers 3–8 only): the CAD tokens attend to the text embeddings, grounding the geometry in the text prompt
  - **Feed-forward network**: two linear layers with ReLU
  - Residual connections and layer normalization throughout
- Design choice — **no cross-attention in layers 1–2**: the first two layers build up a self-consistent geometric representation before introducing text conditioning. This prevents the text signal from overwhelming the low-level geometry at early layers.

**CADSequenceEmbedder** (`models/layers/embedder.py`)
- Each CAD token has four channels: (x_coord, y_coord, flag, index)
- Four separate embedding tables map these discrete values to continuous vectors
- The four embeddings are **summed** into a single 256-dim token representation
- **PositionalEncodingSinCos** adds standard sinusoidal position information

**Attention modules** (`models/layers/attention.py`)
- **MultiHeadAttention**: 8 heads, 32-dim per head (256-dim total). Supports causal masks for self-attention.
- **CrossAttention**: same architecture but Q comes from CAD tokens, K/V come from text embeddings. Uses key padding masks to handle variable-length text.

**Output heads**: two linear projections (256 → 267) predict probability distributions over the vocabulary for the x and y coordinates independently.

### Model Orchestration

**Text2CAD** (`models/text2cad.py`)
- The top-level class that wires everything together
- `forward(cad_tokens, text, masks)` — training mode with teacher forcing: the decoder receives the ground-truth sequence shifted by one position
- `test_decode(text, maxlen, sampling_params)` — inference mode: generates tokens autoregressively from a START token, feeding each prediction back as input

**CELoss** (`models/loss.py`)
- Cross-entropy computed separately for x and y coordinate predictions
- Label smoothing of 0.1 to prevent overconfident predictions
- Padding-aware masking: loss is only computed on non-padding tokens

**Accuracy** (`models/metrics.py`)
- Token-level accuracy with a configurable tolerance (default: 3 quantization levels)
- A prediction within 3 levels of the ground truth counts as correct — this accounts for the fact that adjacent quantization levels represent nearly identical geometry

### Data Loading

**Text2CAD_Dataset** (`dataprep/t2c_dataset.py`)
- Pairs text prompts (from a CSV file) with pre-computed CAD vectors (from .pth files)
- Each CAD model has **4 text descriptions** at different complexity levels: _abstract_, _beginner_, _intermediate_, _expert_ — effectively 4x the training data
- Text preprocessing: lowercasing, special character removal
- Performance features: parallel loading via ThreadPoolExecutor, pickle caching of processed data
- Returns per sample: `(uid, cad_vec_dict, text_string, attention_mask_dict)`

### Training and Inference Scripts

| Script | Purpose |
|--------|---------|
| `train.py` | Full training loop with AdamW, exponential LR decay, gradient clipping, curriculum learning, checkpointing, TensorBoard logging |
| `test.py` | Batch inference on the test split — generates predictions for all test prompts and saves them for evaluation |
| `test_user_input.py` | Interactive single-prompt inference from the command line |

---

## CAD Sequence Processing (CadSeqProc/)

This is the geometry engine — everything between raw CAD data and the neural network's token representation.

### Core: CADSequence

**CADSequence** (`cad_sequence.py`) is the central data structure. It represents a complete CAD design as an ordered list of (sketch, extrusion) pairs.

Key methods:

| Method | What it does |
|--------|-------------|
| `json_to_vec()` | Serializes a DeepCAD JSON file into a flat token sequence: `cad_vec` (coordinates), `flag_vec` (token types), `index_vec` (hierarchy indices). This is the offline preprocessing step. |
| `from_vec()` | The reverse — reconstructs the full sketch/extrusion hierarchy from a flat token sequence. This is what runs at inference time. |
| `create_cad_model()` | Walks the sketch/extrusion list and builds the solid step by step using OpenCASCADE boolean operations (new body, join, cut, intersect). |
| `create_mesh()` | Tessellates the B-Rep solid into a triangle mesh for visualization. |
| `save_stp()` | Exports the solid as a STEP file (B-Rep) or STL file (mesh). |
| `sample_points()` | Generates a point cloud from the mesh surface (used for Chamfer Distance evaluation). |

### Sketch Representation

A sketch is a 2D profile drawn on a plane in 3D space.

**SketchSequence** (`sequence/sketch/sketchsequence.py`)
- A 2D sketch defined within a local coordinate system
- Contains one or more FaceSequence objects
- Handles quantization (float → 8-bit int) and denumericalization (int → float)

**FaceSequence** (`sequence/sketch/face.py`)
- A single bounded planar region
- Composed of closed Loop objects — one outer boundary and optional inner holes

**Loop** (`sequence/sketch/loop.py`)
- A closed wire made of ordered geometric primitives (lines, arcs, circles)
- Builds OCC wires by chaining curve edges end-to-end

**CoordinateSystem** (`sequence/sketch/coord_system.py`)
- Defines the 3D plane on which a sketch lives
- Encoded as 6 values: `(origin_x, origin_y, origin_z, theta, phi, gamma)` — three position coordinates and three Euler angles

### Extrusion Representation

**ExtrudeSequence** (`sequence/transformation/extrude_sequence.py`)
- Defines how a 2D sketch becomes 3D
- Parameters:
  - `extent_one`, `extent_two`: extrusion distances in each direction along the sketch normal
  - Boolean operation type: **NewBody** (first solid), **Join** (union), **Cut** (subtraction), **Intersect**
  - Reference to the sketch's coordinate system
- Uses OCC's `BRepPrimAPI_MakePrism` for the actual extrusion

### Geometric Primitives

| File | Primitive | Definition |
|------|-----------|------------|
| `geometry/line.py` | **Line** | Straight segment between two 2D points |
| `geometry/arc.py` | **Arc** | Circular arc defined by start, midpoint, and end |
| `geometry/circle.py` | **Circle** | Full circle defined by center and radius |
| `geometry/curve.py` | **Curve** | Base class — all primitives can quantize/denumericalize and generate OCC edges |

### OpenCASCADE Utilities

The `OCCUtils/` directory is a wrapper library around pythonocc-core:
- **Topo** (`Topology.py`): traverse shapes — iterate over faces, edges, vertices of a solid
- **Construct** (`Construct.py`): build B-Rep geometry — make edges, wires, faces from points
- **Common** (`Common.py`): bounding box computation, shared helpers
- Shape-specific utilities: `edge.py`, `face.py`, `wire.py`, `shell.py`, `solid.py`

### Dataset Preprocessing Scripts

These scripts convert raw DeepCAD data into the training format:

| Script | Purpose |
|--------|---------|
| `json2vec.py` | **Main preprocessor** — converts DeepCAD JSON files into quantized vector sequences (.pth). Normalizes coordinates to [0, 0.75], quantizes to 8-bit, serializes sketch/extrusion hierarchies into flat token streams, deduplicates identical designs. Run once before training. |
| `json2step.py` | Converts DeepCAD JSON directly into STEP files (for visualization/validation) |
| `json2stl_skt3d.py` | Converts DeepCAD JSON into STL meshes with optional 3D sketch edge overlays |
| `split_json.py` | Creates train/validation/test splits from the full dataset |
| `merge_vlm_minimal.py` | Merges VLM-generated text annotations with minimal JSON representations |
| `eda.py` | Exploratory data analysis — sequence length distributions, curve type frequencies, etc. |

### Constants and Utilities

**macro.py** (`utility/macro.py`) — the single source of truth for all global constants:

```
N_BIT = 8                          Quantization resolution (256 levels)
MAX_CAD_SEQUENCE_LENGTH = 272      Maximum tokens per design
MAX_SKETCH_SEQ_LENGTH = 150        Maximum tokens per sketch
MAX_EXTRUSION = 10                 Maximum extrusions per design
ONE_EXT_SEQ_LENGTH = 10            Tokens per extrusion block
NORM_FACTOR = 0.75                 Coordinate normalization range
CURVE_TYPE = [Line, Arc, Circle]   Supported geometric primitives
EXTRUDE_OPERATIONS = [NewBody, Join, Cut, Intersect]
```

The vocabulary size (267) is computed as: 7 structural tokens + 4 boolean tokens + 2^8 coordinate values.

---

## Web Application (App/)

| File | Purpose |
|------|---------|
| `app.py` | **Gradio web interface** — loads the model at startup, accepts text prompts, returns a 3D viewer (STL) and a downloadable STEP file. Includes a retry mechanism: if OpenCASCADE rejects the generated topology, the model re-generates with nucleus sampling up to 4 times. |
| `batch_eval.py` | Runs multiple prompts through the model in batch for evaluation |
| `debug_cpu_vs_mps.py` | Diagnostic comparing CPU vs Apple MPS inference — MPS produces NaN in the CAD decoder, so CPU is used |
| `debug_inference.py` | Step-by-step inference debugging tool |

### Retry Mechanism

Not every generated token sequence produces valid geometry — OpenCASCADE may reject self-intersecting loops or degenerate extrusions. The app handles this transparently:

1. Generate a CAD sequence using nucleus sampling (p=0.9)
2. Attempt to build the B-Rep via `CADSequence.from_vec().create_mesh()`
3. If OpenCASCADE rejects it, regenerate (the stochastic sampling produces a different sequence)
4. Repeat up to 4 times before reporting failure

---

## Evaluation (Evaluation/)

**eval_seq.py** computes metrics comparing generated CAD models against ground truth:

| Metric | What it measures |
|--------|-----------------|
| **Chamfer Distance (CD)** | Point-cloud similarity. Samples points from both meshes, computes mean nearest-neighbor distance. Multiplied by 1000 for readability. Lower = better. |
| **Curve Precision/Recall/F1** | Per curve type (line, arc, circle): what fraction of predicted curves match ground truth, and vice versa. Uses spatial matching. |
| **Extrusion Count Accuracy** | Whether the model predicted the correct number of extrusion operations |
| **Invalidity Ratio** | Fraction of generated sequences that fail B-Rep validation in OpenCASCADE |

Results are broken down by **text complexity level** (abstract, beginner, intermediate, expert) so you can see how performance varies with prompt difficulty.

---

## CAD Representation and Tokenization

### Quantization Scheme

All continuous coordinates are quantized to **8-bit integers** (256 levels). Sketch coordinates are first normalized to the range [0, 0.75] before quantization, giving an effective spatial resolution of ~0.003 per quantization step.

This is a deliberate precision-compression tradeoff: 8 bits is enough to represent the geometry of typical mechanical parts while keeping the vocabulary small enough for the transformer to learn efficiently.

### Token Vocabulary

The vocabulary has **267 classes**: 256 coordinate values + 7 structural tokens + 4 boolean operation tokens.

| Token | ID | Role |
|-------|----|------|
| `PADDING` | 0 | Pads sequences to fixed length (272) |
| `START` | 1 | Marks the beginning of a sequence |
| `END_SKETCH` | 2 | Terminates a sketch definition |
| `END_FACE` | 3 | Terminates a face boundary |
| `END_LOOP` | 4 | Terminates a closed curve loop |
| `END_CURVE` | 5 | Terminates a single curve primitive |
| `END_EXTRUSION` | 6 | Terminates an extrusion parameter block |
| IDs 7–10 | — | Boolean operation types (mapped from extrusion params) |
| IDs 11–266 | — | Quantized coordinate values (0–255 mapped to geometry) |

### Sequence Format

A complete design is a flat token stream with this hierarchical structure:

```
[START]
  ── Sketch 1 ──
  [coord_system: ox, oy, oz, theta, phi, gamma]     ← 6 tokens: 3D plane definition
  [face_1]
    [loop_1]
      [curve: (x1,y1), (x2,y2)] [END_CURVE]         ← Line: 2 points
      [curve: (x1,y1), (xm,ym), (x2,y2)] [END_CURVE] ← Arc: 3 points
    [END_LOOP]
  [END_FACE]
  [END_SKETCH]
  ── Extrusion 1 ──
  [extent_one, extent_two, boolean_op, ...] [END_EXTRUSION]  ← 10 tokens
  ── Sketch 2 ──
  ...
  ── Extrusion 2 ──
  ...
[PADDING to length 272]
```

Each token carries **three parallel channels**:
- **cad_vec** `(x, y)`: the quantized coordinate pair (integers 0–266)
- **flag_vec**: token type identifier (0–11) — is this a sketch coordinate? an extrusion parameter? a delimiter?
- **index_vec**: hierarchy index (0–10) — which sketch/extrusion group does this token belong to?

The flag and index channels give the model explicit structural awareness beyond what it could infer from the raw coordinates alone.

---

## Training Pipeline

Training is run via `Cad_VLM/train.py` with configuration from `config/trainer.yaml`.

### What's Frozen vs. Trainable

| Component | Parameters | Trainable? |
|-----------|-----------|------------|
| BERT-large | ~340M | Frozen |
| TextAdaptiveLayer | ~100K | Yes |
| CADDecoder | ~50–100M | Yes |
| **Total trainable** | **~50–100M** | |

### Training Loop

1. **Teacher forcing**: the decoder receives the ground-truth token sequence shifted by one position. At each step, it sees tokens 1 through _t-1_ and predicts token _t_.

2. **Loss**: cross-entropy computed independently for x and y coordinates. Label smoothing (0.1) prevents overconfident logits. Padding tokens are masked out.

3. **Optimization**:
   - AdamW (lr = 1e-4)
   - Exponential LR schedule (gamma = 0.999, ~10% total decay over 100 epochs)
   - Gradient clipping at max norm 0.9

4. **Curriculum learning** (optional): early epochs use data ordered by complexity (simpler designs first). Controlled by `curriculum_learning_epoch` in config. Set to 0 to disable.

5. **Validation**: each epoch, runs autoregressive decoding on a small subset and computes sequence accuracy with tolerance. Tracks and saves the best model.

6. **Checkpointing**: model + optimizer state saved every 10 epochs.

### Training Scale

- **Batch size**: 16
- **Epochs**: 150
- **Dataset**: 80K+ CAD models, each with 4 text descriptions = 320K+ training pairs
- **Hardware**: multi-GPU recommended (the training script supports distributed training)

---

## Inference Pipeline

Inference is fully autoregressive — no teacher forcing. The decoder generates one token at a time.

```
1. Encode the text prompt:
   text → BERT → adaptive layer → text_features (batch, seq, 1024)

2. Initialize the CAD sequence:
   cad_seq = [START token]

3. Generate tokens (t = 1 to 271):
   a. Embed current cad_seq → (batch, t, 256)
   b. Self-attention (causal mask)
   c. Cross-attention to text_features (layers 3–8)
   d. Project to vocabulary → logits (batch, 267) for x and y
   e. Sample next token:
      • Greedy: argmax
      • Top-k: sample from k most likely
      • Nucleus: sample from smallest set ≥ p cumulative probability
   f. Append token to cad_seq
   g. If token == END_EXTRUSION and no open sketches → stop

4. Reconstruct geometry:
   CADSequence.from_vec(cad_seq)
     → parse sketches + extrusions
     → build B-Rep solid via OpenCASCADE
     → boolean operations (join/cut/intersect)

5. Export:
   .save_stp("output", type="step")  → STEP file (B-Rep, lossless)
   .create_mesh().export("out.stl")  → STL file (mesh, for visualization)
```

### Sampling Strategies

| Strategy | Parameters | Behavior | Use Case |
|----------|-----------|----------|----------|
| Greedy | `nucleus_prob=0, topk=1` | Always picks highest-probability token | Most conservative; highest validity rate |
| Top-k | `nucleus_prob=0, topk=5–10` | Samples from the k most likely tokens | Moderate diversity with reasonable validity |
| Nucleus (Top-p) | `nucleus_prob=0.9` | Samples from smallest set covering 90% probability mass | Maximum diversity; used in retry loops |

### Inference Timing

- ~1–2 minutes per model on Apple Silicon (CPU)
- ~30 seconds on CUDA GPU
- The bottleneck is the 272-step autoregressive loop (no parallelism possible during generation)

---

## Configuration Reference

All YAML configs are in `Cad_VLM/config/`.

### trainer.yaml

```yaml
text_encoder:
  text_embedder:
    model_name: bert_large_uncased    # BERT variant
    max_seq_len: 512                  # Max input text tokens
    cache_dir: ".cache/huggingface"   # HuggingFace model cache

  adaptive_layer:
    in_dim: 1024     # BERT output dimension
    out_dim: 1024    # Output dimension (passed to decoder cross-attention)
    num_heads: 8     # Self-attention heads
    dropout: 0.1

cad_decoder:
  tdim: 1024         # Text feature dimension (must match adaptive_layer.out_dim)
  cdim: 256          # Internal CAD state dimension
  num_layers: 8      # Transformer decoder layers
  num_heads: 8       # Attention heads per layer
  dropout: 0.1
  ca_level_start: 2  # Cross-attention begins at this layer (0-indexed)

train:
  lr: 0.0001
  batch_size: 16
  num_epochs: 150
  checkpoint_interval: 10
  curriculum_learning_epoch: 0   # Set > 0 to enable curriculum learning
```

### inference_user_input.yaml

Same architecture block, plus:

```yaml
test:
  checkpoint_path: "../Cad_VLM/checkpoints/Text2CAD_1.0.pth"
  nucleus_prob: 0          # 0 = greedy/top-k, >0 = nucleus sampling
  sampling_type: "max"     # "max" for top-k, "nucleus" for top-p
  batch_size: 1

device: cpu                # MPS produces NaN; use cpu on macOS
```

---

## Setup and Installation

### Prerequisites

- Conda (Miniconda or Anaconda)
- ~4GB disk for dependencies
- ~2GB for model checkpoint

### Environment

**macOS (Apple Silicon):**
```bash
conda env create -f environment-mac.yml
conda activate text2cad
```

**Linux (CUDA):**
```bash
conda env create -f environment.yml
conda activate text2cad
```

### Critical Dependencies

| Package | Version | Role |
|---------|---------|------|
| `pythonocc-core` | 7.7.2 | OpenCASCADE B-Rep kernel — the version matters; the API changes across versions |
| `pytorch` | 2.2.1 | Deep learning framework |
| `transformers` | — | BERT model and tokenizer |
| `trimesh` | 4.1.8 | Mesh I/O and processing |
| `gradio` | — | Web interface |
| `loguru` | — | Structured logging |

### Model Weights

Download `Text2CAD_1.0.pth` from [HuggingFace](https://huggingface.co/datasets/SadilKhan/Text2CAD/blob/main/text2cad_v1.0/Text2CAD_1.0.pth) and place it at `Cad_VLM/checkpoints/Text2CAD_1.0.pth`.

---

## Running the System

### Web Interface (Gradio)

```bash
conda activate text2cad
cd App
KMP_DUPLICATE_LIB_OK=TRUE python app.py
```

Opens at `http://localhost:7860`. Enter a text prompt, get a 3D viewer and a downloadable STEP file.

### Command-Line Inference

```bash
cd Cad_VLM
python test_user_input.py --config_path config/inference_user_input.yaml \
  --prompt "A rectangular prism with a hole in the middle."
```

### Training

```bash
cd Cad_VLM
python train.py --config_path config/trainer.yaml
```

Requires the preprocessed dataset. To preprocess raw DeepCAD JSON:

```bash
cd CadSeqProc
python json2vec.py --input_dir $DEEPCAD_JSON \
  --split_json $SPLIT_JSON \
  --output_dir $OUTPUT_DIR \
  --max_workers 8 --padding --deduplicate
```

### Evaluation

```bash
cd Evaluation
python eval_seq.py --input_path ./predictions.pkl --output_dir ./results
```

Produces per-sample and aggregate metrics (Chamfer Distance, curve F1, invalidity ratio) broken down by text complexity level.
