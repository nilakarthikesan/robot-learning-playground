# robot-learning-playground

A self-directed study repo for getting from "I can read PyTorch code" to "I can reason about
how a learned robot policy works."

The curriculum is built around three PyTorch tutorials, worked through in order, with a small
robotics project attached to each one so the concept lands on something physical instead of
staying abstract.

The target is specific: being able to hold a real conversation about **data-driven dexterous
manipulation with multi-fingered hands** — tactile sensing, learning from few demonstrations,
and human-to-robot retargeting. See `notes/00-research-context.md` for why that's the target
and which papers matter.

## Why these three tutorials

The tutorials are about text (classifying names, translating French to English). That looks
unrelated to robots. It isn't, and the reason is worth stating precisely:

**A robot policy is a sequence-to-sequence model.** It consumes a sequence (camera frames,
joint states, a language instruction) and emits a sequence (joint targets, gripper commands).
Every modern learned-manipulation system is some variation on that shape:

| System | Input sequence | Output sequence |
| --- | --- | --- |
| ACT / Action Chunking Transformer | images + joint state | chunk of ~100 future actions |
| Diffusion Policy | recent observation history | short action trajectory |
| RT-2 / OpenVLA / pi-0 | language instruction + images | discretized action tokens |

The encoder-decoder-with-attention architecture in tutorial 3 is the direct ancestor of the
transformer, which is the backbone of all of the above. So the path is:

```
tensors + autograd  ->  RNN (sequence -> label)  ->  seq2seq + attention (sequence -> sequence)  ->  transformer policies
```

Learning it on text first is a feature, not a detour: text data is tiny, trains on a laptop CPU
in minutes, and has no simulator, no robot, and no reward function to confound things. You get
to isolate the architecture.

For multi-fingered hands specifically the mapping is even more direct. Tactile and
proprioceptive streams *are* variable-length sequences, which is the char-RNN case. Retargeting
a human hand pose onto a robot hand with different kinematics *is* translation between two
sequences with no frame-by-frame correspondence, which is the seq2seq case. And attention is how
you express "which contacts matter right now."

## Stages

Each stage has notes (the concepts, in prose, with the parts that are genuinely subtle called
out) and code (a faithful walkthrough of the tutorial, plus a robotics bridge).

| Stage | Concept | Tutorial | Robotics bridge |
| --- | --- | --- | --- |
| 1 | Tensors, autograd, `nn.Module`, training loops | [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | Inverse kinematics for a 2-link arm, solved by gradient descent |
| 2 | RNNs, hidden state, sequence -> label | [Char-RNN Classification](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) | Classify arm trajectories into motion primitives |
| 3 | Encoder-decoder, attention, teacher forcing | [Seq2Seq Translation](https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) | Retargeting as translation: human hand poses -> robot joint commands, with an attention heatmap |
| 4 | Non-parametric policies and self-supervised representations | VINN, BYOL/VICReg, T-Dex | Nearest-neighbor policy vs. the regression policy that failed in stage 1 |

### Stage 1 in detail

`concepts/stage1_basics/` maps one file per section of [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html),
using the tutorial's own numbering:

| File | Tutorial section | Status |
| --- | --- | --- |
| `00_quickstart.py` | Quickstart | todo |
| `01_tensors.py` | Tensors | done |
| `02_datasets_dataloaders.py` | Datasets & DataLoaders | todo |
| `03_transforms.py` | Transforms | todo |
| `04_build_model.py` | Build the Neural Network | todo |
| `05_autograd.py` | Automatic Differentiation | done |
| `06_optimization.py` | Optimizing Model Parameters | todo |
| `07_save_load.py` | Save and Load the Model | todo |

Each file follows the tutorial's sections in order, then adds a "beyond the tutorial"
part for the things the tutorial mentions in passing that cause most real bugs.

A `*_verbatim.py` file, where present, is the tutorial's code with nothing added — meant to
be read side by side with the webpage before reading the annotated version.

## Layout

```
notes/         concept explanations, one file per stage
concepts/      faithful walkthroughs of the PyTorch tutorials
robotics/      the bridge projects - same concepts, robot-flavored data
questions/     running log of things I don't understand yet, + outreach drafts
```

## Setup

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

Everything in this repo is sized to run on a laptop CPU. Nothing needs a GPU.

## How to use this repo

Run the scripts, don't just read them. Every script prints shapes and intermediate values
because **shape confusion is the single biggest source of bugs in sequence models**, and
building the habit of checking shapes early is most of the skill.

`questions/open-questions.md` is the important file. It is the record of what is still unclear,
which is the thing worth actually discussing with someone who does this research.
