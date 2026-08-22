# robot-learning-playground

A self-directed study repo for getting from "I can read PyTorch code" to "I can reason about
how a learned robot policy works."

The curriculum is built around three PyTorch tutorials, worked through in order, with a small
robotics project attached to each one so the concept lands on something physical instead of
staying abstract.

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

## Stages

Each stage has notes (the concepts, in prose, with the parts that are genuinely subtle called
out) and code (a faithful walkthrough of the tutorial, plus a robotics bridge).

| Stage | Concept | Tutorial | Robotics bridge |
| --- | --- | --- | --- |
| 1 | Tensors, autograd, `nn.Module`, training loops | [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | Inverse kinematics for a 2-link arm, solved by gradient descent |
| 2 | RNNs, hidden state, sequence -> label | [Char-RNN Classification](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) | Classify arm trajectories into motion primitives |
| 3 | Encoder-decoder, attention, teacher forcing | [Seq2Seq Translation](https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) | Language command -> action sequence, with an attention heatmap |
| 4 | Where this becomes real robot learning | reading list | read one policy paper and map it onto stage 3 |

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
