# Research context: what she actually works on

Reading someone's papers before asking them questions is the cheapest possible way to make
your questions worth answering. This file exists so the rest of the repo aims at her actual
problems instead of generic robotics.

## Who

**Irmak Guzey** — PhD student at NYU's CILVR lab (Courant), advised by **Lerrel Pinto**.
Started the PhD Sept 2024; master's at NYU before that (Best Master's Thesis Award, Fulbright
scholar). Previously an Embodied AI intern at Meta; earlier roles at Google/X and DOF Robotics.
[Homepage](https://irmakguzey.github.io/) ·
[Scholar](https://scholar.google.com/citations?user=0FEl834AAAAJ&hl=en)

## What

**Data-driven dexterous manipulation for multi-fingered robot hands.** Not mobile robots, not
locomotion, not a 6-DOF arm doing pick-and-place. Hands — many joints, heavy contact,
self-occlusion, and very little task data.

The through-line of her work is one problem: *collecting task-specific data with multi-fingered
hands is brutally hard.* Every paper is an angle of attack on that.

## Her papers, in the order worth reading them

| Year | Paper | The idea | Why it matters here |
| --- | --- | --- | --- |
| 2022 | **Holo-Dex** | Teach dexterity through immersive mixed reality teleop | Where the demos come from |
| 2023 | **T-Dex** (CoRL) | 2.5h of aimless "play" data → self-supervised tactile encoders (BYOL/VICReg); then **non-parametric policies** from a handful of demos, combining tactile + vision. 1.7× over vision/torque-only on 5 tasks | The one to read first — see below |
| 2024 | **Open Teach** | General VR teleoperation system for manipulation | Infrastructure; most-cited of her work |
| 2024 | **TAVI / See to Touch** (ICRA) | Contrastive visual representations → optimal-transport reward matching against **one** human demo → online RL to optimize tactile policies. 73% on 6 contact-rich tasks with a 4-fingered Allegro hand | RL, not just behavior cloning |
| 2025 | **HuDOR** (ICRA) | Object-oriented rewards from visual trackers, training directly from **human video** | The human→robot correspondence problem |
| 2025/26 | **RUKA / RUKA-v2** | Open-source tendon-driven humanoid hand, 11 DOF, buildable under $1,300; v2 adds a 2-DOF parallel wrist and finger abduction (51.3% faster task completion, 21.2% higher success) | Hardware + a learned tendon→joint controller |
| 2026 | **Dexterity from Smart Lenses** | 3D policies from in-the-wild human demonstrations captured on smart glasses | Where the field is going |

Hardware in the loop across these: **Allegro** four-fingered hand (16 DOF), **XELA** tactile
sensors, **Kinova** arm, and increasingly her own **RUKA** hand.

---

## The connection to my stage 1 result

This is the part worth thinking hard about.

In `robotics/stage1_learn_fk.py` I found that training an MLP to do inverse kinematics with
mean-squared error on joint angles fails badly — 0.94 m error on an arm with 1.7 m of reach —
because each hand position has two valid elbow configurations and MSE converges to their
average, which reaches neither.

**T-Dex does not use a regression policy. It uses a non-parametric, nearest-neighbor one.**

A nearest-neighbor policy *retrieves* an action that some demonstrator actually performed. It
cannot emit the average of two valid-but-incompatible actions, because averaging is not an
operation it has. That is structurally immune to the exact failure I measured.

The stated motivation in the paper is data efficiency — you have a handful of demos, so don't
train a policy on them. But avoiding mode-averaging looks like a second, maybe unstated,
benefit. **Whether that's a real part of the reasoning or me over-fitting my toy result to her
paper is the best question I have right now.**

Worth noting how much worse this problem gets on her hardware. My arm had 2 joints and 2
redundant solutions. A 16-DOF Allegro hand grasping an object has a continuum of valid
configurations, and human-to-robot retargeting (Holo-Dex, HuDOR, Smart Lenses) is exactly the
problem of choosing among them. Redundancy is not an edge case in her work; it is the
setting.

## Why these three tutorials, revised

My first guess was "seq2seq is the ancestor of the transformer, and policies are transformers."
Still true, but with her actual work in view there's a more direct reading:

- **Tactile and proprioceptive streams are variable-length sequences.** The char-RNN tutorial
  is the smallest honest example of encoding a variable-length sequence into a
  fixed representation and classifying it. Swap characters for XELA frames and it is the same
  code.
- **Retargeting is sequence-to-sequence.** Human hand poses in → robot joint commands out,
  with different lengths and no frame-by-frame correspondence. That is literally the
  translation problem in tutorial 3.
- **Attention is how you handle "which part of the input matters right now."** For a hand,
  that's which fingers and which contacts are load-bearing at this instant.
- Both tutorials are also deliberately **"from scratch"** — they build the data preprocessing
  by hand rather than importing a dataset. Given that her whole research area is *data*
  problems, that emphasis is unlikely to be accidental.

## What to learn after stage 3, tailored to her

Generic answer was ACT / Diffusion Policy / OpenVLA. Better answer for this lab:

1. **VINN** (Visual Imitation through Nearest Neighbors) — the policy class T-Dex builds on.
2. **BYOL and VICReg** — the self-supervised objectives she uses to get tactile and visual
   encoders without labels. This is the skill that unlocks her pipeline.
3. **Diffusion Policy** — the parametric answer to the multimodality problem, so I can
   compare it against the non-parametric one.
4. **Optimal transport for reward matching** (TAVI) — how you turn one human demo into a
   reward signal.

Notably, none of these need a simulator. Tactile sensing is hard to simulate, which is why
T-Dex trains on real teleoperated data. So "learn Isaac Sim" is probably the wrong instinct
for this lab.
