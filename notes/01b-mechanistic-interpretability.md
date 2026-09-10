# Mechanistic interpretability on the 2-link arm

Code: `robotics/stage1_interpret_fk.py`

Run:

```bash
python robotics/stage1_interpret_fk.py
```

## The question

The analytic forward-kinematics calculation is transparent:

```text
(shoulder angle, elbow angle)
    -> rotate link 1 and link 2
    -> add their displacement vectors
    -> fingertip (x, y)
```

The MLP in `stage1_learn_fk.py` learns the same input-output relationship, but getting the
right answer does not tell us how it represents the calculation internally. Mechanistic
interpretability asks what internal representations contribute to the behavior and tests
those hypotheses by intervening on the model.

This experiment looks for the four trigonometric quantities in the known solution:

```text
cos(shoulder), sin(shoulder), cos(shoulder + elbow), sin(shoulder + elbow)
```

## Two kinds of evidence

### 1. Observation: correlations and linear probes

A correlation asks whether one hidden unit rises and falls with one known geometric feature.
A linear probe asks whether that feature can be reconstructed from the hidden layer as a
whole.

A successful probe means the information is **decodable**. It does not prove that the network
uses the information, and it does not mean that one neuron exclusively represents one
concept. A sufficiently expressive representation can make many quantities decodable.

### 2. Intervention: mean ablation

The script replaces one hidden unit at a time with its average training-set activation, then
remeasures fingertip error. If accuracy reliably worsens, that unit has a causal effect on
the network's output under this intervention.

Ablation is stronger evidence than correlation, but it is not perfect. Changing one
activation can create a hidden state unlike those encountered during training, and another
unit may redundantly carry the same information.

## First result

With the fixed seed in the script, the network reached about `0.022 m` mean held-out
fingertip error. All four geometric quantities were linearly decodable from the first hidden
layer. Individual units had weaker correlations, while ablating the most consequential units
increased error by several tenths of a meter.

That combination is the important result:

- the layer contains a highly usable representation of the known geometry;
- no single correlation establishes a clean "shoulder neuron" or "forearm neuron";
- interventions show that some units matter causally for accurate predictions;
- the network's solution is distributed across multiple units.

The exact values can vary after changing the seed, architecture, or training procedure. A
real conclusion should be repeated across several seeds rather than inferred from one run.

## Why this belongs in the robotics path

The earlier arm exercise compared a known geometric mechanism with a learned approximation.
That gives us ground-truth concepts against which interpretability methods can be checked.
Later, the same questions become harder and more useful:

- Does an RNN hidden state encode trajectory phase or a motion primitive?
- Does an attention model rely on the relevant contact or merely a background cue?
- Does a tactile encoder represent contact geometry, object identity, or teleoperator habits?
- When a policy fails under distribution shift, which internal features drove its action?

This is adjacent to Irmak Guzey's work rather than a stated focus of her publications. Her
work centers on data-efficient dexterous manipulation, tactile representations, human
demonstrations, and retargeting. Interpretability can contribute as a diagnostic method for
those learned representations and policies, but it should not be presented as her primary
research topic.

## Next experiments

1. Repeat training over several random seeds and report confidence intervals.
2. Group correlated units and ablate directions rather than individual neurons.
3. Train the trajectory RNN, then probe whether hidden state encodes motion class and phase.
4. Introduce a spurious feature into the data and test whether the model relies on it.
5. Apply the same observational and interventional distinction to tactile encoders or a
   small vision-language-action model.
