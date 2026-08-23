# Open questions

A running log. Two rules that make this file useful rather than embarrassing:

1. **Anything answerable by reading goes in the top section, and I answer it myself.**
   Asking a researcher something the docs answer in thirty seconds spends credibility for
   nothing.
2. **Only questions that genuinely need her context go in the second section.** Those are
   questions about *her* setup, her tradeoffs, and judgment calls that the literature
   disagrees about. Those are worth someone's time, and asking them well is itself the
   signal that I'm ready to be useful.

---

## Section 1 — I should answer these myself

Status: `open` / `answered`

### Stage 1

- [x] **Why does `.grad` accumulate instead of overwriting?** — *answered.* So you can sum
  gradients from multiple losses, and accumulate across mini-batches to simulate a larger
  batch than fits in memory. RNNs backpropagating through time rely on the same mechanism.
- [x] **Why must `.backward()` be called on a scalar?** — *answered.* With a vector output
  there's no single derivative; you'd need a full Jacobian, or a weighting vector telling
  it how to combine the outputs (`out.backward(torch.ones_like(out))`).
- [x] **Why did Part B's loss plateau at 2.36?** — *answered.* Irreducible variance of the
  targets. For each input there were two valid θ, and MSE's optimum is their mean, so the
  residual is the spread of the targets around that mean. No amount of capacity removes it.
- [ ] **What exactly does `nn.RNN` return, and what are the shapes?** — for stage 2. Need to
  work out `output` vs `h_n` and what `batch_first` changes.
- [ ] **Why does the char-RNN tutorial use `NLLLoss` with `LogSoftmax` instead of
  `CrossEntropyLoss` on raw logits?** — for stage 2. I think they're equivalent and the
  split version is for pedagogy, but I want to confirm and understand the numerical
  stability argument.
- [ ] **In stage 3, why does `AttnDecoderRNN`'s GRU take `2 * hidden_size` as input?** —
  presumably embedding concatenated with the attention context vector. Confirm by reading.

---

## Section 2 — Worth actually asking Irmak

Tailored to her actual research (dexterous manipulation with multi-fingered hands — see
`../notes/00-research-context.md`). Ordered by how much the answer would change what I build
next. **Send one at a time**, starting with #1.

### The strongest one (send first)

1. **Is avoiding mode-averaging part of why T-Dex uses non-parametric policies?** My stage 1
   experiment: an MLP learning inverse kinematics for a 2-link arm with joint-space MSE fails
   badly (0.94 m error on a 1.7 m arm) purely because each target has two valid elbow
   configurations and MSE converges to their average. A nearest-neighbor policy seems
   structurally immune to that, since retrieval can't average two incompatible demos. Is
   that part of the reasoning, or is the motivation mostly data efficiency given you're
   working from a handful of demos?

### About redundancy and retargeting

2. **How much worse does redundancy get on a 16-DOF hand?** My toy case had two discrete
   solutions. An Allegro hand grasping an object has a continuum of valid configurations. Is
   choosing among them mostly handled by the retargeting step, by the reward, or does it
   just not bite as hard in practice as I'd expect?

3. **In Holo-Dex / HuDOR, what actually defines a "correct" retargeting** from a human hand
   to a robot hand with different kinematics? Fingertip positions, joint angles, contact
   pattern, or task outcome? This feels like the crux and I can't tell from the papers which
   choice is doing the work.

### About representations (probably the most useful thing for me to learn)

4. **Why BYOL and VICReg specifically for the tactile encoders in T-Dex?** Was that a
   considered choice among self-supervised objectives, or the-thing-that-worked? And is
   there something about tactile data — sparsity, the fact that contact is mostly nothing
   then suddenly something — that makes the standard augmentation recipes a bad fit?

5. **What does 2.5 hours of play data actually look like?** I'm trying to understand what
   makes play data useful rather than just unlabeled. Is it coverage of contact events, or
   diversity of objects, or something else you were deliberately going for?

### About the tooling and where to be useful

6. **Sim or all real?** My read is that tactile sensing is hard enough to simulate that
   T-Dex trains entirely on real teleoperated data, which suggests "learn Isaac Sim" is the
   wrong instinct for your lab. Is that right, or does sim still play a role somewhere?

7. **Is differentiating through the physics ever practical for you?** My task-space-loss fix
   only worked because the kinematics were smooth torch code, and I assume contact and
   friction break that. Is differentiable simulation useful in your work at all, or is it
   still mostly a promise?

### About direction

8. **Given these three tutorials, where were you pointing me?** My working theory: tactile
   and proprioceptive streams are variable-length sequences (the char-RNN case), retargeting
   human poses to robot joints is structurally translation (the seq2seq case), and attention
   is how you decide which contacts matter right now. Also that both are deliberately "from
   scratch" about data preprocessing, which is the actual bottleneck in your area. Is that
   the intended arc, or am I inventing a story?

9. **What should I read after this?** Current list is VINN, BYOL/VICReg, Diffusion Policy,
   and the optimal-transport reward matching in TAVI. If there's one that's closest to what
   you're working on right now, I'd rather spend the time there.

10. **Is there a piece of infrastructure I could usefully build or fix?** Not asking to be
    put on a project — but data pipelines, teleop tooling, eval harnesses, and RUKA assembly
    are all real work, and I'd rather be useful at the boring end than wait until I've read
    enough papers.

---

## Answers received

*(log her replies here, dated, so this becomes the actual reference document)*
