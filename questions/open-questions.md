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

## Section 2 — Worth actually asking her

Ordered by how much I expect the answer to change what I build next.

### About her setup (highest value — I can't get these anywhere else)

1. **What are the observation and action spaces in your work?** Specifically: are actions
   absolute joint positions, joint deltas, end-effector poses, or velocities? I've read that
   this choice matters more than the architecture, and I'd like to understand why it's
   contentious rather than just picking one.

2. **Sim, real, or both?** If sim, which one (MuJoCo / Isaac / Drake)? If real hardware, how
   do you collect demonstrations — teleop, kinesthetic teaching, scripted policies? I'd
   rather learn the tooling you actually use than something adjacent.

3. **How much data is "enough" in your setting?** Tens of demos, hundreds, thousands? I have
   no intuition for the scale and it seems like it drives every other decision.

### About the multimodality result I hit (stage 1)

4. **Does multimodality bite you in practice, and how do you handle it?** I ran a small
   experiment where an MLP trained to do inverse kinematics on a 2-link arm with joint-space
   MSE fails badly (0.94 m error on a 1.7 m arm) purely because two elbow configurations
   reach each target and MSE learns their average. Switching to a task-space loss through
   the differentiable kinematics fixed it, 17× better with the same network. My read is that
   this is the same failure that motivates diffusion policies and action chunking — is that
   the right way to think about it, or am I over-reading a toy result?

5. **Is differentiating through the physics ever practical for you?** The task-space-loss fix
   only worked because my forward kinematics was smooth torch code. My assumption is that
   this breaks down the moment contact is involved, which is why differentiable simulation
   hasn't taken over. Is that roughly right, or is it more useful than I think?

### About direction

6. **Given these three tutorials, where were you pointing me?** My reading is that seq2seq
   with attention is the ancestor of the transformer, and a policy like ACT is
   architecturally an encoder-decoder that emits action chunks instead of words — so the
   tutorials are teaching the machinery, and the robot part is a change of input and output
   representation. Is that the intended takeaway, or is there something else in there I'm
   missing?

7. **What should I read after this?** I've got ACT, Diffusion Policy, and OpenVLA on my list
   as the three that seem most cited. If there's one paper that's closest to what you
   actually work on, I'd rather spend the time there.

---

## Answers received

*(log her replies here, dated, so this becomes the actual reference document)*
