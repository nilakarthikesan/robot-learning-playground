# Progress update to Irmak — draft

A lighter-weight message than `01-first-message.md`: a progress note that ends in an open
"what would you change" rather than a specific research question. Send this one if the goal
is to keep her in the loop; send 01 if the goal is to start a technical conversation.

---

## Sent version

> Hey Irmak! been going through the tutorials you sent, starting with Learn the Basics.
> Getting the vocabulary straight first: a tensor is a grid of numbers with a shape, a matrix
> multiply moves a point from one frame into another, and a neural network is a stack of those
> multiplies where the numbers are learned from examples instead of derived.
>
> To try to understand them better I think I could try to build a 2-link arm in torch and do
> forward kinematics twice: once with rotation matrices, once with a small MLP trained on 4k
> (angles → fingertip) pairs.

---

## Follow-up: short update + can't meet today

> Hey Irmak! Quick update — the two-approach framing finally clicked for me: rotation matrices
> give the fingertip exactly, worked out by hand, while the network has to find that same
> relationship from examples instead, and the training loop is the part actually doing the
> learning. Took your advice and I'm writing that loop myself rather than using the shortcut.
>
> I'm moving apartments today so I can't meet, but I'll send a proper update this weekend once
> I'm further through the three links.

Deliberately says "clicked" rather than "built" — the arm exercise is in progress, not done.

---

## Notes

- The three one-line definitions are deliberately plain. If she corrects one of them, that is
  a useful conversation and costs her ten seconds.
- Framed as a plan rather than a finished result, which is the honest tense — the arm exercise
  is the thing being worked through, not something already mastered.
- No repo link included — nothing is hosted yet. Add one only if the repo is pushed somewhere.
- Her name is Irmak.
