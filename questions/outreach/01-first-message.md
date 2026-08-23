# First message to Irmak — drafts

Context on her research: `../../notes/00-research-context.md`

## Guiding principles

- **Lead with what's done, not with what you want.** "I worked through X and hit Y" is a
  different message from "can you help me get started."
- **Ask one question, not seven.** One specific question gets answered. A list gets
  postponed and then forgotten. The rest of `../open-questions.md` is for later.
- **Make the question one only she can answer.** Anything the docs cover looks like
  outsourced homework.
- **Reference her actual work.** Her papers are public and she wrote them to be read.
  Showing you read one is the strongest available signal that you're serious, and it costs
  her nothing.
- **Don't ask to be put on the project.** Show the work; let her draw the conclusion.
- **Keep it answerable from a phone** in a spare two minutes.

---

## Option A — text message (recommended)

> Hey Irmak — thanks for those links, going through them in order. Finished the basics one
> and built something small alongside it to make it concrete: a 2-link arm in torch where I
> solve IK with autograd instead of in closed form.
>
> The part that surprised me — if I train an MLP to do IK with MSE on joint angles it fails
> badly, ~0.94m error on a 1.7m arm. Each target has two valid elbow configs and MSE just
> learns their average, which reaches neither.
>
> Then I read T-Dex and wondered if that's part of why you use nearest-neighbor policies
> rather than a regression head — retrieval can't average two incompatible demos the way MSE
> does. Is that actually part of the motivation, or is it mostly about data efficiency with
> only a handful of demos?

**Why this one:** it has a number in it, it demonstrates you read her paper rather than just
her tutorial links, and the question is a real research-taste question with an easy either/or
shape. The "or is it mostly data efficiency" clause matters — it shows you already thought of
the obvious answer, so you're not asking her to explain her own abstract to you. If she
corrects you, that's a conversation, which is the actual goal.

---

## Option B — text, shorter

If Option A feels long for a first text:

> Hey Irmak — working through those tutorials, thanks for sending them. Built a small 2-link
> arm in torch alongside the basics one to get autograd to click.
>
> Found something I didn't expect: learning IK with MSE on joint angles fails badly (~0.94m
> error on a 1.7m arm) because each target has two valid elbow configs and MSE learns their
> average. Made me wonder if avoiding that averaging is part of why T-Dex uses
> nearest-neighbor policies instead of regression, or if that's mostly a data-efficiency
> thing?

---

## Option C — email (only if email is the norm with her)

> Subject: working through the PyTorch tutorials — one question about T-Dex
>
> Hi Irmak,
>
> Thanks for sending those over. I've been going through them in order and building small
> things alongside each one so the concepts land on something concrete rather than staying
> abstract.
>
> After the basics tutorial I wrote a 2-link planar arm in torch to get a feel for autograd
> on something physical. Solving IK by gradient descent on ‖forward_kinematics(θ) − target‖²
> works, and the autograd Jacobian matches the manipulator Jacobian I derived by hand
> exactly, which was satisfying to see.
>
> The unexpected part: I tried training an MLP to learn IK directly, and with MSE on joint
> angles it fails badly — 0.94 m mean error on an arm with 1.7 m of reach. Each hand position
> has two valid solutions (elbow up, elbow down), and squared error converges to their
> average, which reaches neither. The training loss plateaued rather than going to zero,
> which I take to be the irreducible variance of the targets. Switching to a task-space loss
> through the differentiable kinematics fixed it — 17× better with the same network and data.
>
> That sent me to T-Dex, and I'm curious about something. My read is that a nearest-neighbor
> policy is structurally immune to the failure above, because retrieving a demonstrated
> action can't produce the average of two incompatible ones. Is that part of why you went
> non-parametric, or is the motivation mostly data efficiency given you're working from a
> handful of demos? I'd guess it also gets much worse than my toy case on a 16-DOF hand,
> where the redundancy is a continuum rather than two options.
>
> Next up is the char-RNN tutorial, then seq2seq with attention. My working theory on why
> those two are useful for hands is that tactile streams are variable-length sequences and
> retargeting human poses to robot joints is structurally a translation problem — but I'd
> be curious whether that's what you had in mind.
>
> Thanks,
> [YOUR NAME]

---

## Don't send yet

Save for after she replies — these are the natural second and third exchanges, and which one
is right depends entirely on how she answers:

- If she engages on multimodality → ask how it shows up in retargeting on a 16-DOF hand.
- If she says it's mostly data efficiency → ask what would have to change for a parametric
  policy to be worth it.
- If she asks what you're doing next → the question about whether seq2seq→retargeting is the
  intended arc (`../open-questions.md`, Section 2).
- The BYOL/VICReg question is the best one for "what should I learn to be useful," because
  it's about her pipeline rather than about general theory.

## Note on the phone number

She sent it in chat. Keep it out of this repo and out of any file that could get pushed —
there is no reason for a contact number to live in version control.
