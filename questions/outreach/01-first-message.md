# First message — drafts

Replace `[NAME]` before sending. Pick one; don't send both.

## Guiding principles for these

- **Lead with what's done, not with what you want.** "I worked through X and hit Y" is a
  different message from "can you help me get started."
- **Ask one question, not seven.** A message with one specific question gets answered. A
  message with a list gets postponed and then forgotten. The rest of
  `../open-questions.md` is for conversation, not for a first text.
- **Make the question one only she can answer.** Anything the docs cover makes the ask
  look like outsourced homework.
- **Don't ask to be put on the project.** Show the work; let her draw the conclusion. If
  the work is good, the ask becomes unnecessary.
- **Keep it short enough to answer from a phone**, in a spare two minutes, between things.

---

## Option A — text message (recommended)

> Hey [NAME] — started working through those PyTorch tutorials you sent. Went through the
> basics one and then built a small side thing to make it concrete: a 2-link arm where I
> solve IK with autograd instead of closed form.
>
> Ran into something I didn't expect. If I train an MLP to do IK with plain MSE on joint
> angles it fails badly — ~0.94m error on a 1.7m arm — because each target has two valid
> elbow configs and MSE just learns their average. Switching to a task-space loss through
> the kinematics fixed it, 17x better, same network.
>
> Is that the same problem diffusion policies and action chunking are solving? Feels like
> it but I might be over-reading a toy result.

**Why this one works:** it's specific, it has a number in it, and the question is a real
conceptual question that shows you connected a toy experiment to the literature. It's also
answerable in one sentence, which is why you'll get an answer.

---

## Option B — email (if email is the norm with her)

> Subject: working through the PyTorch tutorials — one question
>
> Hi [NAME],
>
> Thanks for sending those over. I've been going through them in order and building small
> things alongside each one so the concepts land on something concrete — I put it in a repo
> if you ever want to look, though it's mostly for me.
>
> I finished the basics tutorial and then wrote a 2-link planar arm in torch to play with
> autograd on something physical. Solving IK by gradient descent on
> ‖forward_kinematics(θ) − target‖² works, and the autograd Jacobian matches the manipulator
> Jacobian I derived by hand exactly, which was a nice thing to see.
>
> The part I didn't expect: I tried training an MLP to learn IK directly, and with MSE on
> joint angles it fails badly — mean error 0.94m on an arm with 1.7m of reach. The cause
> seems to be that each hand position has two valid solutions (elbow up and elbow down) and
> squared error converges to their average, which reaches neither. Its training loss
> plateaued instead of going to zero, which I take to be the irreducible target variance.
> Changing to a task-space loss through the differentiable kinematics fixed it — 17x better
> with the same network and data.
>
> My question: is this the same failure mode that motivates diffusion policies and action
> chunking for behavior cloning? The reasoning I'd give is that a human demonstrator who
> sometimes goes left and sometimes right around an obstacle creates the same one-to-many
> structure, and an MSE policy would split the difference and hit the obstacle. I'd like to
> know whether that's the right way to think about it before I build any more on top of it.
>
> Next up is the char-RNN tutorial, then seq2seq with attention.
>
> Thanks,
> [YOUR NAME]

---

## Follow-up material (don't send yet)

Save these for after she replies — they're the natural second and third exchanges:

- Which of `../open-questions.md` Section 2 questions is most relevant depends entirely on
  how she answers this one. Let her steer.
- If she engages with the multimodality question, question 5 (differentiable physics and
  contact) is the natural follow-on.
- If she asks what you're doing next, question 6 (whether seq2seq→transformer→ACT is the
  intended arc) is a good way to check whether you've understood *why* she sent these three
  specific tutorials.
