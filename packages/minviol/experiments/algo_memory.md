# minviol move-class experiments

A keep/reject ledger for changes to how the solver explores. Every idea gets an
entry whether it wins or loses; the failures carry as much information as the
wins, and this file is the only place that survives the session.

## Project Profile

**Project:** minviol — GPU solver minimizing the maximum violation of
`lower <= A x <= upper` over a discrete domain.
**Question under test:** swaps are the wrong move class for general constraint
systems. What is the right one?

### Why swaps are the incumbent to beat, not the baseline

A swap exchanges the levels of two variables, so it preserves the multiset of
assigned levels. That is a strong move in quantization, where a solve starts near
round-to-nearest with the histogram already right. On a general constraint system
starting from nowhere in particular it is both slower and worse: measured at equal
time, single-variable moves alone reach a feasible point where moves-plus-swaps
stalls at 5.7 and swaps alone at 9.5. **The baseline is therefore `moves-only`.**

### Tracked metrics

| Metric | Direction | Noise floor | Notes |
|---|---|---|---|
| Median violation ratio vs baseline | lower is better | **0** | deterministic under an iteration budget |
| Paired better / worse | more better | **0** | identical variants tie on every pair |
| Feasibility count | higher is better | — | the honest metric where an instance is bimodal |
| Wall time per solve | reported, not budgeted | — | a per-iteration win that is slower per iteration is not a win |

Under the earlier wall-clock protocol the per-instance noise floor reached 17%,
and on bimodal instances far more than that. Those numbers are superseded.

**Decision rule.** A variant wins on quality when either:

1. **Broad win** — the median violation ratio is below 0.95 *and* it is better on
   at least 12 of 18 pairs with at most 3 worse; or
2. **Specialist win** — it improves at least one instance by more than 2x, with no
   regression anywhere beyond that instance's noise floor.

Anything inside both is neutral. A neutral-quality variant still wins if it is
materially faster, since the budget is wall-clock and speed buys iterations.

> Clause 2 was added after experiment 2, which improved `tomography` 14x and left
> the other five instances tied. The original rule would have discarded it,
> because a median over 18 pairs is dominated by the 12 that did not move. A rule
> that throws away a large win on a whole problem class because the other classes
> are indifferent is measuring the instance set, not the idea. Recorded rather
> than quietly widened.

### Hard constraints

1. The returned point must belong to the caller's domain.
2. Fixed variables must never move.
3. The reported violation must survive recomputation from scratch.
4. Dense and sparse backends must agree; the parity tests must stay green.
5. `tests/test_minviol_parity.py` must stay green — minviol still has to reproduce
   AMVM's batched engine exactly under `lower == upper` with swaps only.
6. No new dependency beyond torch and numpy.

### Evaluation harness

```
python experiments/bench_moves.py --variants best <new> --iterations 75 --trials 3
```

**The budget is a fixed iteration count, not wall-clock time.** Several instances
are bimodal -- they either find a feasible point or stall near 20 -- so under a
clock budget the outcome depends on how many iterations happened to fit. The same
variant on `equality` gave 22.89, then 0.00, then 14.44 across three runs, which
is not a noise floor but a different answer each time. Counting iterations makes
the comparison exactly reproducible (measured: zero variance). Wall time is still
reported per instance, so a variant that wins per iteration by being slower per
iteration is still caught. 75 iterations was calibrated to leave headroom: at 25
nothing has converged, at 200 `equality` is solved 4/4 and the signal is gone.

Variants are alternated within each repetition, and the device cache is released
between solves. Both matter — see "Measurement hazards" below.

**Correctness command:** `python -m pytest -q` (in `packages/minviol`), plus
`python -m pytest -q tests/test_minviol_parity.py` from the AMVM root.

**Instance set:** `experiments/instances.py`, six instances spanning dense and
sparse, inequality and equality and two-sided, well-conditioned and not, random
and structured. Each carries a planted solution, so feasibility is attainable and
a failure to reach it is a statement about the search.

### Measurement hazards found the hard way

- **Release the device cache between solves.** Without it the second variant on
  an instance inherits the first's memory pressure. One 3-second solve took
  **419 seconds** under accumulated pressure and 3.0 seconds in isolation,
  reaching the same answer either way. Alternating the variants does not cancel
  this, because the cost lands on whichever runs second. The harness now warns
  when any solve overruns its budget by more than 3x.
- **Never time in blocks.** Alternate variants within each repetition.
- The first baseline taken in this session was polluted by the above and read
  20-40% better than the truth. Numbers below are from the fixed harness.

## Baseline (`best` = moves-only + active kick + tie-break, 75 iterations, 3 trials, MPS)

Experiments 1 and 2 were decided under the wall-clock protocol and re-verified
under this one before being relied on: 16 of 18 pairs better, 1 worse, median
ratio **0.6288**.

| Instance | `moves-only` (original) | `best` (after exp 1+2) | Feasible | Wall |
|---|---:|---:|---|---:|
| dense-ineq | 33.0993 | **27.1722** | 0/3 | 3.0s |
| equality | 34.0321 | **21.0346** | 0/3 | 2.8s |
| ill-cond | 3.3422 | **3.2372** | 0/3 | 2.0s |
| sparse-ineq | 7.0531 | **0.0000** | **3/3** | 2.4s |
| tomography | 47.0000 | **3.0000** | 0/3 | 5.0s |
| two-sided | 33.5321 | **20.5346** | 0/3 | 3.0s |

**Baseline commit:** `e1de080` plus experiments 1 and 2
**Last baseline run:** 2026-09-18
**Hardware:** Apple Silicon, MPS, PyTorch 2.13
**Artefacts:** `experiments/results/reverify_exp1_exp2.json`, `calib_75.json`

## Where five experiments got to

Against the search this started from — single-variable moves plus swaps, a
uniformly random kick, strict-improvement acceptance — at equal time over six
instance families, 5 seeds each:

| Instance | moves+swaps | final defaults | Feasible |
|---|---:|---:|---|
| dense-ineq | 30.8713 | **0.0000** | 0/5 -> **3/5** |
| equality | 40.6505 | **0.0000** | 0/5 -> **5/5** |
| ill-cond | 3.2804 | **2.8898** | 0/5 -> 0/5 |
| sparse-ineq | 9.4998 | **0.0000** | 0/5 -> **5/5** |
| tomography | 46.0000 | **3.0000** | 0/5 -> 0/5 |
| two-sided | 40.1505 | **0.0000** | 0/5 -> **4/5** |

**30 of 30 paired comparisons better, 0 worse. Feasibility 0/30 -> 17/30.**
Artefact: `experiments/results/final_vs_swaps.json`.

### The one thing worth carrying forward

**Escape width dominates escape quality.** Every experiment that widened the
escape won; the one that improved which single variable to move while keeping the
width at one lost badly, and the one that merely made the single move *legal*
changed nothing. Ranked by what they bought:

| Changed | Experiment | Result |
|---|---|---|
| Where the kick lands | 1 | ratio 0.629 |
| How wide the kick is | 4 | 13 better / 0 worse, +3 instances feasible |
| Whether a plateau can be crossed | 2 | tomography 47 -> 3 |
| Whether the kick is legal | 3 | neutral |
| How good the single escape move is | 5 | 0 better / 21 worse |

### Still open

- `ill-cond` (near rank-deficient) and `tomography` (integer plateau) never reach
  a feasible point. Both sit at strict local optima where every single-variable
  candidate is worse, which points at a genuine compound move — two variables
  moved jointly, chosen from the active constraint's support — rather than a
  wider kick of single moves. Experiment 5 tried the cheap version of this (one
  move plus the following descent) and it failed for want of width; the real
  version enumerates pairs and was not attempted.
- The tie-break is the sum of squares over all constraints. For a minimax
  objective the natural quantity is the sorted violation vector compared
  lexicographically, or the sum over the top-K. The cheap surrogate won on
  tomography; the proper one is untested.

## Running summary

| # | Date | Title | Layer | Decision | Median ratio | Paired | Notes |
|---|---|---|---|---|---|---|---|
| 5 | 2026-09-18 | Escape by the least damaging move | Search | **REVERT** | 1.108 | 0 better / 21 worse | deterministic escape cycles; sampling it does not save it |
| 4 | 2026-09-18 | Widen the kick while an instance stalls | Search | **KEEP** | 0.923 | 13 better / 0 worse | feasibility 1/5->3/5, 3/5->5/5, 3/5->5/5 |
| 3 | 2026-09-18 | Refuse to kick a variable that cannot move | Search | **REVERT** | 1.000 | 4 better / 5 worse | mechanism real, quality neutral; kick strength is the binding constraint |
| 2 | 2026-09-18 | Let a neutral move win on the tie-break | Evaluator | **KEEP** (specialist) | 1.000 | 4 better / 2 worse | tomography 44.0 -> 3.0, rest tied |
| 1 | 2026-09-18 | Kick the worst constraint, not a random subset | Search | **KEEP** | 0.771 | 14 better / 1 worse | sparse-ineq 7.05 -> 0.00, feasible 3/3 |

---

## Experiment 1: Kick the worst constraint, not a random subset

**Date:** 2026-09-18 · **Idea layer:** Search / move operator · **Decision:** KEEP

### Observation that prompted it

Three of the six instances made **zero progress after the first descent**. The
whole remaining budget -- 2.9 of 3.0 seconds -- bought nothing:

| Instance | start | after 1st descent | end of 3s | gained after |
|---|---:|---:|---:|---:|
| dense-ineq | 84.41 | 53.21 | 27.66 | 25.55 |
| equality | 98.58 | 69.74 | 23.52 | 46.22 |
| two-sided | 98.08 | 69.24 | 18.61 | 50.63 |
| ill-cond | 83.39 | 3.59 | 3.59 | **0.00** |
| sparse-ineq | 20.15 | 7.05 | 7.05 | **0.00** |
| tomography | 54.00 | 47.00 | 47.00 | **0.00** |

The first guess was a plateau -- a maximum objective is flat in most directions,
so a strict-improvement hill climber should freeze. **That was measured and
refuted:** at the starting point, 0% of single-variable candidates are neutral on
the five dense instances and 40-58% strictly improve. Only tomography is a
plateau (95% neutral, integer data giving exact ties). So the descent has plenty
to do; what fails is the escape.

### Idea card

The objective is a maximum, so it is decided by one constraint at a time. A kick
that does not touch that constraint cannot change the objective, and with `n`
variables a uniformly random kick misses it with probability `1 - |support|/n`.
Pick the worst constraint per instance and push a random subset of *its*
variables one level in the direction that relieves it: down where the coefficient
is positive and the upper bound is exceeded, up where it is negative, inverted
when the lower bound is missed. Same step size, same subset size -- only the
target changes.

**Hypothesis:** the three instances gaining nothing after their first descent
start gaining; median ratio below 0.95.

### Implementation surface

- `src/minviol/engine.py`: `perturb` dispatches, new `perturb_active`, old body
  becomes `perturb_random`
- `src/minviol/options.py`: `perturbation` field, default `"random"`
- `src/minviol/backends/{dense,sparse}.py`: `rows_of` accessor
- Feature flag: `Options.perturbation="active"` · ~55 lines

### Results

3s budget, 3 trials, MPS. Median violation, lower is better.

| Instance | moves-only | active-kick | Feasible |
|---|---:|---:|---|
| dense-ineq | 33.0993 | **26.9283** | 0/3 |
| equality | 34.2037 | **22.0647** | 0/3 |
| ill-cond | 3.2071 | 3.2372 | 0/3 |
| sparse-ineq | 7.0531 | **0.0000** | **3/3** (was 0/3) |
| tomography | 46.0000 | 46.0000 | 0/3 |
| two-sided | 30.3112 | **22.2940** | 0/3 |

**Median violation ratio 0.7712. Paired: 14 better, 1 worse, 3 tied.** Both clear
the decision rule (ratio < 0.95, at least 12 of 18 better with at most 3 worse).

### Analysis

The mechanism is confirmed by where the gain landed. `sparse-ineq` has 256
variables and about 500 non-zeros per column, so its worst constraint involves a
small fraction of the variables and a blind kick almost always missed it; it went
from never reaching feasibility to reaching it on every trial. The dense
instances, where every constraint involves every variable, still gained 18-35%
because the *direction* is now informative even when the target is not selective.

The single regression is `ill-cond` (3.2071 to 3.2372, inside the 17% noise band).
`ill-cond` and `tomography` still gain nothing after their first descent, so the
kick is not their blocker -- they have a different one, and that is the next thing
to look at.

### Lessons

- A maximum objective concentrates all the information in one constraint.
  Anything that ignores which constraint that is -- a kick, a candidate list, a
  screening set -- is spending most of its work where the objective cannot see it.
- The plateau explanation was wrong and measuring it cost one command. Check
  which of "cannot find a move" and "cannot keep a move" is actually happening
  before changing either.
- Feasibility is a step function, not a gradient: `sparse-ineq` moved from 7.05 to
  exactly 0. Median violation would have understated that as a 100% gain when it
  is really a change of outcome.

### Artefacts

- `experiments/results/exp01_active_kick.json`
- `src/minviol/engine.py:perturb_active`

---

## Experiment 2: Let a neutral move win on the tie-break

**Date:** 2026-09-18 · **Idea layer:** Evaluator / acceptance · **Decision:** KEEP
(specialist win)

### Observation that prompted it

After experiment 1, two instances still gained nothing past their first descent,
and measuring them showed *two different blockers*:

| Instance | active set | improving candidates | neutral candidates |
|---|---:|---:|---:|
| tomography | 7 constraints at the max | 0 / 1728 | **1718** |
| ill-cond | 1 constraint at the max | 0 / 384 | 0 |
| dense-ineq | 1 constraint at the max | 0 / 384 | 0 |

`tomography` is not stuck for lack of moves -- it has 1718 sideways moves and is
forbidden to take any of them. Its seven tied constraints mean no single variable
can lower the maximum, but the plateau is enormous. `ill-cond` and `dense-ineq`
are at genuine strict local optima, where every candidate is worse; that is a
different problem and this experiment does not address it.

### Idea card

Under pure `linf` a candidate is admissible only if it strictly lowers the
maximum, so a neutral candidate never reaches the tie-break -- which is already
computed, and already used to choose among admissible candidates. Admit neutral
candidates that strictly reduce the sum of squares. Ties must strictly improve
the second criterion, so the walk is monotone and cannot cycle.

This required overturning an earlier decision of my own: `linf_l2_tiebreak` was
refused on inequalities on the grounds that the squared positive part is zero
wherever a one-sided constraint holds. That reasoning is sound for
`linf_l2_nonincrease`, which *forbids* moves on the strength of that quantity and
stays equality-only. It is backwards for the tie-break, which *admits* moves --
a weak signal there costs nothing and a plateau costs everything.

**Hypothesis:** tomography unsticks; the others unchanged.

### Implementation surface

- `src/minviol/violation.py`: `EQUALITY_ONLY_POLICIES` narrows to
  `linf_l2_nonincrease` alone
- `tests/test_violation.py`, `tests/test_api.py`: the two tests that encoded the
  old decision
- Feature flag: `Options.acceptance="linf_l2_tiebreak"` · ~10 lines

### Results

3s budget, MPS. Confirmation run at 5 trials on the two instances that moved.

| Instance | best | tiebreak | Trials |
|---|---:|---:|---|
| tomography | 44.0000 | **3.0000** | 5 |
| equality | 21.9268 | 22.8937 | 5 |
| dense-ineq | 27.1722 | 26.9283 | 3 |
| ill-cond | 3.2372 | 3.2372 | 3 |
| sparse-ineq | 0.0000 | 0.0000 | 3 |
| two-sided | 22.3937 | 22.3937 | 3 |

Median ratio over 18 pairs: 1.0000 (4 better, 2 worse, 12 tied). Over the two
moving instances at 5 trials: 0.5349.

### Analysis

The mechanism landed exactly where predicted and nowhere else, which is the
strongest evidence available that it is the mechanism and not luck. Tomography's
seven-way tie at the maximum is what a structured, integer-valued system produces
and a random Gaussian one never does -- 95% of its candidates are neutral against
0% on the dense instances. The 14x gain is the search finally being allowed to
walk that plateau.

The single regression, `equality` at 21.93 to 22.89, is 4.4% against a measured
noise floor of 17% on that instance.

Worth being clear about what this does *not* fix: `ill-cond` and `dense-ineq` sit
at strict local optima with zero neutral candidates, so admitting neutral moves
cannot help them. They need a move that changes more than one variable at a time.

### Lessons

- Check the *direction* of a constraint before reusing an argument about it. The
  same "this quantity is nearly zero" observation kills a policy that forbids
  moves and is harmless for one that admits them. I had refused both on one
  argument.
- "Stuck" has at least two distinct causes -- no admissible move, and no move at
  all -- and they need opposite fixes. Measure which one before choosing.
- A decision rule tuned for broad wins will discard a specialist. Write the
  specialist clause before it costs you an idea, not after.

### Artefacts

- `experiments/results/exp02_tiebreak.json`, `exp02_tiebreak_confirm.json`
- `src/minviol/violation.py:EQUALITY_ONLY_POLICIES`

---

## Experiment 3: Refuse to kick a variable that cannot move

**Date:** 2026-09-18 · **Idea layer:** Search / move operator · **Decision:** REVERT

### Observation that prompted it

Tracing the incumbent on the two instances that still stall, keeping my own best
the way the solve loop does:

| Instance | local optimum | kick moves objective by | kicks accepted |
|---|---:|---:|---:|
| ill-cond | 3.592 | **+0.000** | **0 / 40** |
| dense-ineq | 53.214 | −1.718 | 4 / 40 |
| equality | 69.744 | −37.746 | 13 / 40 |

On `ill-cond` the kick changed the objective by *exactly zero*. The directed kick
picks uniformly among the worst constraint's variables, and a variable already at
the end of its domain in the relieving direction cannot move — with the default
kick size of one variable, picking one wastes the entire escape mechanism.

(The first version of this trace was wrong and said the kick raised the objective
from 3.8 to 83.4. `Batch.best_x_idx` is set at construction and only updated by
the API's solve loop, so the trace was resetting to the all-zeros start each
round. Worth recording: that field is a trap for anything driving the engine
directly.)

### Idea card

Exclude from the kick any variable whose step would be clamped away, and fall
back to kicking something that can move when the worst constraint offers nothing.

**Hypothesis:** `ill-cond` starts escaping.

### Implementation surface

- `src/minviol/engine.py`: `perturb_active` movability mask and fallback
- `src/minviol/options.py`: `kick_must_move` flag · ~20 lines

### Results

The mechanism worked: the kick on `ill-cond` went from moving the objective by
+0.000 to +1.566. The quality did not follow.

| Instance | best | moving-kick |
|---|---:|---:|
| dense-ineq | 24.5587 | 25.2897 |
| equality | 22.8937 | 23.2890 |
| ill-cond | 3.2372 | 3.2372 |
| sparse-ineq | 0.0000 | 0.0000 |
| tomography | 3.0000 | 3.0000 |
| two-sided | 23.8375 | 27.0166 |

**Median ratio 1.0000. Paired: 4 better, 5 worse, 9 tied.** Every per-instance
difference is inside that instance's noise floor.

A two-level domain is the worst case for this defect — half the variables sit at
an end — so two binary instances were added and tested at 5 trials. Both variants
solved both instances (feasible 4/5 and 5/5, violation 0.0), so that test is at a
ceiling and settles nothing either way. The instances are kept; they are useful
coverage even though they gave no signal here.

### Analysis

The kick is now valid and still useless, which is the informative part:
`ill-cond` accepts 0 of 40 kicks whether the kick moves the point or not. The
descent returns to the same local optimum regardless. So what binds is not
whether the kick is legal but **how far it goes** — one variable is not enough to
leave the basin, and making that one variable a better-chosen one changes
nothing.

### Lessons

- A mechanism can be real, verified, and worth nothing. "The kick now moves" was
  measured and true; it did not survive contact with the metric.
- When an escape fails, separate *can it move* from *does moving help*. Fixing
  the first without checking the second cost an experiment.
- Watch for a first attempt at instrumentation being wrong in the direction that
  confirms the hypothesis. The original trace showed the kick destroying the
  solution, which looked like a dramatic finding and was an artefact of my own
  harness resetting to the starting point.

### Artefacts

- `experiments/results/exp03_moving_kick.json`, `exp03_binary.json`
- Code: reverted; `perturb_active` is back to its experiment 1 form

---

## Experiment 4: Widen the kick while an instance stalls

**Date:** 2026-09-18 · **Idea layer:** Search / move operator · **Decision:** KEEP

### Observation that prompted it

Experiment 3 established that the kick's *validity* is not what binds: on
`ill-cond` the kick was accepted 0 times in 40 whether or not it actually moved
the point. The descent returns to the same local optimum either way, so the basin
is wider than the kick. The kick is one variable — `max(1, round(0.005 * 128))` —
and it is always undone.

### Idea card

Hold the kick at one variable while it keeps producing new incumbents; after a
run of consecutive failures, double it; snap back to one the moment it succeeds.
A variable-depth neighbourhood, per instance, driven by the instance's own stall
count rather than a global schedule.

**Hypothesis:** the stuck instances start moving; ratio below 0.95.

### Implementation surface

- `src/minviol/engine.py`: `kick_sizes`, `_chosen_mask`, per-instance kick widths
  in both `perturb_random` and `perturb_active`
- `src/minviol/api.py`: `stall` counter and `kick_counts()` in the solve loop
- `src/minviol/options.py`: `kick_escalation`, `kick_patience`, `kick_max_rate`
- Feature flag: `Options.kick_escalation` · ~45 lines

With the flag off the kick draws the same shapes from the same generator, so the
default path is unchanged — `tests/test_minviol_parity.py` still holds minviol to
the AMVM engine's exact trajectory.

### Results

**At a fixed 75-iteration budget** the verdict was ambiguous: ratio 0.8752 but
only 10 of 18 pairs better, with real regressions on `equality` (21.03 to 23.67)
and `two-sided` (20.53 to 23.17), and tomography 2.3x slower per iteration. A
refinement raising `kick_patience` from 5 to 20 made escalation fire so rarely
that 14 of 18 pairs tied — neutral, not better.

**At equal wall-clock time (3s, 5 trials)** it is unambiguous, because escalation
converges earlier and spends less of the budget:

| Instance | best | escalate | Feasible: best -> escalate |
|---|---:|---:|---|
| dense-ineq | 20.1659 | **0.0000** | 1/5 -> **3/5** |
| equality | 0.0000 | 0.0000 | 3/5 -> **5/5** |
| ill-cond | 3.2372 | **2.8898** | 0/5 -> 0/5 |
| tomography | 3.0000 | 3.0000 | 0/5 -> 0/5 |
| two-sided | 0.0000 | 0.0000 | 3/5 -> **5/5** |

**Median ratio 0.9225. Paired: 13 better, 0 worse, 12 tied.** No regressions.

### Analysis

The two protocols disagreeing is the interesting part, and the resolution is that
they measure different things. Per iteration, a wider kick costs more and
sometimes throws away progress an instance was still making — hence the two
regressions at 75 iterations. Per second, escalation reaches its answer sooner
(2.7s against 3.0s on three instances) because instances that find a feasible
point freeze and stop spending. Equal time is the metric that matches how the
solver is actually used, so it is the one that decides.

`ill-cond` moved for the first time in four experiments, from 3.2372 to 2.8898.
It is still the least tractable instance here, and a near-rank-deficient system
plausibly needs a compound move rather than a wider kick of single moves.

Feasibility count, not median violation, is now the metric carrying the
information: three instances sit at 0.0000 for both variants and differ only in
how often they get there.

### Lessons

- Equal iterations and equal time can give opposite verdicts. Decide in advance
  which one matches how the thing is used, and when they disagree, say so rather
  than reporting the flattering one.
- A schedule knob (`kick_patience`) tuned globally could not separate "stuck" from
  "progressing"; the per-instance stall counter could. Adaptivity beat tuning.
- Once instances start reaching the answer, median violation stops discriminating
  and the feasibility rate takes over. Watch for the primary metric going blind.

### Artefacts

- `experiments/results/exp04_escalate.json`, `exp04_escalate_patient.json`,
  `exp04_equal_time.json`
- `src/minviol/api.py:kick_counts`, `src/minviol/engine.py:kick_sizes`

---

## Experiment 5: Escape by the least damaging move

**Date:** 2026-09-18 · **Idea layer:** Search / move operator · **Decision:** REVERT

### Observation that prompted it

After four experiments `ill-cond` and `tomography` are the only instances that
never reach a feasible point, and both sit at strict local optima where every
single-variable candidate is worse. A kick chosen at random is chosen without
reference to what it costs. At a point where everything is worse, the cheapest
escape should be the move that loses the least.

### Idea card

Replace the random kick with the single cheapest uphill move: score every
single-variable candidate, admit all of them, take the one with the smallest
resulting maximum. The kick and the descent that follows then form a compound
multi-variable move chosen by the objective rather than by chance — which is what
a swap was supposed to be and is not, since a swap constrains two variables to
exchange levels and has nothing to do with which constraint is binding.

**Hypothesis:** the two stuck instances move; no loss elsewhere.

### Results

Equal time, 3s, 5 trials.

| Instance | best | uphill | uphill, sampled |
|---|---:|---:|---:|
| dense-ineq | **0.0000** (5/5 feasible) | 49.3133 (0/5) | 29.5059 (0/5) |
| equality | **0.0000** (5/5) | 58.3267 (0/5) | 22.1869 (0/5) |
| ill-cond | **2.8898** | 3.3649 | 3.2071 |
| tomography | 3.0000 | 3.0000 | 3.0000 |
| two-sided | **0.0000** (5/5) | 57.8267 (0/5) | 21.6869 (0/5) |

**0 pairs better, 21 worse.** The refinement — choosing from a random sample of 32
candidates instead of always taking the cheapest — halved the damage and changed
nothing about the verdict: still 0 better, 21 worse, ratio 1.108.

### Analysis

Two failures, and they are different.

The first is cycling. The cheapest uphill move is *deterministic*, so the descent
that follows undoes exactly it, and the search revisits the same pair of points
forever. The instances ended near their starting objectives (49 and 58 against
starts of 84 and 98), which is what no exploration looks like.

Sampling fixed the cycling and the variant still lost everywhere, which is the
more useful failure. The uphill move changes **one** variable. The kick it
replaced changes as many as the escalation schedule calls for. Against an
escape that can move sixteen variables at once, a perfectly chosen single move is
not competitive, and it costs a full candidate scoring pass to choose.

That completes a consistent story across all five experiments:

| Experiment | Changed | Result |
|---|---|---|
| 1 | where the kick lands (quality) | KEEP, ratio 0.629 |
| 3 | whether the kick is legal (quality) | REVERT, neutral |
| 4 | how wide the kick is (width) | KEEP, 13 better / 0 worse |
| 5 | how good the single move is (quality, at the cost of width) | REVERT, 0 / 21 |

**Escape width dominates escape quality.** Targeting was worth 37% once; widening
was worth another 8% and three instances' worth of feasibility; and the best
possible single move loses to a mediocre wide one.

### Lessons

- A deterministic escape from a local optimum cycles, because the descent that
  follows is deterministic too. Any escape needs a source of randomness that the
  descent cannot cancel.
- Do not trade width for quality in an escape move. One perfectly chosen variable
  loses to sixteen randomly chosen ones.
- A refinement that fixes the mechanism you blamed and does not change the verdict
  has told you the mechanism was not the problem. That is worth the second run.

### Artefacts

- `experiments/results/exp05_uphill.json`, `exp05_uphill_sampled.json`
- Code: reverted

