# Warm-start QAOA status

Status: not implemented in Step 7.5.

Reason: a valid warm-start comparison needs a declared continuous relaxation,
a mapping from its fractional solution to the initial quantum state, and the
matching modified mixer. Using the exact brute-force route as an initial state
would reveal the ground truth and invalidate the comparison. Q-Rescue therefore
reports standard X and feasible XY methods only and keeps warm start as a later
controlled experiment.

