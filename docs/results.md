# Results

The randomized experiment contains 84,534 customers split almost evenly between
promotion and control.

The promotion increased purchase probability from about 0.76% to 1.70%, an
incremental response of about 0.95 percentage points. The effect is highly
statistically significant, but sending the promotion to everyone has negative
expected net incremental revenue because the $0.15 contact cost is paid for
every targeted customer.

A response model and an uplift S-learner were then compared under the same
train/validation/test protocol. The response model selected a 40% targeting
budget on validation and produced positive NIR on the held-out test set. The
uplift model selected 30% on validation but transferred less reliably to the
holdout sample.

This is the central result of the project: statistical significance at the
campaign level does not automatically imply a profitable blanket policy, and a
causal targeting model should only be preferred when its heterogeneous effects
are stable out of sample.
