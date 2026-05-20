# Methodology

## Question

A promotion can increase purchases and still lose money. PriceSense asks a more
useful question than "who is likely to buy?": which customers are worth
contacting when the promotion itself has a cost?

## Experiment

The source data come from a randomized promotion experiment. Random assignment
means the difference in purchase rates between promotion and control estimates
the average causal effect of the promotion.

The project reports:

- promotion and control purchase rates;
- incremental response rate (IRR);
- a two-sided z-test for the difference in proportions;
- a bootstrap 95% confidence interval for IRR;
- expected net incremental revenue (NIR).

For a selected customer group of size `N`, the project uses:

`expected NIR = N * (10 * IRR - 0.15)`

where $10 is the product revenue and $0.15 is the cost of sending one
promotion in the original exercise.

## Targeting models

Two deliberately simple strategies are compared.

### Response model

A random forest is trained only on promoted customers to estimate purchase
probability after receiving a promotion. This represents the common approach
of targeting people who look most likely to respond.

### Uplift S-learner

A gradient-boosted classifier is trained on customer features plus the treatment
indicator. For each customer it predicts two counterfactual probabilities:

- probability of purchase if promoted;
- probability of purchase if not promoted.

Their difference is used as an estimated uplift score.

## Validation design

The experiment is split 60/20/20 into train, validation and test sets while
preserving treatment/outcome proportions.

Each model selects its targeting fraction on the validation set by maximizing
estimated NIR over a small grid from 10% to 100% of customers. The selected
fraction is then evaluated once on the held-out test set.

This matters because choosing the budget directly on the test set would make
the reported policy value optimistic.

## Interpretation

The project does not assume that a more sophisticated causal model must win.
The purchase outcome is rare, so individual treatment-effect estimates are
noisy. The holdout comparison is therefore used to judge whether heterogeneous
uplift is stable enough to justify a more complex targeting strategy.
