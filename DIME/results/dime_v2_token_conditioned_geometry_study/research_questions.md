# Research Questions

Q1. Is token-conditioned residual geometry more compact than a label-agnostic baseline at matched storage? Mean distortion difference (conditioned minus agnostic) = 7.17051187422336, 95% CI [2.621006213193699, 11.8592883570959] over 7 clusters.

Q2. Does that geometric compactness translate into better all-member predictive fidelity? Mean TV difference = 0.016158082119944194, 95% CI [0.0009923192237592038, 0.034624546207846274].

Q3. Does it translate into better top-m truncated predictive fidelity (the candidate-selection-relevant target)? Mean TV difference = 0.010356282469941112, 95% CI [-0.0011015291317840292, 0.024027885048407416].

Q4. Are candidate-set fidelity and candidate-weight fidelity genuinely distinct quantities (per S5)? Yes, by construction, verified synthetically -- reported separately throughout. Candidate-recall difference = 0.016007538109333187, 95% CI [0.002745888789440809, 0.03887167071152734].

Q5. Is predictive importance of individual components predictable from naive frequency alone? Mean correlation between log weight and leave-one-out TV impact across sampled clusters = 0.587255797353877.

Q6. Does any representation beat the static-histogram zero-change baseline on query-change preservation? Conditioned beats the static baseline on 0.9824999999999999 of query pairs on average; agnostic on 0.9924999999999999 -- see final_report.md for discussion of this project's recurring finding.

Q7-Q10 (capacity allocation, rare-token preservation, adaptive allocation, overall verdict). See capacity_allocation_results.csv and final_report.md's outcome classification (C).

