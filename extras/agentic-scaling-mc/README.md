# Extra: agentic scaling Monte Carlo (follow-up study, work in progress)

A separate model asking when a verified swarm of small models beats one frontier model at equal cost.
Not part of the paper's results.

```bash
python3 agentic_scaling_mc.py   # prints results, writes mc.npz and cur.npy
python3 fig.py                  # writes agentic_mc.png
```

Headline (100,000 scenarios, assumed priors): the swarm wins 84% of scenarios when verifier accuracy exceeds the frontier model's per-step accuracy, and 6% otherwise. Self-improving the generator alone barely helps; improving the verifier too raises the swarm's win share from 12% to 68% over 20 rounds.
