# Log-Failure Evidence-Cap Sensitivity

## Scope

This experiment evaluates whether RuleDep's log-failure evidence cap materially changes dependency mining. It performs dependency mining and descriptive statistics only; it does not retrain the Stage-2 aggregation model and does not run link prediction.

The evaluated datasets are KG20C, WN18RR, Codex-M, FB15k-237, Codex-L, and YAGO3-10. The tested settings are `cap = 5, 6, 7, 8, 9`, and `no-cap`.

## Evidence definition and safe no-cap mode

For a rule confidence `c`, RuleDep uses

```text
e(c) = -log(1-c)
g_ij = e_ij - e_i - e_j
```

Dependency mining uses smoothed confidence

```text
c = support / (bodySize + 3).
```

Therefore, smoothed confidence is strictly below one. In `no-cap` mode, the implementation computes uncapped smoothed log-failure evidence with `-log1p(-c)`. Confidence is additionally bounded by the largest floating-point value below one so that the unused raw-confidence diagnostic cannot introduce `Inf` or `NaN`. Thus, `no-cap` tests uncapped smoothed evidence rather than an unrelated numerical failure at raw confidence one.

## Experimental settings

All settings other than the evidence cap were held fixed:

- dependency formula: `unified`;
- unseen-negative examples: 3;
- minimum absolute dependency gain: 0.01;
- `TOP_K`: 500;
- minimum support: 2 for KG20C and WN18RR, and 5 for the other datasets;
- the same training triples and AnyBURL rule files for every cap;
- dependency mining only, without Stage-2 training or evaluation.

The command is:

```bash
bash script/run_evidence_cap_sensitivity.sh
```

Completed mining outputs are under `dependency_runs/<dataset>/cap_<setting>/`. The summary script processes at most two datasets concurrently and recycles its worker after each dataset to bound memory use.

One implementation detail should be considered when interpreting very small differences. Binary-rule grounding uses dynamic EDIS sampling, including randomized entity ordering. Each cap was mined in a separate run, so residual differences at approximately the 0.1% level or below may contain run-to-run sampling variation. In particular, changes that occur when no evidence value reaches the tested cap should not be attributed solely to the cap.

## Output files

- `cap_dependency_stats.csv`: complementary, redundant, and total retained dependencies.
- `cap_gain_distribution.csv`: mean, median, P95, P99, P99.9, and maximum absolute dependency gain.
- `cap_overlap_vs_7.csv`: set overlap, cap-7 retention, Spearman gain correlation, and sign changes relative to cap 7.
- `cap_saturation_stats.csv`: singleton and retained joint evidence values reaching each cap.

All four files contain 36 rows: six datasets by six cap settings.

## Results

The table below summarizes the stability of caps 5--9 and compares the no-cap gain tail with cap 7. The dependency-count span is `(maximum - minimum) / count_at_cap_7`. The overlap, correlation, and sign-change columns report the most conservative value among caps 5, 6, 8, 9, and no-cap relative to cap 7.

| Dataset | Cap 5--9 count span | Min. Jaccard vs. 7 | Min. gain Spearman | Max. sign-change rate | No-cap / cap-7 P99.9 | No-cap / cap-7 max |
|---|---:|---:|---:|---:|---:|---:|
| KG20C | 0.0001% | 0.999999 | 0.999999 | 0.0000% | 1.000000 | 1.000000 |
| WN18RR | 0.0378% | 0.998836 | 0.999521 | 0.0227% | 1.000000 | 0.997948 |
| Codex-M | 0.0005% | 0.999877 | 0.999952 | 0.0095% | 1.000077 | 1.000000 |
| FB15k-237 | 0.0011% | 0.999381 | 0.999914 | 0.0227% | 0.999807 | 1.000000 |
| Codex-L | 0.0019% | 0.999448 | 0.999741 | 0.0393% | 1.000228 | 1.032140 |
| YAGO3-10 | 0.0007% | 0.999914 | 0.999969 | 0.0037% | 1.000000 | 1.000000 |

### Q1. Does no-cap produce more extreme dependency gains?

No systematic extreme-gain inflation is observed. Relative to cap 7, the no-cap P99.9 absolute gain changes by at most 0.023% across the six datasets. The maximum is identical on KG20C, Codex-M, FB15k-237, and YAGO3-10; it is 0.21% lower on WN18RR and 3.21% higher on Codex-L. The Codex-L no-cap maximum, 6.67514, is also obtained with caps 8 and 9, while its P99 and P99.9 remain effectively unchanged. Therefore, the experiment does not show a broader or systematically inflated no-cap tail.

This result is consistent with unseen-negative smoothing: near-perfect smoothed confidence is extremely rare in these rule sets. It does not imply that saturation can never be useful on a different rule distribution.

### Q2. Are caps 5--9 stable?

Yes. Across caps 5--9, the largest retained-dependency count span is 0.0378% on WN18RR; it is at most 0.0019% on every other dataset. Against cap 7, Jaccard overlap is at least 0.998836, common-dependency gain Spearman correlation is at least 0.999521, and the largest sign-change rate is 0.0393%. These differences are small enough to conclude that the results are insensitive to the exact threshold within 5--9. Given the randomized dynamic grounding noted above, these tiny residual differences are an upper bound on the cap's observable effect rather than clean evidence that the cap caused every changed dependency.

### Q3. How many rules does cap 7 actually affect?

At cap 7, only one singleton evidence value among 3,751,309 rule records reaches the cap (0.0000267%); it occurs on FB15k-237. Only two retained joint evidence values among 225,271,138 retained dependencies reach the cap (0.000000888%); both occur on Codex-L. The other five datasets have no saturated singleton evidence at cap 7, and the other five have no saturated retained joint evidence.

Even cap 5 affects only 11 singleton records and 54 retained joint dependencies across all six datasets. Thus, saturation acts only on the extreme-confidence tail and does not alter ordinary rule evidence in these experiments.

## Conclusion

For these six datasets, unseen-negative smoothing already prevents practically relevant near-one-confidence amplification: no-cap does not produce systematic P99/P99.9/max gain inflation, and cap 7 directly affects only one singleton and two retained joint evidence values. Consequently, the present results do not show that a cap is empirically necessary for these datasets. They do show that keeping a cap is a low-impact robustness safeguard for exceptional near-perfect-confidence cases.

The results are highly stable throughout caps 5--9, so 7 should be described as a practical saturation threshold within a broad stable range, not as a finely tuned or theoretically optimal constant.
