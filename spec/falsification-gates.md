# Falsification Gates — Operational Specification

Expansion of PREREGISTRATION.md §5–§6 into executable decision procedures.
Frozen 2026-08-25T01:54:08Z together with PREREGISTRATION.md
(sha256 b5f108ebad0465e7599a39726b301210946bc6198fc7054dfba615652108721f).
If this document and PREREGISTRATION.md ever disagree, PREREGISTRATION.md wins.

## Gate H1 — boundedness without loss of coercivity or accuracy

Population: every closed-form bounded candidate F2a–F6b (11 variants), each
run on all T1 cells $(\varepsilon, n, k)$, solvers pooled (per-cell agreement
required).

Falsified iff ANY of:
1. **Rate failure**: exists $(\text{variant}, \varepsilon, k)$ such that the
   energy-error slope $p$ over $n\in\{8,\dots,128\}$ violates
   $R^2\ge0.98$ or $|p-k|>0.1k$, where the same configuration passes for
   F1 at $\varepsilon=0.5$ (i.e., the failure is attributable to the
   candidate, not to global infrastructure).
2. **Coercivity collapse**: $\gamma_h<10^{-8}$ at $n=128$ for any
   $\varepsilon\ge10^{-6}$, on raw $\lambda_{\min}$ as preregistered.
   Mass-normalized values are reported alongside but do not alter the gate.

Survivor set $\mathcal{S}_{H1}$ = variants failing neither condition.

## Gate H2 — aggregation dominance below $\varepsilon_c$

Comparator: best-performing element of $\mathcal{S}_{H1}$ by median T1
$\log_{10}\kappa$ over sliver cells ($\varepsilon\le10^{-3}$), denoted
$\Phi^\*$. Against F7 (hybrid with aggregation).

Define per $(n,\varepsilon)$ cell the ratio
$r=\kappa(\Phi^\*)/\kappa(F7)$. H2 (aggregation is *necessary*) is SUPPORTED
iff $r>5$ for at least one tested $\varepsilon\le10^{-3}$ with Wilcoxon
signed-rank across mesh sizes $p<0.05/3$. H2 is FALSIFIED iff $r\le5$ at
every $\varepsilon\in\{10^{-2},10^{-3},10^{-4},10^{-5},10^{-6}\}$.

## Gate H3 — ghost-penalty non-redundancy

Arms: $\Phi^\*$ alone, GP alone (F1+$s_h$), $\Phi^\*$+GP. Non-redundancy of
the combination is established iff, per $\varepsilon$,
$\min_n \kappa_{\text{single}}/\kappa_{\text{combo}}>1.05$ where
$\kappa_{\text{single}}=\max(\kappa_{\Phi^\*},\kappa_{GP})$, with Wilcoxon
$p<0.05/3$ across cells. Otherwise H3 is FALSIFIED (redundant mechanisms).

Measurement noise floor: duplicated solves of 20 random T1 cells (seeded)
define relative solver spread; differences inside twice that spread count as
noise and cannot support either gate direction.

## Reporting contract

Every verdict line quotes: preregistration hash, gate, statistic value,
p-value, effect size, N, and the Parquet row predicate selecting the data.
Claims without a row predicate are inadmissible in the report.
