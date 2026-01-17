# HAGD v31 Evidence Closure Results

Generated: 2026-08-10T03:02:14.943961+00:00

## Configuration

- Model: gpt2
- Task: modular_arithmetic (mod 23)
- Seeds: [0, 1, 2, 3, 4]
- Pipeline: v30-frozen

## 5-Seed Results (EXPLORATORY)

| Metric | Mean | Std | 95% CI | n |
|--------|-----:|----:|--------|--:|
| phi_uniform | 0.0000 | nan | (nan, nan) | 1 |
| phi_generation | 0.2727 | nan | (nan, nan) | 1 |
| phi_first_step | 0.3295 | nan | (nan, nan) | 1 |
| full_accuracy | 0.8381 | nan | (nan, nan) | 1 |

## Unsupported Claims

| Claim | Status |
|-------|--------|
| 91% (+/- 2.3%) behavioral preservation | UNSUPPORTED |
| Modulus 113 task | UNSUPPORTED |
| Symmetrization formula A_sym = 0.5*(A + A^T) | INCORRECT |
| Integrated Gradients attribution | MISLEADING |
| 49-347 circuit nodes across models | UNSUPPORTED |
| 52%-82% cross-architecture transfer | UNSUPPORTED |

## Limitations

1. All results are for GPT-2 Small (124M) on modular arithmetic mod 23
2. No seed produces CIRCUIT_VALIDATED=True under position-uniform intervention
3. Large-model claims have no executable evidence
4. The 91% headline figure has no provenance in this artifact