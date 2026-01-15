# Results Data Availability

Raw experimental results will be published upon paper acceptance.

**Current status:**
- Paper under review: COGSYS-D-26-00043 (submitted 2026-01-14)
- Code available for reproduction: **V2.1 FIXED** (all corrections applied)
- Script: `src/test_inverse_turing_V2_1_FIXED.py`

**Expected results (V2.1 FIXED with DDZ leak corrected):**
- **SHOCK mode:** ~68-75% balanced accuracy, p < 0.01 (robust ΔΓ signal)
- **SMOOTHING mode:** ~68-75% balanced accuracy, p < 0.01 (robust ΔΓ signal)
- **Control validation:** mean_ddz(Meta) ≈ mean_ddz(Adversary) ± 5% (no leak)

**Key correction (V2.1):**
- Adversary now COMPUTES real ddz values (not forced to 0)
- But DOES NOT USE ddz for selection (only dz)
- This ensures marginal ddz distributions match, with only PATTERN differences

**Timeline:**
- Code: ✅ Available now (V2.1 FIXED validated 2026-01-15)
- Results JSON files: Upon paper acceptance
- Deprecated versions (V2.0/V2_DUAL, V1.0/PAPERGRADE): ❌ Removed (methodological flaws)

**Run experiments yourself:**
```bash
# SHOCK mode (10 seeds, ~17h)
python src/test_inverse_turing_V2_1_FIXED.py --config medium --mode shock --output results_v2_1_shock_medium.json

# SMOOTHING mode (10 seeds, ~17h)
python src/test_inverse_turing_V2_1_FIXED.py --config medium --mode smoothing --output results_v2_1_smoothing_medium.json

# Fast demo (3 seeds, ~1-2h)
python src/test_inverse_turing_V2_1_FIXED.py --config fast --mode shock --output results_v2_1_shock_fast.json
```
