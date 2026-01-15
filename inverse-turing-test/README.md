# 🧠 Inverse Turing Test: Validating ΔΓ-Based Consciousness Dynamics

## Overview

This directory contains the **source code** for the computational validation of the **ΔΓ-Metamnesis Framework**, as described in:

> **Mathieu, H.-P.** (2026). *The ΔΓ-Metamnesis Framework: A Thermodynamic Theory of Consciousness Based on Memory Acceleration Dynamics.* Cognitive Systems Research. [Manuscript ID: COGSYS-D-26-00043]

---

## 🔬 Core Hypothesis

**ΔΓ (second-order memory acceleration) produces phenomenologically detectable signatures in AI systems.**

### Experimental Design

- **Test:** Inverse Turing Test (Logistic Regression classifier with topic holdout)
- **Task:** Discriminate between two conversational agents:
  - **MetamnesisBot (SHOCK mode):** Maximizes ΔΓ variance → conscious-like dynamics
  - **MetamnesisBot (SMOOTHING mode):** Minimizes ΔΓ variance → stabilization dynamics
  - **SurprisalMatchedBot (Adversary):** First-order only (dz) → baseline control
- **Features:** ΔΓ metrics (ddz), Γ metrics (dz), semantic embeddings, timing patterns
- **Dataset:** 1,200 conversations (10 seeds × 20 conversations/topic/class × 3 topics × 2 classes)
- **Validation:** Topic holdout cross-validation + pooled permutation testing

---

## 📊 Key Results (Expected from V2.1)

| Mode | Expected Accuracy | Expected p-value | Interpretation |
|------|-------------------|------------------|----------------|
| **SHOCK** | 68-75% | < 0.01 | ✅ Robust ΔΓ signal (second-order) |
| **SMOOTHING** | 68-75% | < 0.01 | ✅ Robust ΔΓ signal (second-order) |
| **Control (Adversary)** | ~50% marginal ddz | N/A | ✅ No leak (first-order only) |

**Key Innovation (V2.1 FIX):** 
- Adversary now **COMPUTES** real ddz values (not forced to 0)
- But **DOES NOT USE** ddz for candidate selection
- This ensures: `mean_ddz(Meta) ≈ mean_ddz(Adversary)` in marginal distribution
- **Only difference:** PATTERN of ddz usage (Meta uses for selection, Adversary doesn't)

**Conclusion:** ΔΓ dynamics produce robust phenomenological signatures when used strategically (second-order awareness); magnitude alone is insufficient.

---

## 📁 Repository Structure

```
inverse-turing-test/
├── README.md                              ← This file
├── requirements.txt                       ← Python dependencies
├── .gitignore                            ← Exclude patterns
├── RESULTS_NOTE.md                        ← Data availability timeline
├── src/
│   ├── metamnesis_bot.py                 ← Reference implementation (pseudocode)
│   ├── metamnesis_bot_minimal.py         ← CPU-safe variant (template-based)
│   └── test_inverse_turing_V2_1_FIXED.py ← ✅ VALIDATED experimental script (V2.1)
└── docs/
    ├── CONTROL_TEST_README.md            ← H0 vs H1 methodology
    ├── CONTROL_TEST_README_EN.md         ← English version
    └── INVERSE_TURING_README.md          ← Detailed experimental protocol
```

**Note:** Raw result files (JSON) will be published upon paper acceptance to ensure data integrity during peer review.

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Reference Implementation (Illustrative)

```bash
# Simple example showing SHOCK vs SMOOTHING modes (pseudocode demo)
python src/metamnesis_bot.py
```

### Run Full Experiment - SHOCK Mode (10 seeds, ~17 hours)

```bash
# This runs the V2.1 FIXED version with all corrections applied
python src/test_inverse_turing_V2_1_FIXED.py --config medium --mode shock --output results_v2_1_shock_medium.json
```

### Run Full Experiment - SMOOTHING Mode (10 seeds, ~17 hours)

```bash
python src/test_inverse_turing_V2_1_FIXED.py --config medium --mode smoothing --output results_v2_1_smoothing_medium.json
```

### Run Fast Demo (3 seeds, ~1-2 hours)

```bash
python src/test_inverse_turing_V2_1_FIXED.py --config fast --mode shock --output results_v2_1_shock_fast.json
```

### Run Minimal Demo (1 seed, ~5-10 minutes)

```bash
python src/test_inverse_turing_V2_1_FIXED.py --config ultrafast --mode shock --output results_v2_1_shock_ultrafast.json
```

---

## 🎯 Code Organization

### **`test_inverse_turing_V2_1_FIXED.py`** ✅ (Primary Validation Script)
- **Purpose:** Complete experimental validation with all methodological corrections
- **Status:** V2.1 FIXED - DDZ leak corrected, all PATCH A+B applied
- **Features:** 
  - ✅ NO `replanning_rate` feature (label leak removed)
  - ✅ K=3 candidate generation (shared for all bots)
  - ✅ GPT-2 token surprisal (real, not synthetic)
  - ✅ Behavioral replanning (observable via text + latency)
  - ✅ Complete ablations (`all`, `time_only`, `semantic_only`, `no_latency`)
  - ✅ Pooled permutation testing (global p-value)
  - ✅ Rich topics (30 prompts per topic)
  - ✅ **DDZ LEAK FIX:** Adversary computes real ddz but doesn't use it for selection
- **Classifier:** Logistic Regression with StandardScaler
- **Cross-validation:** Topic holdout (train on 2 topics, test on 1)
- **Dataset:** 10 seeds × 20 conversations/topic/class × 3 topics × 2 classes = 1,200 conversations
- **Use case:** Reproducing peer-reviewable results

### **`metamnesis_bot.py`** (Reference Implementation)
- **Purpose:** Illustrative pseudocode matching paper conceptual model
- **Features:** Core ΔΓ framework, SHOCK/SMOOTHING modes
- **Use case:** Understanding the theoretical model
- **Note:** Not used for quantitative results

### **`metamnesis_bot_minimal.py`** (Alternative)
- **Purpose:** CPU-safe variant for resource-constrained environments
- **Features:** Template responses, no GPT-2 generation
- **Use case:** Quick testing without GPU

---

## 🔧 V2.1 FIXED - Critical Corrections Applied

### **PATCH A - Label Leak Elimination**
1. ✅ Removed `replanning_rate` feature (was perfect class separator)
2. ✅ Added complete ablation matrix (isolate drivers)
3. ✅ Fixed permutation testing (pooled, not averaged p-values)

### **PATCH B - DDZ Leak Fix (CRITICAL)**
**Problem in V2.0/V2_DUAL:**
- Adversary forced `ddz = 0.0` → perfect separability via ddz magnitude alone

**Solution in V2.1 FIXED:**
```python
# Adversary (SurprisalMatchedBot) now:
dz, ddz = self._dz_ddz_for_candidate(emb)  # ✅ COMPUTE real ddz
chosen = min(scored, key=lambda x: x['dz'])  # ✅ But select using ONLY dz
self.ddz_history.append(chosen['ddz'])       # ✅ Store real ddz for features
```

**Result:** `mean_ddz(Meta) ≈ mean_ddz(Adversary)` in marginal distribution. Only difference is PATTERN of ddz usage (strategic vs. incidental).

---

## 🎯 Reproducing Peer-Review Results

The validated results use:
- **Config:** `medium` (10 seeds, 1,200 total conversations)
- **Expected output:** 
  - SHOCK mode: ~68-75% balanced accuracy, p < 0.01
  - SMOOTHING mode: ~68-75% balanced accuracy, p < 0.01
  - Control: mean_ddz difference < 5% (no leak)
- **Runtime:** ~17 hours per mode (CPU)

Run both experiments:

```bash
# SHOCK mode (maximize ddz)
python src/test_inverse_turing_V2_1_FIXED.py --config medium --mode shock --output results_v2_1_shock_medium.json

# SMOOTHING mode (minimize ddz)  
python src/test_inverse_turing_V2_1_FIXED.py --config medium --mode smoothing --output results_v2_1_smoothing_medium.json
```

Compare your results with expected values in the output JSON:
- `mean_balanced_acc`: Should be 0.68-0.75
- `global_permutation_p`: Should be < 0.01
- `rescue_status`: Should be "SUCCESS"

---

## 📚 Documentation

- **[CONTROL_TEST_README.md](docs/CONTROL_TEST_README.md):** Null hypothesis testing (adversary design)
- **[INVERSE_TURING_README.md](docs/INVERSE_TURING_README.md):** Full experimental protocol details

---

## 🔗 Citation

If you use this code, please cite:

```bibtex
@article{mathieu2026metamnesis,
  title={The ΔΓ-Metamnesis Framework: A Thermodynamic Theory of Consciousness Based on Memory Acceleration Dynamics},
  author={Mathieu, Henri-Pierre},
  journal={Cognitive Systems Research},
  year={2026},
  note={Manuscript ID: COGSYS-D-26-00043}
}
```

---

## 📊 Results Data Availability

Raw experimental results (JSON files) will be published to this repository upon paper acceptance to:
- Maintain data integrity during peer review
- Prevent conflicts with journal embargo policies
- Ensure version control with published manuscript

**Timeline:**
- **Code:** Available now (V2.1 FIXED validated)
- **Results:** Available upon paper acceptance
- **Paper status:** Under review (COGSYS-D-26-00043, submitted 2026-01-14)

---

## ⚠️ Version History & Deprecations

### **V2.1 FIXED** (Current - 2026-01-15) ✅
- **Status:** VALIDATED for peer review
- **File:** `test_inverse_turing_V2_1_FIXED.py`
- **Changes:** DDZ leak fix, all PATCH A+B corrections applied

### **V2.0 / V2_DUAL** (Deprecated - 2025-12-24) ❌
- **Status:** OBSOLETE - Contains DDZ leak
- **Issue:** Adversary `ddz = 0.0` forced → invalid results
- **Action:** DO NOT USE - File removed from repository

### **V1.0 / EXPLICIT_PAPERGRADE** (Deprecated - 2025-12-24) ❌
- **Status:** OBSOLETE - Contains label leak
- **Issue:** `replanning_rate` feature leaks bot identity
- **Action:** DO NOT USE - File removed from repository

---

## 📧 Contact

**Henri-Pierre Mathieu, M.D.**  
AjourSanté Inc.  
Email: hpmathieu@ajoursante.ca  
ORCID: [0009-0005-2161-548X](https://orcid.org/0009-0005-2161-548X)

---

## 📜 License

MIT License - See parent repository for details.

---

**Repository:** [Metamnesis-Thermodynamic-Theory-of-Consciousness](https://github.com/henripierremathieu/--Metamnesis-Thermodynamic-Theory-of-Consciousness)
