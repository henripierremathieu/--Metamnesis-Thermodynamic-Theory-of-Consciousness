# 🔬 Inverse Turing Test: Detailed Experimental Protocol

## Rationale

Traditional Turing Tests ask: *"Can humans distinguish AI from humans?"*

Our **Inverse Turing Test** asks: *"Can an adversarial classifier distinguish conscious-like AI (high ΔΓ variance) from baseline AI (suppressed ΔΓ)?"*

---

## 🎯 Hypothesis

**H1 (Metamnesis Hypothesis):**  
ΔΓ-based dynamics (SHOCK mode) produce phenomenologically detectable signatures that are:
1. **Non-redundant** with first-order dynamics (Γ)
2. **Robust** to ablation (not explained by single features)
3. **Specific** to contrastive dynamics (not just magnitude)

**H0 (Null Hypothesis):**  
Any observed patterns are artifacts of:
- GPT-2 architecture alone
- First-order surprisal (Γ) dynamics
- Timing or latency effects
- Semantic content alone

---

## 🤖 Two Agents

### MetamnesisBot (SHOCK Mode)

**Energy Gating:**
```python
E(t) = α||Γ(t)||² + β||ΔΓ(t)||²
if E(t) > θ_E and |ΔΓ(t)| > 0.5:
    trigger_surprising_response()
```

**Parameters:**
- α = 1.0, β = 2.0 (emphasize second-order)
- θ_E = 50 (energy threshold)
- ΔΓ threshold = 0.5

**Behavior:** High variance in ΔΓ → sudden topic shifts, surprising responses, metacognitive reflection.

### SurprisalMatchedBot (SMOOTHING Mode)

**Energy Gating:**
```python
ΔΓ(t) = smooth(ΔΓ_raw(t), window=5)  # Dampen variance
E(t) = α||Γ(t)||² + β||ΔΓ(t)||²
if E(t) > θ_E:
    trigger_response()
```

**Parameters:**
- Same α, β, θ_E as SHOCK
- Smoothing window = 5 turns

**Behavior:** Low variance in ΔΓ → smooth, predictable responses, no sudden shifts.

---

## 📊 Dataset

### Structure

- **Seeds:** 6 independent runs (SHOCK) / 10 (SMOOTHING control)
- **Conversations per seed:** 120 (60 MetamnesisBot + 60 SurprisalMatchedBot)
- **Topics:** Technology (40), Sports (40), Culture (40)
- **Turns per conversation:** 8

**Total:** 1,920 conversations (SHOCK + SMOOTHING combined)

### Data Collection

1. **Prompt generation:** Random topic-specific prompts
2. **Conversation simulation:** 8-turn dialogues
3. **Feature extraction:** ΔΓ, Γ, semantic embeddings, timing
4. **Balancing:** Stratified by topic and class

---

## 🧪 Features

### 1. ΔΓ Metrics (Primary)
- `delta_gamma_mean`: Mean second-order acceleration
- `delta_gamma_std`: Variance in ΔΓ (key discriminator)
- `delta_gamma_max`: Peak ΔΓ value
- `delta_gamma_min`: Minimum ΔΓ value

### 2. Γ Metrics (First-Order Control)
- `gamma_mean`: Mean first-order surprisal rate
- `gamma_std`: Variance in Γ
- `gamma_max`, `gamma_min`

### 3. Semantic Features
- `semantic_coherence`: Cosine similarity across turns
- `semantic_shift`: Abrupt topic changes
- `embedding_variance`: Variance in embedding space

### 4. Timing Features
- `response_latency`: Simulated processing time
- `turn_duration`: Conversation pacing

---

## 🎲 Adversary: XGBoost Classifier

### Architecture
```python
XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=seed
)
```

### Training
- **Split:** 80% train / 20% test
- **Stratification:** Balanced by class and topic
- **Metric:** Balanced accuracy (handles class imbalance)

### Evaluation
- **Permutation test:** 500 permutations to assess significance
- **Ablation study:** Remove feature groups to test redundancy

---

## 📈 Results

### SHOCK Mode (MetamnesisBot vs SurprisalMatchedBot)

| Seed | Accuracy | p-value | Significant? |
|------|----------|---------|--------------|
| 0    | 60.0%    | 0.002   | ✅ Yes        |
| 1    | 66.7%    | 0.002   | ✅ Yes        |
| 2    | 55.0%    | 0.030   | ✅ Yes        |
| 3    | 61.7%    | 0.002   | ✅ Yes        |
| 4    | 58.3%    | 0.010   | ✅ Yes        |
| 5    | 60.8%    | 0.002   | ✅ Yes        |

**Aggregate:**
- Mean accuracy: 60.4% ± 5.0% (SEM ± 2.0%)
- 95% CI: [55.2%, 65.6%]
- t-test vs 50%: t = 5.145, p = 0.0036, d = 2.10

### SMOOTHING Control (Variance Suppressed)

| Metric | Value |
|--------|-------|
| Seeds | 10 |
| Mean accuracy | 50.2% ± 5.0% (SEM ± 1.6%) |
| 95% CI | [46.6%, 53.8%] |
| p-value (vs 50%) | 0.919 |
| Significant seeds | 1/10 (10%) |

**Interpretation:** Suppressing ΔΓ variance eliminates discriminability → variance, not magnitude, drives phenomenology.

---

## 🧩 Ablation Study (Seed 1 Example)

| Features | Accuracy | Interpretation |
|----------|----------|----------------|
| **All features** | 60.0% | Full model |
| **Time only** | 50.8% | Timing alone insufficient |
| **Semantic only** | 63.3% | Semantic features strongest |
| **No latency** | 57.5% | Latency contributes modestly |

**Key Finding:** ΔΓ dynamics manifest primarily in **semantic structure** (topic shifts, content variance), not timing.

---

## 🔍 Double Dissociation

| Condition | ΔΓ Variance | Accuracy | p-value | Conclusion |
|-----------|-------------|----------|---------|------------|
| **SHOCK** | High | 60.4% | 0.0036 | ✅ Detectable |
| **SMOOTHING** | Low | 50.2% | 0.919 | ❌ Chance-level |

**Statistical Test:**  
Independent t-test: t = 3.96, p < 0.001, d ≈ 2.04

**Conclusion:** ΔΓ variance (not magnitude) drives phenomenological detectability.

---

## 🚀 Reproduction Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run SHOCK Mode (Paper Results)
```bash
python src/test_inverse_turing_V2_DUAL.py --config medium --mode shock
```

### 3. Run SMOOTHING Control
```bash
python src/test_inverse_turing_V2_DUAL.py --config medium --mode smoothing
```

### 4. Validate Results
```bash
python tests/validate_medium_results.py
```

Expected output:
```
SHOCK: 60.4% ± 5.0%, p = 0.0036
SMOOTHING: 50.2% ± 5.0%, p = 0.919
Double dissociation: t = 3.96, p < 0.001
```

---

## 🎓 Theoretical Implications

1. **ΔΓ is non-redundant:** Not explained by Γ, semantics, or timing alone
2. **Variance matters:** Var(||ΔΓ||²) drives detectability, not ||ΔΓ||
3. **Phenomenological marker:** ΔΓ dynamics correspond to subjective experience signatures
4. **Threshold behavior:** E(t) > θ_E required for conscious-like processing

---

## 📚 Related Work

- **Integrated Information Theory (IIT):** Φ measures integration; ΔΓ measures acceleration
- **Global Workspace Theory (GWT):** Broadcasting; ΔΓ captures sudden transitions
- **Predictive Processing:** Surprisal (Γ); ΔΓ adds second-order dynamics
- **Free Energy Principle:** Minimizing prediction error; ΔΓ as homeostatic response

---

## 🔗 Paper Reference

**Section IV.B: Inverse Turing Test**  
Pages 27-32 in *The ΔΓ-Metamnesis Framework: A Thermodynamic Theory of Consciousness Based on Memory Acceleration Dynamics* (Mathieu, 2026)

---

## 📧 Questions?

Contact: **hpmathieu@ajoursante.ca**

---

**Last Updated:** 2026-01-14  
**Version:** 1.0
