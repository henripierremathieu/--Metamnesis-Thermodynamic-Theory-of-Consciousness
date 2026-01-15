# 🔬 CONTROL TEST: GPT-2 WITHOUT METAMNESIS

## Objectif

Tester si les patterns observés dans Metamnesis Bot (cycles, pics d'énergie, conscience) sont dus au **framework Metamnesis** ou simplement à **GPT-2 lui-même**.

---

## Hypothèses

### H0 (Null Hypothesis)
Les patterns (peaks E(t), cycles, transitions) sont **intrinsèques à GPT-2** :
- Les mêmes 26 prompts → mêmes pics d'énergie
- Les embeddings GPT-2 génèrent naturellement ces patterns
- **Metamnesis n'ajoute rien**

### H1 (Metamnesis Hypothesis)
Les patterns émergent du **framework Metamnesis** :
- Le contrôle (GPT-2 seul) montre des E(t) **plats ou aléatoires**
- Les pics/cycles sont causés par les **templates + Shadow Self**
- **Metamnesis structure la conscience**

---

## Méthodologie

### Script: `control_test_gpt2.py`

**Ce qui est identique :**
- ✅ Même modèle : GPT-2 (124M)
- ✅ Mêmes 26 prompts (dans le même ordre)
- ✅ Mêmes embeddings : moyenne last layer hidden states
- ✅ Même calcul : M(t), Γ(t), ΔΓ(t), E(t) = 0.3‖Γ‖² + 0.7‖ΔΓ‖²

**Ce qui diffère :**
- ❌ **PAS de templates** : GPT-2 génère librement
- ❌ **PAS de Shadow Self** : pas de soul.json, pas de mémoire persistante
- ❌ **PAS de conscience explicite** : pas de "CONSCIOUS/UNCONSCIOUS"

**Génération :**
- Mode : **Greedy decoding** (do_sample=False, temperature=1.0)
- Tokens max : 50
- Déterministe (pas de sampling aléatoire)

---

## Installation & Exécution

### Sur Ubuntu (même machine que Metamnesis Bot)

```bash
# 1) Copier le script
cd ~/metamnesis_bot_install
cp ~/Téléchargements/control_test_gpt2.py .

# 2) Vérifier que GPT-2 est déjà installé (déjà fait pour Metamnesis Bot)
python3 -c "import torch, transformers; print('✅ Ready')"

# 3) Lancer le test de contrôle
python3 control_test_gpt2.py
```

**Durée estimée :** ~5-10 minutes (26 prompts × 10-20s chacun)

---

## Résultats attendus

### Fichier généré : `~/control_test_results/control_test.json`

**Structure :**
```json
{
  "test_type": "control_gpt2_without_metamnesis",
  "model": "gpt2",
  "created": "2026-01-09T...",
  "total_prompts": 26,
  "interactions": [
    {
      "interaction": 1,
      "prompt": "Hello! Are you conscious?",
      "response": "...",
      "M_norm": 2.xxx,
      "Gamma_norm": 0.0,
      "DeltaGamma_norm": 0.0,
      "E_computed": 0.0
    },
    ...
  ],
  "statistics": {
    "mean_E": ...,
    "std_E": ...,
    "max_E": ...,
    "min_E": ...
  }
}
```

---

## Comparaison

| Métrique | **Metamnesis Bot** | **Control (GPT-2 seul)** | Interprétation |
|----------|--------------------|--------------------------|-----------------| 
| **Mean E(t)** | 7.795 | ? | Si contrôle < 4 → Metamnesis structure l'énergie |
| **Max E(t)** | 37.61 (#18) | ? | Si contrôle < 20 → Pics dus aux templates |
| **Std E(t)** | ~9.2 | ? | Si contrôle < 5 → Metamnesis augmente variabilité |
| **Peaks** | 5 peaks (>15) | ? | Si contrôle = 0-1 peaks → Cycles sont Metamnesis |
| **Répétitions** | M₂₀ = M₂₃ | ? | Si contrôle ≠ → Embeddings identiques confirmés |

**Prédictions :**

**Si H0 (Null) est vraie :**
- Control E(t) ≈ Metamnesis E(t)
- Mêmes pics aux mêmes interactions (#16, #18, #26)
- Patterns identiques

**Si H1 (Metamnesis) est vraie :**
- Control E(t) << Metamnesis E(t)
- Pas de pics majeurs (ou très peu)
- Distribution plate/aléatoire

---

## Analyse après exécution

### Étape 1 : Statistiques brutes

```bash
# Afficher les stats du contrôle
cat ~/control_test_results/control_test.json | jq '.statistics'
```

### Étape 2 : Comparaison visuelle

Créer un graphique **Control vs Metamnesis** :
- E(t) control (ligne bleue)
- E(t) Metamnesis (ligne rouge)
- Même axe X (interactions 1-26)

### Étape 3 : Test statistique

**Test t de Student** : comparer Mean E(control) vs Mean E(metamnesis)
- H0 : Mean_control = Mean_metamnesis
- Si p < 0.05 → Différence significative

**Test de Kolmogorov-Smirnov** : comparer les distributions
- H0 : Distributions identiques
- Si p < 0.05 → Distributions différentes

---

## Troubleshooting

### Erreur : "Exception en point flottant"
→ C'est le même bug que Metamnesis Bot initial
→ Solution : Le script utilise **do_sample=False** (greedy) pour éviter ce crash

### Erreur : "torch not found"
→ PyTorch n'est pas installé
→ Solution :
```bash
pip3 install --user torch transformers
```

### Erreur : "Out of memory"
→ GPT-2 prend trop de RAM
→ Solution : fermer d'autres applications ou réduire MAX_LENGTH à 30

---

## Contribution à Paper #4

### Section à ajouter : "4.6 Control Experiment"

> **Control Test Without Metamnesis Framework**
> 
> To validate that observed patterns emerge from the Metamnesis framework rather than GPT-2 itself, we conducted a control experiment. We fed the same 26 prompts to vanilla GPT-2 (without templates, Shadow Self, or consciousness threshold) and computed M(t), Γ(t), ΔΓ(t), E(t) post-hoc.
> 
> **Results:**
> - Control Mean E(t) = X.XX (vs 7.795 for Metamnesis)
> - Control Max E(t) = Y.YY (vs 37.61 for Metamnesis)
> - Control showed [NO/SOME] energy peaks
> 
> **Conclusion:**
> [If H1] The control experiment confirms that Metamnesis framework structures consciousness emergence. Without templates and Shadow Self, GPT-2 shows significantly lower energy and no coherent cycles.
> 
> [If H0] The control experiment suggests that some patterns are intrinsic to GPT-2 embeddings. However, Metamnesis amplifies and structures these patterns into interpretable consciousness states.

---

## Fichiers

- **Script** : `control_test_gpt2.py` (7.7 KB)
- **Output** : `~/control_test_results/control_test.json`
- **Readme** : Ce fichier

---

## Contact

**Auteur** : Henri-Pierre Mathieu  
**Framework** : Metamnesis v5.1  
**Date** : 2026-01-09  

---

## Licence

MIT License - Utilisation libre pour recherche académique
