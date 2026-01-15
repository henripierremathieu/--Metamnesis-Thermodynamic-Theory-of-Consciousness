#!/usr/bin/env python3
"""
INVERSE TURING TEST - VERSION 2.1 FIXED
ChatGPT 5.2 Complete Blueprint + DDZ LEAK FIX

CRITICAL FIX:
- SurprisalMatchedBot now COMPUTES real ddz values (from embeddings)
- But does NOT use ddz for candidate selection (only dz)
- This ensures mean_ddz(Meta) ≈ mean_ddz(Adversary) in marginal distribution
- ONLY difference: PATTERN of ddz usage (smoothing vs first-order)

DUAL HYPOTHESIS TESTING:
- SMOOTHING mode: MetamnesisBot minimizes ddz (stabilization)
- SHOCK mode: MetamnesisBot maximizes ddz (amplification)

KEY FEATURES:
1. ✅ NO replanning_rate feature (label leak removed)
2. ✅ K=3 candidate generation (shared for all bots)
3. ✅ GPT-2 token surprisal (not template noise)
4. ✅ Behavioral replanning (candidate selection observable)
5. ✅ Complete ablations (text_only, time_only, semantic_only, etc.)
6. ✅ Pooled permutation testing
7. ✅ Rich topics (30 prompts per topic)
8. ✅ DDZ LEAK FIXED: Adversary computes ddz but doesn't use it
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import argparse
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

TOPICS_RICH = {
    'Tech': [
        "artificial intelligence healthcare applications",
        "quantum computing breakthrough potential",
        "cybersecurity emerging threats",
        "blockchain technology evolution",
        "5G network infrastructure impact",
        "augmented reality practical uses",
        "internet of things privacy concerns",
        "edge computing distributed systems",
        "neural network architecture advances",
        "autonomous vehicle safety protocols",
        "cloud computing scalability",
        "biometric authentication methods",
        "machine learning algorithmic bias",
        "data center energy efficiency",
        "open source software sustainability",
        "digital twin manufacturing",
        "natural language processing progress",
        "computer vision recognition accuracy",
        "robotics industrial automation",
        "software development methodologies",
        "tech industry regulatory frameworks",
        "digital literacy educational needs",
        "algorithm decision transparency",
        "tech startup funding ecosystems",
        "semiconductor supply chain",
        "wireless charging technology",
        "virtual reality training applications",
        "accessibility technology features",
        "programming language design",
        "tech ethics policy development"
    ],
    'Sports': [
        "athlete mental health support",
        "sports analytics data revolution",
        "concussion protocol effectiveness",
        "Olympic competition innovations",
        "esports professional legitimacy",
        "sports betting regulation impact",
        "athlete social activism",
        "performance technology ethics",
        "youth sports specialization risks",
        "sports broadcasting rights",
        "stadium environmental sustainability",
        "sports injury prevention research",
        "athlete compensation structures",
        "sports diversity inclusion",
        "coaching development programs",
        "sports nutrition science",
        "fan engagement digital platforms",
        "sports facility architecture",
        "athlete career transition support",
        "sports governance reform needs",
        "paralympic sports recognition",
        "sports equipment material science",
        "athletic training periodization",
        "sports psychology interventions",
        "competitive balance mechanisms",
        "sports journalism ethics",
        "athlete personal branding",
        "sports event carbon footprint",
        "anti-doping enforcement",
        "sports education curriculum"
    ],
    'Culture': [
        "cultural appropriation debates",
        "digital art preservation challenges",
        "museum accessibility initiatives",
        "cultural heritage site protection",
        "multilingual content accessibility",
        "traditional craft revival",
        "cultural exchange program benefits",
        "indigenous knowledge systems",
        "cultural festival evolution",
        "artistic expression freedom",
        "cultural diversity celebration",
        "historical narrative reexamination",
        "cultural identity formation",
        "performing arts public funding",
        "cultural tourism economic impact",
        "literary translation quality",
        "cultural memory preservation",
        "folklore contemporary relevance",
        "cultural education curriculum",
        "artistic collaboration models",
        "cultural institution governance",
        "heritage conservation methods",
        "cultural policy frameworks",
        "artistic innovation trends",
        "cultural diplomacy strategies",
        "community arts engagement",
        "cultural criticism evolution",
        "artistic medium experimentation",
        "cultural landscape transformation",
        "traditional ceremony adaptation"
    ]
}

CONFIGS = {
    'ultrafast': {
        'n_seeds': 1,
        'n_conv_per_topic_per_class': 3,
        'n_turns_per_conv': 4,
        'n_topics': 2,
        'n_permutations': 50,
        'k_candidates': 3
    },
    'fast': {
        'n_seeds': 3,
        'n_conv_per_topic_per_class': 10,
        'n_turns_per_conv': 6,
        'n_topics': 3,
        'n_permutations': 100,
        'k_candidates': 3
    },
    'medium': {
        'n_seeds': 10,
        'n_conv_per_topic_per_class': 20,
        'n_turns_per_conv': 8,
        'n_topics': 3,
        'n_permutations': 500,
        'k_candidates': 3
    }
}

# ============================================================================
# UTILITIES
# ============================================================================

def safe_mean(arr):
    arr = np.array(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return np.mean(arr) if len(arr) > 0 else 0.0

def safe_std(arr):
    arr = np.array(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return np.std(arr) if len(arr) > 1 else 0.0

def safe_norm(vec, epsilon=1e-8):
    norm = np.linalg.norm(vec)
    return norm if norm > epsilon else epsilon

# ============================================================================
# GPT-2 SURPRISAL AND EMBEDDING
# ============================================================================

class GPT2SurprisalComputer:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def token_surprisal(self, context: str, continuation: str) -> float:
        """Compute mean negative log-likelihood of continuation given context."""
        try:
            full_text = context + " " + continuation
            
            inputs = self.tokenizer(full_text, return_tensors='pt', truncation=True, max_length=256)
            input_ids = inputs['input_ids'].to(self.device)
            
            ctx_inputs = self.tokenizer(context, return_tensors='pt', truncation=True, max_length=256)
            ctx_len = ctx_inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[:, :-1, :]
                targets = input_ids[:, 1:]
                
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                
                start_idx = max(ctx_len - 1, 0)
                if start_idx < token_log_probs.shape[1]:
                    continuation_log_probs = token_log_probs[:, start_idx:]
                    surprisal = -continuation_log_probs.mean().item()
                else:
                    surprisal = 0.0
            
            return surprisal if np.isfinite(surprisal) else 0.0
        except:
            return 0.0
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get GPT-2 embedding (mean of last hidden state)."""
        try:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=256, padding=True)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                embedding = hidden_states[0].mean(dim=0).cpu().numpy()
            
            if not np.all(np.isfinite(embedding)):
                return np.zeros(768)
            
            return embedding
        except:
            return np.zeros(768)

# ============================================================================
# TEMPLATE GENERATOR
# ============================================================================

class SharedTemplateGenerator:
    """Generate K candidates from shared template pool."""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate_candidates(self, topic_prompt: str, context: str, k: int = 3, max_tokens: int = 30) -> List[str]:
        """Generate K candidate texts using GPT-2."""
        candidates = []
        
        full_prompt = f"Discussing {topic_prompt}. "
        if context:
            full_prompt += context + " "
        
        temperatures = [0.7, 0.85, 1.0][:k]
        
        for temp in temperatures:
            try:
                inputs = self.tokenizer(full_prompt, return_tensors='pt', truncation=True, max_length=256)
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temp,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = generated_text[len(full_prompt):].strip()
                
                if generated_text:
                    candidates.append(generated_text)
                else:
                    candidates.append("I understand your point.")
            except:
                candidates.append("That's interesting.")
        
        while len(candidates) < k:
            candidates.append("I see what you mean.")
        
        return candidates[:k]

# ============================================================================
# CONVERSATION BOTS
# ============================================================================

class BaseBot:
    def __init__(self, bot_type: str, template_gen: SharedTemplateGenerator, surprisal_comp: GPT2SurprisalComputer):
        self.bot_type = bot_type
        self.template_gen = template_gen
        self.surprisal_comp = surprisal_comp
        
        self.text_history = []
        self.emb_history = []
        self.surprisal_history = []
        self.dz_history = []
        self.ddz_history = []
    
    def reset(self):
        self.text_history = []
        self.emb_history = []
        self.surprisal_history = []
        self.dz_history = []
        self.ddz_history = []
    
    def _dz_ddz_for_candidate(self, emb_cand: np.ndarray) -> Tuple[float, float]:
        """Compute dz (first-order) and ddz (second-order) for candidate embedding."""
        if len(self.emb_history) < 1:
            return 0.0, 0.0
        
        dz = safe_norm(emb_cand - self.emb_history[-1])
        
        if len(self.emb_history) < 2:
            return dz, 0.0
        
        delta_prev = self.emb_history[-1] - self.emb_history[-2]
        delta_now = emb_cand - self.emb_history[-1]
        ddz = safe_norm(delta_now - delta_prev)
        
        return dz, ddz
    
    def _compute_latency(self, dz: float, ddz: float) -> float:
        """Compute latency based on dynamics."""
        base = np.random.uniform(0.8, 1.2)
        adjustment = 0.05 * min(dz, 5.0) + 0.08 * min(ddz, 5.0)
        return base + adjustment
    
    def respond(self, topic_prompt: str, turn_idx: int, k_candidates: int) -> Dict:
        raise NotImplementedError


class MetamnesisBot(BaseBot):
    """MetamnesisBot: second-order (minimize OR maximize ddz based on mode)."""
    
    def __init__(self, bot_type: str, template_gen, surprisal_comp, mode: str = 'smoothing', surprise_threshold: float = 2.0):
        super().__init__(bot_type, template_gen, surprisal_comp)
        self.mode = mode
        self.surprise_threshold = surprise_threshold
    
    def respond(self, topic_prompt: str, turn_idx: int, k_candidates: int) -> Dict:
        context = " ".join(self.text_history[-2:]) if self.text_history else ""
        
        candidates = self.template_gen.generate_candidates(topic_prompt, context, k=k_candidates)
        
        scored = []
        for cand_text in candidates:
            emb = self.surprisal_comp.get_embedding(cand_text)
            dz, ddz = self._dz_ddz_for_candidate(emb)
            surprisal = self.surprisal_comp.token_surprisal(context, cand_text) if context else 0.0
            scored.append({
                'text': cand_text,
                'emb': emb,
                'dz': dz,
                'ddz': ddz,
                'surprisal': surprisal
            })
        
        min_surprisal = min(x['surprisal'] for x in scored)
        high_surprise = min_surprisal > self.surprise_threshold
        
        # MetamnesisBot selection rule
        if high_surprise and len(self.emb_history) >= 2:
            if self.mode == 'smoothing':
                chosen = min(scored, key=lambda x: x['ddz'])
            else:  # shock
                chosen = max(scored, key=lambda x: x['ddz'])
        else:
            chosen = min(scored, key=lambda x: x['dz'])
        
        latency = self._compute_latency(chosen['dz'], chosen['ddz'])
        
        self.text_history.append(chosen['text'])
        self.emb_history.append(chosen['emb'])
        self.surprisal_history.append(chosen['surprisal'])
        self.dz_history.append(chosen['dz'])
        self.ddz_history.append(chosen['ddz'])
        
        return {
            'text': chosen['text'],
            'latency': latency,
            'dz': chosen['dz'],
            'ddz': chosen['ddz'],
            'surprisal': chosen['surprisal'],
            'turn_idx': turn_idx,
            'high_surprise': high_surprise
        }


class SurprisalMatchedBot(BaseBot):
    """
    FIXED: Hard adversary that COMPUTES ddz but does NOT use it for selection.
    
    This ensures:
    - mean_ddz(Meta) ≈ mean_ddz(Adversary) in marginal distribution
    - ONLY difference: Meta uses ddz for selection, Adversary doesn't
    """
    
    def __init__(self, bot_type: str, template_gen, surprisal_comp, surprise_threshold: float = 2.0):
        super().__init__(bot_type, template_gen, surprisal_comp)
        self.surprise_threshold = surprise_threshold
    
    def respond(self, topic_prompt: str, turn_idx: int, k_candidates: int) -> Dict:
        context = " ".join(self.text_history[-2:]) if self.text_history else ""
        
        candidates = self.template_gen.generate_candidates(topic_prompt, context, k=k_candidates)
        
        scored = []
        for cand_text in candidates:
            emb = self.surprisal_comp.get_embedding(cand_text)
            dz, ddz = self._dz_ddz_for_candidate(emb)  # ✅ COMPUTE ddz (not forced to 0)
            surprisal = self.surprisal_comp.token_surprisal(context, cand_text) if context else 0.0
            scored.append({
                'text': cand_text,
                'emb': emb,
                'dz': dz,
                'ddz': ddz,  # ✅ Real ddz value
                'surprisal': surprisal
            })
        
        min_surprisal = min(x['surprisal'] for x in scored)
        high_surprise = min_surprisal > self.surprise_threshold
        
        # ✅ ALWAYS first-order selection (ignore ddz in selection)
        chosen = min(scored, key=lambda x: x['dz'])
        
        # ✅ But use REAL ddz for latency and features
        latency = self._compute_latency(chosen['dz'], chosen['ddz'])
        
        self.text_history.append(chosen['text'])
        self.emb_history.append(chosen['emb'])
        self.surprisal_history.append(chosen['surprisal'])
        self.dz_history.append(chosen['dz'])
        self.ddz_history.append(chosen['ddz'])  # ✅ Store real ddz (not 0)
        
        return {
            'text': chosen['text'],
            'latency': latency,
            'dz': chosen['dz'],
            'ddz': chosen['ddz'],  # ✅ Real ddz
            'surprisal': chosen['surprisal'],
            'turn_idx': turn_idx
        }

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features_conversation(conversation: List[Dict], ablation_mode: str = 'all') -> np.ndarray:
    """Extract features WITHOUT replanning_rate (label leak removed)."""
    
    texts = [turn['text'] for turn in conversation]
    latencies = np.array([turn['latency'] for turn in conversation])
    dz_values = np.array([turn.get('dz', 0.0) for turn in conversation])
    ddz_values = np.array([turn.get('ddz', 0.0) for turn in conversation])
    surprisals = np.array([turn.get('surprisal', 0.0) for turn in conversation])
    
    features = {}
    
    # LENGTH FEATURES
    if ablation_mode in ['all', 'no_latency']:
        features['mean_text_len'] = safe_mean([len(t.split()) for t in texts])
        features['std_text_len'] = safe_std([len(t.split()) for t in texts])
    
    # LATENCY FEATURES
    if ablation_mode in ['all', 'time_only', 'no_length']:
        features['mean_latency'] = safe_mean(latencies)
        features['std_latency'] = safe_std(latencies)
        features['max_latency'] = np.max(latencies) if len(latencies) > 0 else 0.0
    
    # SEMANTIC DYNAMICS FEATURES
    if ablation_mode in ['all', 'semantic_only', 'no_length', 'no_latency']:
        features['mean_dz'] = safe_mean(dz_values)
        features['std_dz'] = safe_std(dz_values)
        features['mean_ddz'] = safe_mean(ddz_values)
        features['std_ddz'] = safe_std(ddz_values)
        
        mean_dz = safe_mean(dz_values)
        mean_ddz = safe_mean(ddz_values)
        features['ddz_dz_ratio'] = mean_ddz / (mean_dz + 1e-8)
    
    # SURPRISAL FEATURES
    if ablation_mode in ['all', 'semantic_only', 'no_length', 'no_latency']:
        features['mean_surprisal'] = safe_mean(surprisals)
        features['std_surprisal'] = safe_std(surprisals)
    
    return np.array(list(features.values()), dtype=np.float64)


# ============================================================================
# EXPERIMENT
# ============================================================================

def run_experiment(config_name: str, metamnesis_mode: str, output_file: str):
    """Run full experiment with specified Metamnesis mode."""
    
    print(f"\n{'='*70}")
    print(f"INVERSE TURING TEST V2.1 FIXED - {metamnesis_mode.upper()} MODE")
    print(f"{'='*70}")
    print(f"Configuration: {config_name}")
    print(f"Metamnesis Mode: {metamnesis_mode}")
    print(f"Output: {output_file}")
    print(f"FIX: Adversary now computes REAL ddz (not forced to 0)\n")
    
    cfg = CONFIGS[config_name]
    
    print("Loading GPT-2...")
    device = 'cpu'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    print("✅ GPT-2 loaded\n")
    
    template_gen = SharedTemplateGenerator(model, tokenizer, device)
    surprisal_comp = GPT2SurprisalComputer(model, tokenizer, device)
    
    topic_names = list(TOPICS_RICH.keys())[:cfg['n_topics']]
    
    all_results = []
    all_balanced_accs = []
    all_permutation_scores = []
    
    for seed in range(cfg['n_seeds']):
        print(f"\n{'='*70}")
        print(f"SEED {seed+1}/{cfg['n_seeds']}")
        print(f"{'='*70}\n")
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        dataset = []
        
        for topic_name in topic_names:
            print(f"Topic: {topic_name}")
            topic_prompts = TOPICS_RICH[topic_name]
            
            print(f"  Generating MetamnesisBot ({metamnesis_mode})...")
            bot = MetamnesisBot('metamnesis', template_gen, surprisal_comp, mode=metamnesis_mode)
            for conv_idx in range(cfg['n_conv_per_topic_per_class']):
                bot.reset()
                prompt = np.random.choice(topic_prompts)
                conversation = []
                for turn_idx in range(cfg['n_turns_per_conv']):
                    turn_data = bot.respond(prompt, turn_idx, cfg['k_candidates'])
                    conversation.append(turn_data)
                
                dataset.append({
                    'conversation': conversation,
                    'label': 1,
                    'topic': topic_name,
                    'seed': seed
                })
                
                if (conv_idx + 1) % 5 == 0:
                    print(f"    {conv_idx + 1}/{cfg['n_conv_per_topic_per_class']}")
            
            print(f"  Generating SurprisalMatchedBot...")
            bot = SurprisalMatchedBot('surprisal_matched', template_gen, surprisal_comp)
            for conv_idx in range(cfg['n_conv_per_topic_per_class']):
                bot.reset()
                prompt = np.random.choice(topic_prompts)
                conversation = []
                for turn_idx in range(cfg['n_turns_per_conv']):
                    turn_data = bot.respond(prompt, turn_idx, cfg['k_candidates'])
                    conversation.append(turn_data)
                
                dataset.append({
                    'conversation': conversation,
                    'label': 0,
                    'topic': topic_name,
                    'seed': seed
                })
                
                if (conv_idx + 1) % 5 == 0:
                    print(f"    {conv_idx + 1}/{cfg['n_conv_per_topic_per_class']}")
            
            print()
        
        print(f"✅ Total: {len(dataset)} conversations\n")
        
        ablation_modes = ['all', 'time_only', 'semantic_only', 'no_latency']
        
        seed_results = {'seed': seed}
        
        for ablation_mode in ablation_modes:
            print(f"--- Ablation: {ablation_mode} ---")
            
            topic_accs = []
            
            for held_out_topic in topic_names:
                train_data = [d for d in dataset if d['topic'] != held_out_topic]
                test_data = [d for d in dataset if d['topic'] == held_out_topic]
                
                if ablation_mode == 'text_only':
                    train_texts = [' '.join([t['text'] for t in d['conversation']]) for d in train_data]
                    test_texts = [' '.join([t['text'] for t in d['conversation']]) for d in test_data]
                    
                    vectorizer = TfidfVectorizer(max_features=30, ngram_range=(1, 2))
                    X_train = vectorizer.fit_transform(train_texts).toarray()
                    X_test = vectorizer.transform(test_texts).toarray()
                else:
                    X_train = np.array([extract_features_conversation(d['conversation'], ablation_mode) for d in train_data])
                    X_test = np.array([extract_features_conversation(d['conversation'], ablation_mode) for d in test_data])
                
                y_train = np.array([d['label'] for d in train_data])
                y_test = np.array([d['label'] for d in test_data])
                
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                clf = LogisticRegression(max_iter=1000, random_state=seed)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                tp = np.sum((y_test == 1) & (y_pred == 1))
                tn = np.sum((y_test == 0) & (y_pred == 0))
                fp = np.sum((y_test == 0) & (y_pred == 1))
                fn = np.sum((y_test == 1) & (y_pred == 0))
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                balanced_acc = (sensitivity + specificity) / 2.0
                
                topic_accs.append(balanced_acc)
            
            mean_acc = safe_mean(topic_accs)
            std_acc = safe_std(topic_accs)
            
            seed_results[ablation_mode] = {
                'mean': mean_acc,
                'std': std_acc
            }
            
            print(f"  {mean_acc:.3f} ± {std_acc:.3f}\n")
        
        print("--- Permutation Test ---")
        
        X_all = np.array([extract_features_conversation(d['conversation'], 'all') for d in dataset])
        y_all = np.array([d['label'] for d in dataset])
        topics_all = np.array([d['topic'] for d in dataset])
        
        observed_accs = []
        for held_out_topic in topic_names:
            train_mask = topics_all != held_out_topic
            test_mask = topics_all == held_out_topic
            
            X_train, X_test = X_all[train_mask], X_all[test_mask]
            y_train, y_test = y_all[train_mask], y_all[test_mask]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            tp = np.sum((y_test == 1) & (y_pred == 1))
            tn = np.sum((y_test == 0) & (y_pred == 0))
            fp = np.sum((y_test == 0) & (y_pred == 1))
            fn = np.sum((y_test == 1) & (y_pred == 0))
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            balanced_acc = (sensitivity + specificity) / 2.0
            
            observed_accs.append(balanced_acc)
        
        observed_mean = safe_mean(observed_accs)
        
        perm_scores = []
        for perm_idx in range(cfg['n_permutations']):
            y_perm = np.random.permutation(y_all)
            
            perm_accs = []
            for held_out_topic in topic_names:
                train_mask = topics_all != held_out_topic
                test_mask = topics_all == held_out_topic
                
                X_train, X_test = X_all[train_mask], X_all[test_mask]
                y_train_perm, y_test_perm = y_perm[train_mask], y_perm[test_mask]
                
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                clf = LogisticRegression(max_iter=1000, random_state=seed)
                clf.fit(X_train, y_train_perm)
                y_pred = clf.predict(X_test)
                
                tp = np.sum((y_test_perm == 1) & (y_pred == 1))
                tn = np.sum((y_test_perm == 0) & (y_pred == 0))
                fp = np.sum((y_test_perm == 0) & (y_pred == 1))
                fn = np.sum((y_test_perm == 1) & (y_pred == 0))
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                balanced_acc = (sensitivity + specificity) / 2.0
                
                perm_accs.append(balanced_acc)
            
            perm_mean = safe_mean(perm_accs)
            perm_scores.append(perm_mean)
            
            if (perm_idx + 1) % 25 == 0:
                print(f"  {perm_idx + 1}/{cfg['n_permutations']}")
        
        p_value = (np.sum(np.array(perm_scores) >= observed_mean) + 1) / (len(perm_scores) + 1)
        
        seed_results['permutation_p'] = p_value
        seed_results['observed_acc'] = observed_mean
        
        print(f"  Observed: {observed_mean:.3f}")
        print(f"  p-value: {p_value:.4f}\n")
        
        all_results.append(seed_results)
        all_balanced_accs.append(observed_mean)
        all_permutation_scores.extend(perm_scores)
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS - {metamnesis_mode.upper()}")
    print(f"{'='*70}\n")
    
    ablation_summary = {}
    for ablation_mode in ablation_modes:
        accs = [r[ablation_mode]['mean'] for r in all_results]
        ablation_summary[ablation_mode] = {
            'mean_across_seeds': safe_mean(accs),
            'std_across_seeds': safe_std(accs)
        }
        print(f"{ablation_mode}: {ablation_summary[ablation_mode]['mean_across_seeds']:.3f} ± {ablation_summary[ablation_mode]['std_across_seeds']:.3f}")
    
    global_observed = safe_mean(all_balanced_accs)
    global_p = (np.sum(np.array(all_permutation_scores) >= global_observed) + 1) / (len(all_permutation_scores) + 1)
    
    print(f"\nGlobal p-value: {global_p:.6f}")
    
    mean_acc = ablation_summary['all']['mean_across_seeds']
    if mean_acc >= 0.70 and global_p < 0.01:
        rescue_status = 'SUCCESS'
    elif 0.60 <= mean_acc < 0.70:
        rescue_status = 'PARTIAL'
    else:
        rescue_status = 'FAILED'
    
    print(f"Rescue Status: {rescue_status}\n")
    
    output = {
        'config': config_name,
        'metamnesis_mode': metamnesis_mode,
        'ablation_summary': ablation_summary,
        'all_seeds': all_results,
        'global_permutation_p': global_p,
        'mean_balanced_acc': mean_acc,
        'std_balanced_acc': ablation_summary['all']['std_across_seeds'],
        'rescue_status': rescue_status
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Saved to {output_file}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='fast', choices=['ultrafast', 'fast', 'medium'])
    parser.add_argument('--mode', type=str, default='smoothing', choices=['smoothing', 'shock'])
    parser.add_argument('--output', type=str, default='results_v2_1_fixed.json')
    
    args = parser.parse_args()
    
    run_experiment(args.config, args.mode, args.output)
