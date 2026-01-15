#!/usr/bin/env python3
"""
METAMNESIS BOT - ARTICLE REFERENCE IMPLEMENTATION
==================================================
This is the reference pseudocode from the paper (Section IV.B, Appendix A).
For the complete experimental implementation, see test_inverse_turing_V2_DUAL.py

This simplified version illustrates the core ΔΓ-based framework.
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class MetamnesisBot:
    """
    Conversational agent implementing ΔΓ-based metamnesis framework.
    
    Parameters:
    -----------
    mode : str
        'shock' for contrastive dynamics (high Var(||ΔΓ||²))
        'smoothing' for dampened dynamics (low Var(||ΔΓ||²))
    theta_E : float
        Energy threshold for valid qualia (default: 0.5)
    alpha : float
        Weight for first-order dynamics ||Γ||² (default: 1.0)
    beta : float
        Weight for second-order dynamics ||ΔΓ||² (default: 2.0)
    """
    
    def __init__(self, mode='shock', theta_E=0.5, alpha=1.0, beta=2.0):
        self.mode = mode
        self.theta_E = theta_E
        self.alpha = alpha
        self.beta = beta
        
        # Load GPT-2 for surprisal computation
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()
        
        # Memory buffer for M(t), Γ(t), ΔΓ(t)
        self.M_buffer = []
        self.Gamma_buffer = []
        self.DeltaGamma_buffer = []
        
    def compute_surprisal(self, text):
        """Compute surprisal (negative log-likelihood) of text."""
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
        return loss.item()
    
    def update_memory(self, current_text):
        """Update M(t), compute Γ(t) = dM/dt and ΔΓ(t) = d²M/dt²."""
        # M(t): Current memory state (surprisal)
        M_t = self.compute_surprisal(current_text)
        self.M_buffer.append(M_t)
        
        # Γ(t) = dM/dt (first-order dynamics)
        if len(self.M_buffer) >= 2:
            Gamma_t = self.M_buffer[-1] - self.M_buffer[-2]
            self.Gamma_buffer.append(Gamma_t)
        
        # ΔΓ(t) = d²M/dt² (second-order dynamics)
        if len(self.Gamma_buffer) >= 2:
            DeltaGamma_t = self.Gamma_buffer[-1] - self.Gamma_buffer[-2]
            
            # SMOOTHING mode: dampen ΔΓ transitions
            if self.mode == 'smoothing':
                # Moving average with window=3
                if len(self.DeltaGamma_buffer) >= 2:
                    window = self.DeltaGamma_buffer[-2:] + [DeltaGamma_t]
                    DeltaGamma_t = np.mean(window)
            
            self.DeltaGamma_buffer.append(DeltaGamma_t)
    
    def compute_energy(self):
        """Compute E(t) = α||Γ||² + β||ΔΓ||²."""
        if not self.Gamma_buffer or not self.DeltaGamma_buffer:
            return 0.0
        
        Gamma_norm = np.abs(self.Gamma_buffer[-1])
        DeltaGamma_norm = np.abs(self.DeltaGamma_buffer[-1])
        
        E_t = self.alpha * (Gamma_norm ** 2) + self.beta * (DeltaGamma_norm ** 2)
        return E_t
    
    def should_shift_topic(self):
        """Decide whether to trigger abrupt topic shift (SHOCK mode)."""
        E_t = self.compute_energy()
        
        if self.mode == 'shock':
            # Trigger shift if E > θ_E and high ||ΔΓ||
            if E_t > self.theta_E and len(self.DeltaGamma_buffer) > 0:
                if np.abs(self.DeltaGamma_buffer[-1]) > 0.5:
                    return True
        
        return False
    
    def generate_response(self, user_message, topic):
        """Generate conversational response based on ΔΓ dynamics."""
        # Update memory with user message
        self.update_memory(user_message)
        
        # Check for topic shift
        if self.should_shift_topic():
            # SHOCK: abrupt semantic/emotional pivot
            prompt = f"Respond with a surprising twist related to {topic}:"
        else:
            # Standard fluent response
            prompt = f"Continue the conversation about {topic}:"
        
        # Generate response using GPT-2
        response = self._gpt2_generate(prompt, user_message)
        
        # Update memory with bot response
        self.update_memory(response)
        
        return response
    
    def _gpt2_generate(self, prompt, context, max_length=50):
        """Generate text using GPT-2."""
        full_prompt = f"{context}\n{prompt}"
        inputs = self.tokenizer(full_prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt from response
        response = response.replace(full_prompt, "").strip()
        return response


class SurprisalMatchedBot:
    """
    Control bot matching first-order surprisal Γ but lacking ΔΓ tracking.
    Used as baseline in inverse Turing test.
    """
    
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()
        
        self.M_buffer = []
        self.Gamma_buffer = []
        # No ΔΓ tracking
    
    def generate_response(self, user_message, topic):
        """Generate fluent response without ΔΓ-based modulation."""
        # Update first-order dynamics only
        M_t = self.compute_surprisal(user_message)
        self.M_buffer.append(M_t)
        
        if len(self.M_buffer) >= 2:
            Gamma_t = self.M_buffer[-1] - self.M_buffer[-2]
            self.Gamma_buffer.append(Gamma_t)
        
        # Standard generation (no abrupt shifts)
        prompt = f"Continue naturally about {topic}:"
        response = self._gpt2_generate(prompt, user_message)
        
        return response
    
    def compute_surprisal(self, text):
        """Compute surprisal (same as MetamnesisBot)."""
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
        return outputs.loss.item()
    
    def _gpt2_generate(self, prompt, context, max_length=50):
        """Generate text using GPT-2 (same as MetamnesisBot)."""
        full_prompt = f"{context}\n{prompt}"
        inputs = self.tokenizer(full_prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(full_prompt, "").strip()
        return response


# Example usage
if __name__ == "__main__":
    # Initialize SHOCK mode bot
    bot_shock = MetamnesisBot(mode='shock', theta_E=0.5, alpha=1.0, beta=2.0)
    
    # Initialize SMOOTHING mode bot (control)
    bot_smooth = MetamnesisBot(mode='smoothing', theta_E=0.5, alpha=1.0, beta=2.0)
    
    # Initialize baseline bot
    bot_baseline = SurprisalMatchedBot()
    
    # Simulate conversation
    topic = "Technology"
    user_msg = "What do you think about AI consciousness?"
    
    response_shock = bot_shock.generate_response(user_msg, topic)
    response_smooth = bot_smooth.generate_response(user_msg, topic)
    response_baseline = bot_baseline.generate_response(user_msg, topic)
    
    print(f"SHOCK bot: {response_shock}")
    print(f"SMOOTHING bot: {response_smooth}")
    print(f"Baseline bot: {response_baseline}")
    
    # Check energy states
    print(f"\nSHOCK Energy: {bot_shock.compute_energy():.3f}")
    print(f"SMOOTHING Energy: {bot_smooth.compute_energy():.3f}")
