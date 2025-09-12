import os
import time
from typing import Optional, List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelRuntime:
    """Handles model loading and inference operations"""
    
    def __init__(self, model_path: str):
        """Initialize the model runtime
        
        Args:
            model_path: Path to the model directory
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"[ModelRuntime] Loading from {model_path}")
        print(f"[ModelRuntime] Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
            
        self.model.eval()
        
        # Setup RNG
        self.rng = torch.Generator(device=self.device)
        
        print(f"[ModelRuntime] Model loaded successfully")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.tokenizer.get_vocab())
    
    def get_eos_token(self) -> str:
        """Get EOS token string"""
        return self.tokenizer.eos_token or "</s>"
    
    def get_eos_token_id(self) -> int:
        """Get EOS token ID"""
        return self.tokenizer.eos_token_id
    
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text and return tensor"""
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def decode_single(self, token_id: int, skip_special_tokens: bool = True) -> str:
        """Decode a single token ID to text"""
        return self.tokenizer.decode([token_id], skip_special_tokens=skip_special_tokens)
    
    def generate_batch(self, input_ids, attention_mask, max_new_tokens=50, temperature=0.7):
        """Generate for a batch of inputs"""
        try:
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode each output in the batch
            batch_results = []
            for i in range(outputs.shape[0]):
                # Remove input tokens to get only generated text
                generated_tokens = outputs[i][input_ids.shape[1]:]
                decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                batch_results.append(decoded)
                
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Batch generation failed: {e}")
            raise
        
    def generate_single(self, 
                       prompt: str,
                       max_new_tokens: int = 128,
                       temperature: float = 0.7,
                       top_k: int = 0,
                       top_p: float = 0.9,
                       seed: int = 42) -> str:
        """Generate text for a single prompt (no batching)
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (0 = disabled)
            top_p: Top-p sampling
            seed: Random seed
            
        Returns:
            Generated text
        """
        # Set seed
        self.rng.manual_seed(seed)
        
        # Tokenize input
        input_ids = self.tokenize(prompt)
        input_length = input_ids.shape[1]
        
        # Generation parameters
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else 1.0,
            "top_p": top_p,
            "use_cache": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        if top_k > 0:
            generation_kwargs["top_k"] = top_k
            
        # Generate
        with torch.no_grad():
            output = self.model.generate(**generation_kwargs)
            
        # Decode only the new tokens
        new_tokens = output[0][input_length:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return generated_text
    
    def forward_batch(self, 
                     input_ids: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None,
                     past_key_values: Optional[tuple] = None) -> Dict[str, Any]:
        """Forward pass for batched inputs
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]  
            past_key_values: Past key-value cache
            
        Returns:
            Dictionary with logits and past_key_values
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )
            
        return {
            "logits": outputs.logits,
            "past_key_values": outputs.past_key_values
        }
    
    def sample_tokens(self, 
                     logits: torch.Tensor,
                     temperatures: List[float],
                     top_ks: List[int],
                     top_ps: List[float],
                     seeds: List[int]) -> List[int]:
        """Sample next tokens from logits for multiple requests
        
        Args:
            logits: Model logits [batch_size, vocab_size]
            temperatures: Temperature for each request
            top_ks: Top-k values for each request
            top_ps: Top-p values for each request  
            seeds: Random seeds for each request
            
        Returns:
            List of sampled token IDs
        """
        batch_size = logits.shape[0]
        sampled_tokens = []
        
        for i in range(batch_size):
            # Set seed for this request
            self.rng.manual_seed(seeds[i])
            
            # Get logits for this request
            request_logits = logits[i, -1, :].unsqueeze(0)  # [1, vocab_size]
            
            # Apply temperature
            if temperatures[i] > 0:
                request_logits = request_logits / temperatures[i]
            
            # Apply top-k filtering
            if top_ks[i] > 0:
                top_k_logits, top_k_indices = torch.topk(request_logits, top_ks[i])
                request_logits = torch.full_like(request_logits, float('-inf'))
                request_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Apply top-p filtering
            if top_ps[i] < 1.0:
                sorted_logits, sorted_indices = torch.sort(request_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_ps[i]
                # Keep at least one token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                request_logits[indices_to_remove] = float('-inf')
            
            # Sample
            if temperatures[i] > 0:
                probs = torch.softmax(request_logits, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1, generator=self.rng)
            else:
                token_id = torch.argmax(request_logits, dim=-1, keepdim=True)
            
            sampled_tokens.append(token_id.item())
        
        return sampled_tokens
