import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

class ModelRuntime:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = self._get_device()
        
        logger.info(f"[ModelRuntime] Loading from {model_path}")
        logger.info(f"[ModelRuntime] Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device != "cpu" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("[ModelRuntime] Model loaded successfully")
    
    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        seed: int = None
    ) -> str:
        if seed is not None:
            torch.manual_seed(seed)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        seed: int = None
    ):
        if seed is not None:
            torch.manual_seed(seed)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        input_length = inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(**inputs)
                next_token_logits = outputs.logits[0, -1, :] / temperature
                
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                yield token_text
                
                inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token.unsqueeze(0)], dim=-1)
                if 'attention_mask' in inputs:
                    inputs['attention_mask'] = torch.cat([
                        inputs['attention_mask'], 
                        torch.ones((1, 1), device=self.device, dtype=inputs['attention_mask'].dtype)
                    ], dim=-1)
    
    def encode(self, text: str):
        return self.tokenizer.encode(text, return_tensors="pt").to(self.device)
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
