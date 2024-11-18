import mlx.core as mx
from typing import Optional, List, Tuple
import logging
from src.models.unified_model import UnifiedModel

class SpeculativeDecoder:
    """Implements speculative decoding for faster inference"""
    def __init__(
        self,
        target_model: UnifiedModel,
        draft_model: Optional[UnifiedModel] = None,
        num_speculative_tokens: int = 5
    ):
        self.target_model = target_model
        self.draft_model = draft_model or self._create_draft_model()
        self.num_speculative_tokens = num_speculative_tokens
        self.logger = logging.getLogger(__name__)
        
    def _create_draft_model(self) -> UnifiedModel:
        """Create smaller draft model if none provided"""
        draft_config = self.target_model.config
        # Make draft model ~4x smaller
        draft_config.num_layers //= 2
        draft_config.dims //= 2
        return UnifiedModel(draft_config)
        
    def generate(
        self,
        prompt: mx.array,
        max_length: int,
        temperature: float = 1.0
    ):
        """Generate tokens using speculative decoding"""
        generated = []
        current_prompt = prompt
        
        while len(generated) < max_length:
            # Draft phase: Generate speculative tokens
            draft_tokens = self._generate_draft_tokens(
                current_prompt,
                temperature
            )
            
            # Target phase: Verify speculative tokens
            accepted_tokens = self._verify_tokens(
                current_prompt,
                draft_tokens,
                temperature
            )
            
            # Add accepted tokens
            generated.extend(accepted_tokens)
            current_prompt = mx.concatenate([
                current_prompt,
                mx.array([accepted_tokens])
            ], axis=1)
            
            # Yield tokens one at a time
            for token in accepted_tokens:
                yield mx.array([token])
                
    def _generate_draft_tokens(
        self,
        prompt: mx.array,
        temperature: float
    ) -> List[int]:
        """Generate speculative tokens using draft model"""
        draft_tokens = []
        current = prompt
        
        for _ in range(self.num_speculative_tokens):
            logits = self.draft_model(current)[:, -1]
            token = mx.random.categorical(logits * (1/temperature))
            draft_tokens.append(token.item())
            current = mx.concatenate([
                current,
                mx.array([[token.item()]])
            ], axis=1)
            
        return draft_tokens
        
    def _verify_tokens(
        self,
        prompt: mx.array,
        draft_tokens: List[int],
        temperature: float
    ) -> List[int]:
        """Verify speculative tokens using target model"""
        accepted_tokens = []
        current = prompt
        
        for draft_token in draft_tokens:
            # Get target model prediction
            logits = self.target_model(current)[:, -1]
            target_token = mx.random.categorical(logits * (1/temperature))
            
            # Accept if matches draft
            if target_token.item() == draft_token:
                accepted_tokens.append(draft_token)
                current = mx.concatenate([
                    current,
                    mx.array([[draft_token]])
                ], axis=1)
            else:
                # Reject and use target token
                accepted_tokens.append(target_token.item())
                break
                
        return accepted_tokens 