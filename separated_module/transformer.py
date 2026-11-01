"""Transformer model."""

import numpy as np
from .common import _OPT_AVAILABLE, optimized_ame_encoder, optimized_anisotropy, optimized_project_heads, optimized_qkv_input_grad, optimized_qkv_weight_grad, optimized_softmax_2d
from .geometric_weight_shaping import GeometricWeightShaping
from .target_memory import TargetMemoryMixin


class Transformer(TargetMemoryMixin):
    """From-scratch NumPy Transformer encoder for classification (token ids) and regression (continuous features).

    Multi-head self-attention + feed-forward blocks with layer norm, dropout, label smoothing, cosine
    warm-up learning-rate schedule and manual back-propagation. Uses AME / anisotropy scores of the input
    to measure attention quality, and mixes in TargetMemoryMixin for optional memory-blended predictions.
    """
    def __init__(self, vocab_size, d_model=8, n_heads=2, num_classes=7, learning_rate=0.01, 
                 attn_dropout=0.0, ffn_dropout=0.0, weight_decay=1e-4,
                 n_features=None, task='classification'):
        self.d_model = d_model  # Embedding dimension
        self.n_heads = n_heads
        self.attn_dropout_rate = attn_dropout
        self.ffn_dropout_rate  = ffn_dropout
        self.transformer_lr = learning_rate
        self.weight_decay = weight_decay

        self.task = task   
        self.n_features = n_features
        if self.n_features is not None and self.n_features != d_model:
            self.input_proj = np.random.randn(self.n_features, d_model) * 0.02
            self.input_proj_bias = np.zeros(d_model)
        else:
            self.input_proj = None
            self.input_proj_bias = None
  
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        
        # Positional embeddings (word order)
        self.pos_embedding = np.random.randn(100, d_model) * 0.02  
        
        # Multi-head attention parameters
        self.W_q = np.random.randn(n_heads, d_model, d_model // n_heads) * 0.02
        self.W_k = np.random.randn(n_heads, d_model, d_model // n_heads) * 0.02
        self.W_v = np.random.randn(n_heads, d_model, d_model // n_heads) * 0.02

        self.W_q_fixed = self.W_q.copy()
        self.W_k_fixed = self.W_k.copy()
        self.W_v_fixed = self.W_v.copy()

        self.W_o = np.random.randn(d_model, d_model) * 0.02
        self.encoded = False 
        self._attn_quality_step = 0
        self._forward_count = 0
        self._attn_scale = None

        # Feed-forward network
        self.ffn1 = np.random.randn(d_model, d_model * 4) * 0.02
        self.ffn2 = np.random.randn(d_model * 4, d_model) * 0.02
        
        # Layer norms
        self.ln1_scale = np.ones(d_model)
        self.ln1_shift = np.zeros(d_model)
        self.ln2_scale = np.ones(d_model)
        self.ln2_shift = np.zeros(d_model)
        
        # Output layer
        self.output = np.random.randn(d_model, num_classes) * 0.02
        self.output_bias = np.zeros(num_classes)
        
        self.cache = {}

        self.head_dim = self.d_model // self.n_heads


    def _clear_forward_cache(self, keep_essential=True, max_age_forward_passes=5):
        """Clear cache entries older than max_age_forward_passes."""
        
        essential_keys = {'input_ids', 'mask', 'alpha', 'attn_weights', 'probs', 'logits'}
        oldest_to_keep = self._forward_count - max_age_forward_passes
        
        # Group cache keys by their forward pass number
        forward_keys = {}
        latest_keys = {}
        
        for key in list(self.cache.keys()):
            if '_f' in key:
                parts = key.split('_f')
                if len(parts) == 2:
                    base_key, fwd_num_str = parts
                    try:
                        fwd_num = int(fwd_num_str)
                        if fwd_num not in forward_keys:
                            forward_keys[fwd_num] = []
                        forward_keys[fwd_num].append(key)
                    except ValueError:
                        pass
            elif key.startswith('latest_'):
                latest_keys[key] = True
        
        # Delete old forward passes
        for fwd_num in list(forward_keys.keys()):
            if fwd_num < oldest_to_keep:
                for key in forward_keys[fwd_num]:
                    if key in self.cache:
                        del self.cache[key]
                del forward_keys[fwd_num]
        
        # Also delete latest pointers for essential keys that are too old
        if keep_essential:
            for latest_key in list(latest_keys.keys()):
                actual_key = latest_key.replace('latest_', '')
                # Check if we still have any version of this key from recent passes
                has_recent = False
                for fwd_num in forward_keys:
                    if fwd_num >= oldest_to_keep:
                        for key in forward_keys[fwd_num]:
                            if key.startswith(actual_key + '_f'):
                                has_recent = True
                                break
                    if has_recent:
                        break
                
                if not has_recent and actual_key in essential_keys:
                    # No recent versions of this essential key, delete the latest pointer
                    del self.cache[latest_key]

    def layer_norm(self, x, scale, shift):
        """Layer-normalise x over the feature axis, then apply the learned scale and shift."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return scale * (x - mean) / np.sqrt(var + 1e-5) + shift

    def apply_update(self, param, grad, lr):
        """Return `param` after one SGD step with L2 weight decay applied at update time."""
        # L2 weight decay applied directly at update time
        # equivalent to: grad += weight_decay * param
        update = param - lr * (grad + self.weight_decay * param)
        return update

    def dropout(self, x, rate=0.1, training=True, alpha=None):
        """Inverted dropout with the given rate (identity when not training); `alpha` optionally modulates the rate."""
        if not training or rate == 0.0:
            return x, None
        
        # If alpha provided, scale the effective drop rate by it
        # low alpha (early training, fixed attention) → very light dropout
        # high alpha (dynamic attention active)       → full dropout rate
        effective_rate = rate * alpha if alpha is not None else rate
        
        if effective_rate == 0.0:
            return x, None

        mask = (np.random.rand(*x.shape) > effective_rate).astype(np.float32)
        return x * mask / (1.0 - effective_rate), mask

    def _get_attn_scale(self, d_k):
        """Cache scale factor — that d_k never changes at runtime."""
        if self._attn_scale is None:
            self._attn_scale = 1.0 / np.sqrt(d_k)
        return self._attn_scale



    def attention(self, Q, K, V, mask=None):
        """Scaled dot-product attention softmax(QK^T / sqrt(d)) V with an optional mask; returns (output, attention weights)."""
        d_k = Q.shape[-1]

        # cached scale, no sqrt per call
        scale = self._get_attn_scale(d_k)

        # np.einsum avoids explicit transpose + matmul allocation
        # 'bhqd,bhkd->bhqk' computes Q @ K.T without creating K.T
        scores = np.einsum('bhqd,bhkd->bhqk', Q, K) * scale

        # inplace clip
        np.clip(scores, -50, 50, out=scores)

        # inplace mask application
        if mask is not None:
            # mask: (B,1,1,T) broadcasts to (B,H,T,T)
            scores += (1.0 - mask) * -1e9   # inplace add instead of np.where

        # fused softmax with numerical stability inplace
        scores -= scores.max(axis=-1, keepdims=True)  # stability shift inplace
        np.exp(scores, out=scores)                     # inplace exp
        scores /= scores.sum(axis=-1, keepdims=True) + 1e-8  # inplace normalize
        weights = scores   

        output = np.matmul(weights, V)
        return output, weights


    def softmax(self, x):
        """Numerically stable softmax over the last axis."""
        if _OPT_AVAILABLE:
            return optimized_softmax_2d(np.asarray(x, dtype=np.float64))    

        if x.ndim == 3:
            shifted = x - np.max(x, axis=-1, keepdims=True)
        else:
            shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    

    def multi_head_attention(self, x, mask=None, alpha=None,
                  W_q_mix=None, W_k_mix=None, W_v_mix=None):
        """Project x to Q/K/V, split into heads, run attention per head and merge/project the result back to d_model."""
        batch_size, seq_len, d_model = x.shape
        if W_q_mix is None:
            one_minus = 1.0 - alpha
            W_q_mix = one_minus * self.W_q_fixed + alpha * self.W_q
            W_k_mix = one_minus * self.W_k_fixed + alpha * self.W_k
            W_v_mix = one_minus * self.W_v_fixed + alpha * self.W_v        

        # Optimized Project heads with Cython implementation.
        if _OPT_AVAILABLE:
            B, S, D = batch_size, seq_len, d_model
            M = D // self.n_heads 
            Q = optimized_project_heads(x, W_q_mix, B, S, self.n_heads, D, M)
            K = optimized_project_heads(x, W_k_mix, B, S, self.n_heads, D, M)
            V = optimized_project_heads(x, W_v_mix, B, S, self.n_heads, D, M)
        
        else:
            Q = np.einsum('bsd,hdm->bhsm', x, W_q_mix)
            K = np.einsum('bsd,hdm->bhsm', x, W_k_mix)
            V = np.einsum('bsd,hdm->bhsm', x, W_v_mix)    
        
        # Store for backward
        self.cache['Q'] = Q
        self.cache['K'] = K
        self.cache['V'] = V
        self.cache['x_attn_input'] = x
        
        # Attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        self.cache['attn_weights'] = attn_weights
        self.cache['attn_output'] = attn_output
        
        # Concatenate heads
        attn_output = attn_output.transpose(0, 2,1, 3).reshape(batch_size, seq_len, -1)
        self.cache['attn_concat'] = attn_output
        
        # Final linear projection
        output = np.matmul(attn_output, self.W_o)
        self.cache['attn_out'] = output
        
        return output, attn_weights
        
    def _handle_indices(self, input_ids, dtype=None):
        if not dtype:
            dtype = np.int32

        try:
            try:
                ids = np.asarray(input_ids)
                return ids
            except:
                def flatten(x):
                    for item in x:
                        if isinstance(item, (list, tuple)):
                            yield from flatten(item)
                        else:
                            yield item

                if isinstance(input_ids, (list, tuple)):
                    flat_ids = list(flatten(input_ids))
                else:
                    flat_ids = input_ids  # already a flat array/tensor

                ids = np.asarray(flat_ids, dtype=dtype)
        except:
            flat_ids = self.pipeline._safe_to_2d_float(input_ids)
            ids = np.asarray(flat_ids, dtype=dtype)

        return ids

    def forward_continuous(self, X, training=True, attn_dropout=0.1, ffn_dropout=0.1):
        
        """Forward pass for continuous feature sequences X (batch, seq, features) in regression mode; returns (predictions, attention weights)."""
        eps = 1e-3
        self._forward_count += 1
        self._clear_forward_cache(keep_essential=True, max_age_forward_passes=5)

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X[np.newaxis, ...]

        batch_size, seq_len, n_features = X.shape

        if self.input_proj is not None:
            x = np.matmul(X, self.input_proj) + self.input_proj_bias   # (B, seq_len, d_model)
        else:
            x = X   

        x = x + self.pos_embedding[:seq_len]   

        self.cache['embedded_input'] = x
        self.cache['input_ids'] = None   
        mask = None                       

        self.cache['mask'] = None
        self.cache['seq_len'] = seq_len
        self.cache['batch_size'] = batch_size
        self.cache['x_token'] = x
        self.cache['x_pos'] = x

        AME = self.AME_Encoder(x)
        consistency = 1.0 / (1.0 + np.std(X))
        if np.isnan(AME) or np.isinf(AME):
            AME = 1.0 - consistency
        alpha = 1.0 / (1.0 + np.exp(-AME))
        if not np.isfinite(alpha):
            alpha = 0.5 * (1.0 - ffn_dropout)

        one_minus_alpha = 1.0 - alpha
        W_q_mix = one_minus_alpha * self.W_q_fixed + alpha * self.W_q
        W_k_mix = one_minus_alpha * self.W_k_fixed + alpha * self.W_k
        W_v_mix = one_minus_alpha * self.W_v_fixed + alpha * self.W_v

        attn_out, attn_weights = self.multi_head_attention(x, mask=mask, alpha=alpha,
                                                            W_q_mix=W_q_mix, W_k_mix=W_k_mix, W_v_mix=W_v_mix)

        current_alpha = self.cache.get('alpha', 0.0)
        attn_out, attn_drop_mask = self.dropout(attn_out, rate=self.attn_dropout_rate, training=training, alpha=current_alpha)
        self.cache['attn_drop_mask'] = attn_drop_mask

        self._cached_attn_quality = getattr(self, '_cached_attn_quality', 0.5)
        alpha = 0.95 * alpha + 0.05 * self._cached_attn_quality
        self.alpha = alpha
        self.cache['alpha'] = alpha

        self.cache['x_ln1_input'] = x + attn_out
        x, ln1_mean, ln1_var = self.layer_norm_with_cache(self.cache['x_ln1_input'], self.ln1_scale, self.ln1_shift)
        self.cache['x_after_ln1'] = x
        self.cache['ln1_mean'] = ln1_mean
        self.cache['ln1_var'] = ln1_var

        self.cache['ffn_input'] = x
        ffn_pre = np.matmul(x, self.ffn1)
        self.cache['ffn_pre'] = ffn_pre
        ffn_act = np.maximum(0, ffn_pre)
        ffn_act, ffn_drop_mask = self.dropout(ffn_act, rate=self.ffn_dropout_rate, training=training, alpha=current_alpha)
        self.cache['ffn_act'] = ffn_act
        self.cache['ffn_drop_mask'] = ffn_drop_mask

        ffn_out = np.matmul(ffn_act, self.ffn2)
        self.cache['ffn_out'] = ffn_out

        self.cache['x_ln2_input'] = x + ffn_out
        x, ln2_mean, ln2_var = self.layer_norm_with_cache(self.cache['x_ln2_input'], self.ln2_scale, self.ln2_shift)
        self.cache['x_after_ln2'] = x
        self.cache['ln2_mean'] = ln2_mean
        self.cache['ln2_var'] = ln2_var

        x_pooled = np.mean(x, axis=1)
        self.cache['x_pooled'] = x_pooled

        logits = np.matmul(x_pooled, self.output) + self.output_bias
        self.cache['logits'] = logits

        if self.task == 'regression':
            probs = logits         
        else:
            probs = self.softmax(logits)

        self.cache['probs'] = probs
        return probs, attn_weights


    def forward(self, input_ids, AME=None, _update_quality_matrix=None, embedded=False, pad_token_id=0, training=True, attn_dropout=0.1, ffn_dropout=0.1):
        """Forward pass for token-id input: embeddings + positional encoding, attention and feed-forward blocks, pooling and softmax head. Returns (class probabilities, attention weights)."""
        eps = 1e-3
        self._forward_count += 1

            
        # Clean up old cache entries before storing new ones
        self._clear_forward_cache(keep_essential=True, max_age_forward_passes=5) 

        if embedded:
            x = self._handle_indices(input_ids, dtype=np.int64)
            if x.ndim == 2:
                x = x[np.newaxis, ...]
            batch_size, seq_len, _ = x.shape
            self.cache['embedded_input'] = x
            self.cache['input_ids'] = None
            mask = None
        else:
            input_ids = self._handle_indices(input_ids, dtype=np.int32)
            if input_ids.ndim == 1:
                input_ids = input_ids[np.newaxis, :]

            x = self.token_embedding[input_ids]
            x = x + self.pos_embedding[:x.shape[1]]
            batch_size, seq_len = input_ids.shape
            self.cache['embedded_input'] = None
            self.cache['input_ids'] = input_ids
            mask = self.padding_mask_utility(input_ids, pad_token_id)  # (B,1,1,T)            

        self.cache['mask'] = mask if not embedded else None
        self.cache['seq_len'] = seq_len
        self.cache['batch_size'] = batch_size
        self.cache['x_token'] = x
        self.cache['x_pos'] = x
        
        # Multi-head attention with residual     
        if AME is None:
            AME = self.AME_Encoder(x) 

        consistency = 1.0 / (1.0 + np.std(input_ids))
        if np.isnan(AME) or np.isinf(AME):
            AME = 1.0 - consistency

        alpha = 1.0 / (1.0 + np.exp(-AME))
        if not np.isfinite(alpha):
            alpha = 0.5 * (1.0 - ffn_dropout)

        if len(input_ids) == 0:
            print('[⚠️] score computation: input_ids is empty — using safe default consistency')
            consistency = 0.5 * (1.0 - attn_dropout)  # neutral, hardcoded, cannot be NaN
        else:
            std_val = np.std(input_ids)
            consistency = 1.0 / (1.0 + std_val) if np.isfinite(std_val) else 0.5 * (1.0 - attn_dropout)

        if _update_quality_matrix is None:
            # input ids ranged 0 to 1.
            alpha_rate = 1.0 / (1.0 + alpha)
            _update_quality_matrix = (alpha_rate * (1.0 - ffn_dropout) * consistency) + eps

        if np.isnan(_update_quality_matrix) or np.isinf(_update_quality_matrix):
            if np.isnan(consistency) or np.isinf(consistency):
                consistency = 0.5 * (1.0 - ffn_dropout)
            _update_quality_matrix = consistency

        score = int(1.0 / _update_quality_matrix + 1e-6)

        _should_update = (
            not training or
            not hasattr(self, '_attn_quality_step') or
            self._attn_quality_step % max(1, score) == 0
        )
        
        one_minus_alpha = 1.0 - alpha
        W_q_mix = one_minus_alpha * self.W_q_fixed + alpha * self.W_q
        W_k_mix = one_minus_alpha * self.W_k_fixed + alpha * self.W_k
        W_v_mix = one_minus_alpha * self.W_v_fixed + alpha * self.W_v

        attn_out, attn_weights = self.multi_head_attention(x, mask=mask, alpha=alpha,
                      W_q_mix=W_q_mix, W_k_mix=W_k_mix, W_v_mix=W_v_mix )
  
        current_alpha = self.cache.get('alpha', 0.0) 

        attn_out, attn_drop_mask = self.dropout(attn_out, rate=self.attn_dropout_rate, training=training, alpha=current_alpha)   
        self.cache['attn_drop_mask'] = attn_drop_mask  

        if training and hasattr(self, '_attn_quality_step'):
            self._attn_quality_step += 1
            if _should_update and self._attn_quality_step % 10 == 0:
                attn_quality = self.attention_quality_computing(attn_weights, AME=AME, mask=mask)
                self._cached_attn_quality = attn_quality
            else:
                attn_quality = getattr(self, '_cached_attn_quality', 0.5)
        else:
            self._attn_quality_step  = 0
            self._cached_attn_quality = alpha * (1.0 - current_alpha)
            attn_quality = self._cached_attn_quality 

        alpha = 0.95 * alpha + 0.05 * attn_quality 

        self.alpha = alpha
        self.cache['alpha'] = alpha  # store in cache     

        self.cache['x_ln1_input'] = x + attn_out
        x, ln1_mean, ln1_var = self.layer_norm_with_cache(
                self.cache['x_ln1_input'], self.ln1_scale, self.ln1_shift
            )   

        self.cache['x_after_ln1'] = x
        self.cache['ln1_mean']    = ln1_mean   # reused in backward
        self.cache['ln1_var']     = ln1_var    

        # Feed-forward with residual
        self.cache['ffn_input'] = x
        ffn_pre = np.matmul(x, self.ffn1)
        self.cache['ffn_pre'] = ffn_pre
        
        ffn_act = np.maximum(0, ffn_pre)  # ReLU
           

        ffn_act, ffn_drop_mask = self.dropout(ffn_act, rate=self.ffn_dropout_rate, training=training, alpha=current_alpha)
     
        self.cache['ffn_act'] = ffn_act
        self.cache['ffn_drop_mask'] = ffn_drop_mask   

        ffn_out = np.matmul(ffn_act, self.ffn2)
        self.cache['ffn_out'] = ffn_out
        
        self.cache['x_ln2_input'] = x + ffn_out
        x, ln2_mean, ln2_var = self.layer_norm_with_cache(
                self.cache['x_ln2_input'], self.ln2_scale, self.ln2_shift
            )        
        self.cache['x_after_ln2'] = x
        self.cache['ln2_mean']    = ln2_mean
        self.cache['ln2_var']     = ln2_var        
        
        if mask is not None:
            # Reshape mask to (B, T, 1) for broadcasting against (B, T, D)
            token_mask = mask[:, 0, 0, :, np.newaxis]        # (B, T, 1)
            x_masked   = x * token_mask                       # zero out padding
            lengths    = token_mask.sum(axis=1)               # (B, 1) valid token counts
            x_pooled   = x_masked.sum(axis=1) / (lengths + 1e-6)  # (B, D)
        else:
            x_pooled = np.mean(x, axis=1)

        self.cache['x_pooled'] = x_pooled
        
        # Output projection
        logits = np.matmul(x_pooled, self.output) + self.output_bias
        self.cache['logits'] = logits
        
        probs = self.softmax(logits)
        self.cache['probs'] = probs
        
        return probs, attn_weights


    def layer_norm_with_cache(self, x, scale, shift, eps=1e-5):
        """Layer norm that returns mean and var for backward reuse."""
        mean  = x.mean(axis=-1, keepdims=True)
        var   = x.var(axis=-1,  keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + eps)
        return x_hat * scale + shift, mean, var 


    def layer_norm_backward(self, d_out, x, scale, shift,
               mean=None, var=None):
        """Gradients of layer norm w.r.t. its input, scale and shift."""
        eps = 1e-5
        if mean is None:
            mean = x.mean(axis=-1, keepdims=True)
        if var is None:
            var  = x.var(axis=-1,  keepdims=True)

        std = np.sqrt(var + eps)
        x_hat = (x - mean) / std
        
        N = x.shape[-1]
        dx_hat = d_out * scale
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * std**-3, axis=-1, keepdims=True)
        dmean = np.sum(dx_hat * (-1.0 / std), axis=-1, keepdims=True)
        
        dx = (
        dx_hat / std
        + dvar * 2*(x-mean)/N
        + dmean / N
        )
        
        return dx
    

    # fixed attention backward allow the transformer to not update its Q, K, V projections, allowing much stable attention, while sacrificing flexibility.
    def fixed_attention_backward(self, d_logits, lr=0.01, max_norm=1.0):

        """Back-propagate d_logits through the model with fixed (non-dynamic) attention weighting and update all parameters."""
        # Gradient for output layer
        d_output = d_logits
        alpha = self.cache.get('alpha', 1.0)

        d_Wo = np.dot(self.cache['x_pooled'].T, d_output)
        d_bo = np.sum(d_output, axis=0, keepdims=True)
        
        # Gradient for pooled features
        d_pooled = np.dot(d_output, self.output.T)
        
        # Expand pooled gradient to all positions
        d_x = np.repeat(d_pooled[:, np.newaxis, :] / self.cache['seq_len'], self.cache['seq_len'], axis=1)
        
        # Layer norm 2 gradient
        d_x = self.layer_norm_backward(d_x, self.cache['x_ln1_input'],
                                        self.ln1_scale, self.ln1_shift,
                                        mean=self.cache.get('ln1_mean'),
                                        var=self.cache.get('ln1_var'))
                                        
        # FFN gradients
        d_ffn = d_x
        
        # Gradient for FFN2
        d_ffn2 = np.sum(np.matmul(self.cache['ffn_act'].transpose(0, 2, 1), d_ffn), axis=0)
        
        # Gradient for FFN1 through ReLU
        d_ffn_act = np.matmul(d_ffn, self.ffn2.T)
        ffn_drop_mask = self.cache.get('ffn_drop_mask')
        if ffn_drop_mask is not None:
            d_ffn_act = d_ffn_act * ffn_drop_mask / (1.0 - self.ffn_dropout_rate)

        d_ffn_pre = d_ffn_act * (self.cache['ffn_pre'] >= 0)   # ReLU backward unchanged

        d_prev = np.matmul(d_ffn_pre, self.ffn1.T)
        d_ffn1 = np.sum(np.matmul(self.cache['ffn_input'].transpose(0, 2, 1), d_ffn_pre), axis=0)
        
        # Layer norm 1 gradient
        d_x = self.layer_norm_backward(d_x - self.cache['attn_out'], 
                                        self.cache['x_ln1_input'],
                                        self.ln1_scale, self.ln1_shift)
        
        d_ffn = d_x
        d_residual_ffn = d_ffn
        dx = d_prev + d_residual_ffn
        d_attn = dx

        # Gradient for attention output projection
        attn_drop_mask = self.cache.get('attn_drop_mask')
        if attn_drop_mask is not None:
            d_attn = d_attn * attn_drop_mask / (1.0 - self.attn_dropout_rate)

        d_Wo_attn = np.sum(np.matmul(self.cache['attn_concat'].transpose(0,2,1), d_attn), axis=0)        

        grads = {
                'output':  d_Wo,
                'ffn2':    d_ffn2,
                'ffn1':    d_ffn1,
                'W_o':     d_Wo_attn,
            }

        grads, norm = self.clip_gradients(grads, max_norm)        

        # Update weights
        self.output = self.apply_update(self.output, grads['output'], lr)
        self.ffn2   = self.apply_update(self.ffn2,   grads['ffn2'],   lr)
        self.ffn1   = self.apply_update(self.ffn1,   grads['ffn1'],   lr)
        self.W_o    = self.apply_update(self.W_o,    grads['W_o'],    lr)
        # output_bias intentionally excluded — biases don't get weight decay
            
        return d_x
    

    def dynamic_backward(self, d_logits, lr=0.01, max_norm=1.0):
        """Back-propagate d_logits through the model, including the dynamic alpha attention scaling, with gradient clipping (`max_norm`) and parameter updates."""
        # Gradient for output layer
        d_output = d_logits
        alpha = self.cache.get('alpha', 1.0)

        d_Wo = np.dot(self.cache['x_pooled'].T, d_output)
        d_bo = np.sum(d_output, axis=0)
        
        # Gradient for pooled features
        d_pooled = np.dot(d_output, self.output.T)
        
        # Expand pooled gradient to all positions
        mask = self.cache['mask']  # (B, 1, 1, T)
        if mask is not None:
            token_mask = mask[:, 0, 0, :, np.newaxis]             # (B, T, 1)
            lengths    = token_mask.sum(axis=1, keepdims=True)    # (B, 1, 1)
            d_x        = (d_pooled[:, np.newaxis, :] / (lengths + 1e-6)) * token_mask
        else:
            d_x = np.repeat(d_pooled[:, np.newaxis, :] / self.cache['seq_len'], self.cache['seq_len'], axis=1)    

        # Layer norm 2 gradient
        d_x = self.layer_norm_backward(d_x, self.cache['x_ln2_input'],
                                self.ln2_scale, self.ln2_shift,
                                mean=self.cache.get('ln2_mean'),
                                var=self.cache.get('ln2_var'))

        # FFN gradients
        d_ffn = d_x
        
        # Gradient for FFN2
        d_ffn2 = np.sum(np.matmul(self.cache['ffn_act'].transpose(0, 2, 1), d_ffn), axis=0)
        
        # Gradient for FFN1 through ReLU
        d_ffn_act = np.matmul(d_ffn, self.ffn2.T)
        ffn_drop_mask = self.cache.get('ffn_drop_mask')
        if ffn_drop_mask is not None:
            d_ffn_act = d_ffn_act * ffn_drop_mask / (1.0 - self.ffn_dropout_rate)

        d_ffn_pre = d_ffn_act * (self.cache['ffn_pre'] >= 0)   # ReLU backward unchanged

        d_prev = np.matmul(d_ffn_pre, self.ffn1.T)
        d_ffn1 = np.sum(np.matmul(self.cache['ffn_input'].transpose(0, 2, 1), d_ffn_pre), axis=0)
        
        # Layer norm 1 gradient
        d_ln1 = self.layer_norm_backward(
            d_prev, self.cache['x_ln1_input'], self.ln1_scale, self.ln1_shift
        ) 

        d_residual = d_ln1
        d_attn = d_ln1
        dx = d_prev + d_residual

        # Gradient for attention output projection
        attn_drop_mask = self.cache.get('attn_drop_mask')
        if attn_drop_mask is not None:
            d_attn = d_attn * attn_drop_mask / (1.0 - self.attn_dropout_rate)

        d_Wo_attn = np.sum(np.matmul(self.cache['attn_concat'].transpose(0, 2, 1), d_attn), axis=0)

        d_attn_concat = np.matmul(d_attn, self.W_o.T)
        batch, seq_len, _ = d_attn_concat.shape
        d_head = self.n_heads
        d_dim = self.d_model // self.n_heads

        d_attn_heads = d_attn_concat.reshape(batch, seq_len, d_head, d_dim) .transpose(0, 2, 1, 3)      

        V = self.cache['V']
        K = self.cache['K']
        Q = self.cache['Q']
        weight = self.cache['attn_weights']
        
        d_V = np.matmul(weight.transpose(0, 1, 3, 2), d_attn_heads)
        d_weights = np.matmul(d_attn_heads, V.transpose(0, 1, 3, 2))

        d_scores = weight * (d_weights - np.sum(d_weights * weight, axis=-1, keepdims=True))
        d_scores /= np.sqrt(Q.shape[-1])

        d_Q = np.matmul(d_scores, K)
        d_K = np.matmul(d_scores.transpose(0, 1, 3, 2), Q)

        x = self.cache['x_attn_input']

        if _OPT_AVAILABLE:
            B, S = x.shape[0], x.shape[1]
            H, M = self.n_heads, self.d_model // self.n_heads

            d_W_q = optimized_qkv_weight_grad(x, d_Q, B, S, H, self.d_model, M)
            d_W_k = optimized_qkv_weight_grad(x, d_K, B, S, H, self.d_model, M)
            d_W_v = optimized_qkv_weight_grad(x, d_V, B, S, H, self.d_model, M)

            d_x_q = optimized_qkv_input_grad(d_Q, self.W_q, B, S, H, self.d_model, M)
            d_x_k = optimized_qkv_input_grad(d_K, self.W_k, B, S, H, self.d_model, M)
            d_x_v = optimized_qkv_input_grad(d_V, self.W_v, B, S, H, self.d_model, M)  
        else:          
            d_W_q = np.einsum('bsd, bhsm->hdm', x, d_Q)
            d_W_k = np.einsum('bsd, bhsm->hdm', x, d_K)
            d_W_v = np.einsum('bsd, bhsm->hdm', x, d_V)

            d_x_q = np.einsum('bhsm, hdm->bsd', d_Q, self.W_q)
            d_x_k = np.einsum('bhsm, hdm->bsd', d_K, self.W_k)
            d_x_v = np.einsum('bhsm, hdm->bsd', d_V, self.W_v)

        d_x_attn_input = d_x_q + d_x_k + d_x_v
        d_x_total = d_x_attn_input + d_residual

        input_ids = self.cache.get('input_ids')

        # Update weights
        grads = {
                'output': d_Wo,
                'ffn2':   d_ffn2,
                'ffn1':   d_ffn1,
                'W_o':    d_Wo_attn,
                'W_q':    alpha * d_W_q,   # already alpha-scaled, clip the combined thing
                'W_k':    alpha * d_W_k,
                'W_v':    alpha * d_W_v,
            }
        grads, norm = self.clip_gradients(grads, max_norm)  

        self.output = self.apply_update(self.output, grads['output'], lr)
        self.ffn2   = self.apply_update(self.ffn2,   grads['ffn2'],   lr)
        self.ffn1   = self.apply_update(self.ffn1,   grads['ffn1'],   lr)
        self.W_o    = self.apply_update(self.W_o,    grads['W_o'],    lr)
        self.W_q    = self.apply_update(self.W_q,    grads['W_q'],    lr)
        self.W_k    = self.apply_update(self.W_k,    grads['W_k'],    lr)
        self.W_v    = self.apply_update(self.W_v,    grads['W_v'],    lr)

        if input_ids is not None:
            emb_norm = np.linalg.norm(d_x_total)
            emb_coef = min(1.0, max_norm / (emb_norm + 1e-6))

            flat_ids   = input_ids.flatten()                          # (B*T,)
            flat_grads = d_x_total.reshape(-1, self.d_model) / self.cache['seq_len']  # (B*T, D)

            np.add.at(self.token_embedding, flat_ids, -lr * emb_coef * flat_grads)
            self.pos_embedding[:seq_len] -= lr * emb_coef * d_x_total.mean(axis=0)
        else:
            self.pos_embedding[:seq_len] -= lr * d_x_total.mean(axis=0)
            norm = d_x_total

        return norm

    def smoothing_labels_utility(self, y_true, smoothing=0.1):
        """Apply label smoothing to one-hot targets: y * (1 - s) + s / num_classes."""
        # y_true: (B, num_classes) one-hot
        try:
            num_classes = y_true.shape[1]
        except:
            y_true_2d = y_true.reshape(-1, 1)
            num_classes = y_true_2d.shape[1] 
            
        return y_true * (1.0 - smoothing) + smoothing / num_classes

    def learning_rate_warm_up(self, epoch, epochs, lr_base, schedule='cosine_warmup', warmup_frac=0.1):
        """Learning rate for `epoch`: linear warm-up for the first `warmup_frac` of training, then cosine decay (schedule='cosine_warmup')."""
        warmup_epochs = int(epochs * warmup_frac)
        
        if schedule == 'cosine_warmup':
            if epoch < warmup_epochs:
                # Linear warmup
                return lr_base * (epoch + 1) / warmup_epochs
            else:
                # Cosine decay after warmup
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return lr_base * 0.5 * (1 + np.cos(np.pi * progress))

        elif schedule == 'step':
            # Halve lr every 30% of training
            step = int(epochs * 0.3)
            return lr_base * (0.5 ** (epoch // step))

        elif schedule == 'constant':
            return lr_base

        return lr_base

    def padding_mask_utility(self, input_ids, pad_token_id=0):
        """Build an attention mask that hides padding tokens (id == pad_token_id)."""
        # input_ids: (B, T)
        # Returns: (B, 1, 1, T) — broadcast-ready for (B, heads, T_q, T_k)
        mask = (input_ids != pad_token_id).astype(np.float32)
        return mask[:, np.newaxis, np.newaxis, :]   # (B, 1, 1, T)
        
    def clip_gradients(self, grads: dict, max_norm: float = 1.0):
        """Clip a dict of gradients to a global L2 norm of `max_norm`."""
        #clip gradients function to prevent overflow.
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        clip_coef  = max_norm / (total_norm + 1e-6)

        if clip_coef < 1.0:
            for g in grads.values():
                g *= clip_coef    

        return grads, total_norm    


    def batch_padding_utility(self, sequences, pad_token_id=0):
        """Pad a list of token-id sequences to the same length with `pad_token_id`."""
        # sequences: list of 1-D np arrays of varying length
        max_len = max(len(s) for s in sequences)
        padded  = np.full((len(sequences), max_len), pad_token_id, dtype=np.int32)
        for i, s in enumerate(sequences):
            padded[i, :len(s)] = s
        return padded   # (B, T)

    def train_step_regression(self, X, y_true, lr=0.01, max_norm=1.0, mode=None):
        """One regression training step on (X, y_true): forward, MSE loss, backward and update. Returns the loss."""
        y_true = np.asarray(y_true, dtype=np.float64)
        if y_true.ndim == 1:
            y_true = y_true[:, np.newaxis]   # (B, 1)

        preds, attn_weights = self.forward_continuous(X, training=True,
                                                    attn_dropout=self.attn_dropout_rate,                                           ffn_dropout=self.ffn_dropout_rate)

        loss = np.mean((preds - y_true) ** 2)
        d_logits = 2 * (preds - y_true) / y_true.shape[0]   

        if mode == 'fixed_backward':
            self.fixed_attention_backward(d_logits, lr, max_norm=max_norm)
        else:
            self.dynamic_backward(d_logits, lr, max_norm=max_norm)

        return loss

    def train_regression(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001, batch_size=64):
        """Mini-batch regression training loop with validation, run for `epochs` epochs."""
        n = X_train.shape[0]
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_shuf, y_shuf = X_train[idx], y_train[idx]

            epoch_losses = []
            for start in range(0, n, batch_size):
                end = start + batch_size
                loss = model.train_step_regression(X_shuf[start:end], y_shuf[start:end], lr=lr, mode='dynamic')
                epoch_losses.append(loss)

            val_preds, _ = model.forward_continuous(X_val, training=True, attn_dropout=0.0, ffn_dropout=0.0)
            val_loss = np.mean((val_preds - y_val) ** 2)

            if epoch % 10 == 0:
                print(f"[=] Epoch {epoch} | Train Loss: {np.mean(epoch_losses):.4f} | Val Loss: {val_loss:.4f}") 


    def train(self, input_ids_list, y_true_list, epochs=100, mode=None,
            lr=0.01, embedded=False, max_norm=1.0,
            schedule='cosine_warmup', pad_token_id=0, batch_size=None):

        """Full classification training loop: pads/batches inputs, applies label smoothing and the LR schedule, calls `train_step` per sample/batch. Returns (losses, accuracies) per epoch, or (None, None) if training could not run."""
        # Main Training function for Transformer
        losses  = []
        accs    = []
        y_true_smoothed_list = []        
        d_model = self.d_model

        input_ids_list = self._sanitize_string_chars(input_ids_list)
        y_true_list = self._sanitize_string_chars(y_true_list)

        # W_o geometric init — unchanged
        if not self.encoded:
            self.shaping = GeometricWeightShaping(d_model, d_model)
            shaping_input = input_ids_list
            if embedded:
                shaping_input = np.vstack([
                    x.reshape(-1, x.shape[-1]) if x.ndim >= 2 else x
                    for x in input_ids_list
                ])
            self.W_o     = self.shaping.weight_shaping(shaping_input)
            self.encoded = True

        # batch padding — unchanged
        if batch_size is not None and not embedded:
            input_ids_list = [
                self.batch_padding_utility(input_ids_list[i:i+batch_size], pad_token_id)
                for i in range(0, len(input_ids_list), batch_size)
            ]
            y_true_list = [
                np.stack(y_true_list[i:i+batch_size])
                for i in range(0, len(y_true_list), batch_size)
            ]

        AME = self.AME_Encoder(input_ids_list)
        AMR = 1.0 / (1.0 + np.exp(-AME))

        # precompute LR schedule once
        lr_schedule = [
            self.learning_rate_warm_up(e, epochs, lr, schedule)
            for e in range(epochs)
        ]


        training_not_allowed = AMR < 0.1 or np.isnan(AMR) or np.isinf(AMR)
        if not training_not_allowed:
            print(f"[==] Starting comprehensive training for {epochs} epochs with mode: {mode}, learning rate: {lr}, schedule: {schedule}")

            for y in y_true_list:
                y_arr = np.asarray(y, dtype=np.float64)

                # normalize to exactly 2D before smoothing
                if y_arr.ndim == 0:
                    y_arr = y_arr.reshape(1, 1)
                elif y_arr.ndim == 1:
                    y_arr = y_arr.reshape(1, -1)
                elif y_arr.ndim > 2:
                    y_arr = y_arr.reshape(-1, y_arr.shape[-1])

                smoothed = self.smoothing_labels_utility(y_arr, smoothing=0.1)

                # guarantee 2D output regardless of what smoothing returns
                smoothed = np.asarray(smoothed, dtype=np.float64)
                if smoothed.ndim == 0:
                    smoothed = smoothed.reshape(1, 1)
                elif smoothed.ndim == 1:
                    smoothed = smoothed.reshape(1, -1)
                elif smoothed.ndim > 2:
                    smoothed = smoothed.reshape(-1, smoothed.shape[-1])

                y_true_smoothed_list.append(smoothed)

            for epoch in range(epochs):
                epoch_losses = []
                epoch_accs   = []
            
                current_lr = lr_schedule[epoch]
                self.alpha = min(1.0, epoch / 100)

                for input_ids, y_true, y_true_smooth in zip(
                    input_ids_list, y_true_list, y_true_smoothed_list
                ):
                    if input_ids.ndim == 1:
                        input_ids    = input_ids[np.newaxis, :]
                    if y_true.ndim == 1:
                        y_true       = y_true[np.newaxis, :]
                    
                    loss, acc = self.train_step(
                        input_ids, epoch, y_true,
                        current_lr, AME=AME, mode=mode,
                        embedded=embedded, max_norm=max_norm,
                        pad_token_id=pad_token_id,
                        y_true_smooth=y_true_smooth   # pass precomputed
                    )
                    epoch_losses.append(loss)
                    epoch_accs.append(acc)

                avg_loss = float(np.mean(epoch_losses))
                avg_acc  = float(np.mean(epoch_accs))
                losses.append(avg_loss)
                accs.append(avg_acc)

                if epoch % 10 == 0:
                    print(f"[=] Epoch {epoch} | loss: {avg_loss:.4f} | Acc: {avg_acc:.2%}")

                
            return losses, accs
        else:
            print(f'[!] Transformer Training not allowed due to low AMR or invalid value, AMR: {AMR}. Skipping training.')
            return None, None



    def train_step(self, input_ids, epoch, y_true, lr=0.01, AME=None,
                mode=None, embedded=False, max_norm=1.0,
                pad_token_id=0, y_true_smooth=None):

        """One classification training step: forward pass, cross-entropy loss, attention-quality weighting and backward/update. Returns (loss, accuracy)."""
        y_true = np.asarray(y_true)
        if len(y_true.shape) < 2:
            y_true = y_true.reshape(-1, 1)

        if not embedded and input_ids.ndim == 1:
            input_ids = input_ids[np.newaxis, :]
        if y_true.ndim == 1:
            y_true = y_true[np.newaxis, :]

        probs, attn_weights = self.forward(
            input_ids, AME=AME, embedded=embedded,
            pad_token_id=pad_token_id, training=True,
            attn_dropout=self.attn_dropout_rate,
            ffn_dropout=self.ffn_dropout_rate
        )

        # comprehensive shape normalization before any use
        if y_true_smooth is not None:
            y_true_smooth = np.asarray(y_true_smooth, dtype=np.float64)

            # handle 0-d scalar — expand to (1, 1)
            if y_true_smooth.ndim == 0:
                print(f'[!] y_true_smooth was scalar ({float(y_true_smooth):.4f}) — recomputing')
                y_true_smooth = self.smoothing_labels_utility(y_true, smoothing=0.1)

            # squeeze extra leading dims safely
            while y_true_smooth.ndim > 2:
                y_true_smooth = y_true_smooth.squeeze(0)

            # ensure at least 2D
            if y_true_smooth.ndim == 1:
                y_true_smooth = y_true_smooth[np.newaxis, :]

            # final sanity — if still not 2D something is deeply wrong
            if y_true_smooth.ndim != 2:
                print(f'[!] y_true_smooth shape {y_true_smooth.shape} unrecoverable — recomputing')
                y_true_smooth = self.smoothing_labels_utility(y_true, smoothing=0.1)

        else:
            # compute fresh
            y_true_smooth = self.smoothing_labels_utility(y_true, smoothing=0.1)

        # shape alignment, y_true_smooth guaranteed 2D
        if y_true_smooth.shape[1] != probs.shape[1]:
            if y_true_smooth.shape[1] > probs.shape[1]:
                y_true_smooth = y_true_smooth[:, :probs.shape[1]]
                y_true        = y_true[:, :probs.shape[1]]
            else:
                pad           = probs.shape[1] - y_true_smooth.shape[1]
                y_true_smooth = np.pad(y_true_smooth, ((0, 0), (0, pad)))
                y_true        = np.pad(y_true,        ((0, 0), (0, pad)))

        loss     = -np.mean(np.sum(y_true_smooth * np.log(probs + 1e-8), axis=1))
        d_logits = (probs - y_true_smooth) / y_true_smooth.shape[0]

        if mode == 'fixed_backward':
            self.fixed_attention_backward(d_logits, lr, max_norm=max_norm)
        else:
            self.dynamic_backward(d_logits, lr, max_norm=max_norm)

        preds = np.argmax(probs, axis=1)
        true  = np.argmax(y_true, axis=1)
        acc   = float(np.mean(preds == true))

        return loss, acc

        
    def _sanitize_string_chars(self, x):
        if isinstance(x, (str, np.str_)):
            clean_str = str(x).replace('[', '').replace(']', '').replace('...', '').strip()
            x = np.fromstring(clean_str, sep=' ')

        if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(x.astype(str).flatten()).replace('[', '').replace(']', '')
            skip_values = {"...", "NaN", "null"}
            x = np.fromiter((v for v in clean_str.split() if v not in skip_values), dtype=float)

        return x

    def predict(self, input_ids, embedded=False):
        """Return (predicted class indices, probabilities, attention weights) for input_ids (inference mode)."""
        if not embedded and input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)

        AME = self.AME_Encoder(input_ids)
        probs, attn_weights = self.forward(input_ids, AME=AME, embedded=embedded, training=False, attn_dropout=0.0, ffn_dropout=0.0)
        preds = np.argmax(probs, axis=1)
        
        return preds, probs, attn_weights


    def AME_Encoder(self, x):
        """Compute the AME score of x (Cython kernel when available, numpy fallback)."""
        eps = 1e-5
        x = self._sanitize_string_chars(x)

        if len(x) == 0:
            print('[!] X size is 0, AME Will be replaced by minimum confidence threshold')
            return self.attn_dropout_rate

        # Optimized AME_Encoder for Transformer
        x = np.asarray(x)        
        if _OPT_AVAILABLE and np.asarray(x).ndim == 2:
            AME = optimized_ame_encoder(np.asarray(x, dtype=np.float64))  
            if np.isnan(AME) or np.isinf(AME):
                return
            return AME

        X = np.asarray(x)
        # Regular AME Equations, higher AME provides capabilities for the model to experience errors during abstraction
        # Lower AME means lower chance for un optimal abstraction.
        try:
            gradient = np.gradient(x, axis=-1)
        except:
            x = np.reshape(x, (x.shape[0], -1))
            gradient = np.gradient(x, axis=-1, edge_order=1)

        if len(X.shape) > 1:
            grad_energy = np.mean(np.linalg.norm(gradient, axis=-1)) + eps       
            X_mag = np.mean(np.linalg.norm(X, axis=-1)) + eps
        else:
            grad_energy = np.mean(np.linalg.norm(gradient)) + eps  
            X_mag = np.mean(np.linalg.norm(X)) + eps

        AME = (np.log1p(X_mag) * np.log1p(grad_energy)) + eps
        return AME


    def anisotropy_measurement(self, x):
        """Measure the anisotropy of x (Cython kernel when available, numpy fallback)."""
        eps = 1e-5

        x = self._sanitize_string_chars(x)
        if _OPT_AVAILABLE:
            x = np.asarray(x)            
            x = x.reshape(x.shape[0], -1)
            return optimized_anisotropy(np.asarray(x, dtype=np.float64))

        try:
            gradient = np.gradient(x)
        except:
            x = np.reshape(x, (x.shape[0], -1))
            subnet = x[:min(10, x.shape[0]), :min(10, x.shape[1])]
            gradient = np.gradient(subnet.flatten())

        val = [np.linalg.norm(v) for v in gradient]
        anisotropy = np.std(val) / np.mean(val) + eps 

        return anisotropy



    # attention quality computing provides the transformer a robust geometric complexity alignment scalar,
    #  this scalar can be used to compute alpha for a much stable forward pass in scarce data environment, allowing it to complement with AWE MLP below.
    def attention_quality_computing(self, attn_weights, AME=None, mask=None):
        """Score the quality of attention weights (sharpness / spread, optionally masked and scaled by AME) used to weight the loss and gradients."""
        eps = 1e-5
        batch, heads, seq_len, _ = attn_weights.shape

        if mask is not None:
            mask_expanded = np.broadcast_to(
                mask, (batch, heads, seq_len, seq_len)
            )
            # inplace operations, ARM64 NEON works better on contiguous memory
            attn_weights = attn_weights * mask_expanded
            row_sums     = attn_weights.sum(axis=-1, keepdims=True) + eps
            attn_weights = attn_weights / row_sums

        # fuse AME and anisotropy into single gradient pass
        # instead of calling AME_Encoder + anisotropy_measurement separately
        # both call np.gradient internally — compute once, reuse
        unsuitable_shape_condition = (
            attn_weights.shape[0] % 2 != 0,
            attn_weights.shape[1] % 2 != 0,   
            attn_weights.shape[2] % 2 != 0, 
            attn_weights.shape[3] % 2 != 0,    
        )
        if unsuitable_shape_condition:
            return self.robust_attention_quality_computing(attn_weights, AME=AME, mask=mask)
        else:
            gradient = np.gradient(attn_weights)

        # AME inline — avoids second np.gradient call in AME_Encoder
        # gradient is already a list of arrays, one per dimension
        # stacked into single array for vectorized norm — ARM64 NEON friendly
        if AME is None:
            grad_stack  = np.stack([g.ravel() for g in gradient])  # (ndim, N)
            grad_norms  = np.linalg.norm(grad_stack, axis=1)       # (ndim,) — one NEON call
            grad_energy = grad_norms.mean()
            X_mag       = np.linalg.norm(attn_weights.ravel())  / attn_weights.size
            AME = np.log1p(X_mag) * np.log1p(grad_energy)

        AMR = 1.0 / (1.0 + np.exp(-float(AME)))

        # anisotropy inline — reuses grad_norms, no second gradient call
        anisotropy_val = grad_norms.std() / (grad_norms.mean() + eps)

        # fuse entropy + max + var into single pass over attn_weights
        # avoids 3 separate scans of same array
        flat        = attn_weights.reshape(batch * heads * seq_len, seq_len)

        # entropy — one pass
        log_w       = np.log(flat + eps)
        entropy     = -(flat * log_w).sum(axis=-1)            # (B*H*T,)
        norm_entropy = 1.0 - entropy.mean() / np.log(seq_len + eps)

        # max — same flat array
        avg_max     = flat.max(axis=-1).mean()

        # var — same flat array
        norm_var    = np.clip(flat.var() * seq_len, 0.0, 1.0)

        # quality score
        qualified     = (1.0 - AMR) + eps * anisotropy_val
        quality_score = (qualified * norm_entropy +
                        qualified * avg_max +
                        anisotropy_val * norm_var)

        return float(np.clip(quality_score, 0.0, 1.0)) 


    def robust_attention_quality_computing(self, attn_weights, AME=None, mask=None):
        """NaN-safe version of attention_quality_computing that also uses anisotropy."""
        eps = 1e-5
        eps = 1e-5
        batch, heads, seq_len, _ = attn_weights.shape    

        if AME is None:
            AME = self.AME_Encoder(attn_weights)

        anisotropy = self.anisotropy_measurement(attn_weights)

        entropy = -np.sum(attn_weights * np.log(attn_weights + eps), axis=-1)
        max_entropy = np.log(seq_len)
        norm_entropy = 1.0 - (np.mean(entropy) / max_entropy)

        max_attn = np.max(attn_weights, axis=-1)
        avg_max = np.mean(max_attn)

        var_attn = np.var(attn_weights)
        norm_var = np.clip(var_attn * seq_len, 0, 1)

        AMR = 1.0 / (1.0 + np.exp(-AME))  # abstract modelling rate
        qualified = (1.0 - AMR) + eps * anisotropy

        quality_score = qualified * norm_entropy + qualified * avg_max + anisotropy * norm_var
        dynamic_alpha = np.clip(quality_score, 0, 1.0)

        return dynamic_alpha
