"""KNNAugmentedTransformer: transformer with per-head k-NN memory."""

import numpy as np
from .geometric_weight_shaping import GeometricWeightShaping
from .per_head_memory import PerHeadMemory
from .transformer import Transformer


class KNNAugmentedTransformer(Transformer):
    """Transformer whose attention is augmented with an external per-head k-NN memory.

    Each head retrieves the k closest stored keys/values (PerHeadMemory / HNSW) and blends the memory
    attention with local attention through a learned gate.
    """
    def __init__(self, *args, memory_capacity=4000, memory_k=8, memory_metric='euclidean', 
                 mem_gate_init_bias=3.0, memory_min_neighbors=None, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.head_dim = self.d_model // self.n_heads
        self.memory_capacity = memory_capacity
        self.memory = self.memory = PerHeadMemory(self.n_heads, self.head_dim, capacity=self.memory_capacity,
                                    memory_k=memory_k, metric=memory_metric)

        self.memory_min_neighbors = memory_min_neighbors if memory_min_neighbors is not None else max(1, memory_k // 2)
        self.mem_gate = np.full(self.n_heads, mem_gate_init_bias)
        self.memory_k = memory_k
        self.memory_write_enabled = True

    @staticmethod
    def memory_attention(Q, K_mem, V_mem, mem_mask):
        """Attention of queries over the retrieved memory keys/values (masked by mem_mask); returns the memory output and weights."""
        b, h, s, d = Q.shape
        k = K_mem.shape[3]
        assert K_mem.shape == (b, h, s, k, d) and V_mem.shape == K_mem.shape, (Q.shape, K_mem.shape, V_mem.shape)
        assert mem_mask.shape == (b, h, s, k), mem_mask.shape
        scale = 1.0 / np.sqrt(d)
        
        scores = np.einsum('bhsd,bhskd->bhsk', Q, K_mem) * scale
        scores = np.where(mem_mask > 0, scores, -1e9)
        scores = scores - scores.max(axis=-1, keepdims=True)

        exp_scores = np.exp(scores) * mem_mask
        weights = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-8)
        out = np.einsum('bhsk,bhskd->bhsd', weights, V_mem)

        return weights, out

    @staticmethod
    def memory_attention_backward(d_mem_out, Q, k_mem, V_mem, mem_weights):
        """Gradients of the memory-attention branch w.r.t. Q, the memory keys and the memory values."""
        d_k = Q.shape[-1]
        scale = 1.0 / np.sqrt(d_k)

        d_weights = np.einsum('bhsd,bhskd->bhsk', d_mem_out, k_mem)
        d_scores = mem_weights * (d_weights - np.sum(d_weights * mem_weights, axis=-1, keepdims=True))
        d_scores = d_scores * scale

        d_Q = np.einsum('bhsk,bhskd->bhsd', d_scores, k_mem)
        return d_Q

    def _sigmoid(self, x):
        sigmoid = 1.0 / (1.0 + np.exp(-x))
        return sigmoid

    def multi_head_attention(self, x, mask=None, alpha=None,
                              W_q_mix=None, W_k_mix=None, W_v_mix=None):
    
        """Multi-head attention that also queries the k-NN memory, mixes memory and local attention with the learned gate and writes new keys/values to memory."""
        batch_size, seq_len, d_model = x.shape
        if W_q_mix is None:
            one_minus = 1.0 - alpha
            W_q_mix = one_minus * self.W_q_fixed + alpha * self.W_q
            W_k_mix = one_minus * self.W_k_fixed + alpha * self.W_k
            W_v_mix = one_minus * self.W_v_fixed + alpha * self.W_v

        Q = np.einsum('bsd,hdm->bhsm', x, W_q_mix)
        K = np.einsum('bsd,hdm->bhsm', x, W_k_mix)
        V = np.einsum('bsd,hdm->bhsm', x, W_v_mix)

        self.cache['Q'] = Q
        self.cache['K'] = K
        self.cache['V'] = V
        self.cache['x_attn_input'] = x

        local_out, local_weights = self.attention(Q, K, V, mask)
        self.cache['local_out'] = local_out
        self.cache['attn_weights'] = local_weights

        if getattr(self, 'memory_read_enabled', True):
            K_mem, V_mem, mem_mask = self.memory.retrieve(Q)
        else:
            K_mem = V_mem = np.zeros(Q.shape[:3] + (self.memory_k, Q.shape[-1]), dtype=np.float32)
            mem_mask = np.zeros(Q.shape[:3] + (self.memory_k,), dtype=np.float32)

        mem_weights, mem_out = self.memory_attention(Q, K_mem, V_mem, mem_mask)
        self.cache['K_mem'] = K_mem
        self.cache['V_mem'] = V_mem
        self.cache['mem_mask'] = mem_mask
        self.cache['mem_out'] = mem_out
        self.cache['mem_weights'] = mem_weights

        has_memory = (mem_mask.sum(axis=-1, keepdims=True) >= self.memory_min_neighbors).astype(np.float32)
        gate = self._sigmoid(self.mem_gate).reshape(1, self.n_heads, 1, 1)
        gate_used = has_memory * gate + (1.0 - has_memory) * 1.0
        self.cache['gate'] = gate
        self.cache['has_memory'] = has_memory

        attn_output = gate_used * local_out + (1.0 - gate_used) * mem_out
        self.cache['attn_output'] = attn_output

        attn_output_c = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        self.cache['attn_concat'] = attn_output_c

        output = np.matmul(attn_output_c, self.W_o)
        self.cache['attn_out'] = output

        return output, local_weights

    def predict(self, input_ids, embedded=False):
        """Inference as in Transformer.predict, using the memory-augmented forward pass."""
        prev = self.memory_write_enabled
        self.memory_write_enabled = False
        try:
            return super().predict(input_ids, embedded=embedded)
        finally:
            self.memory_write_enabled = prev

    def _gate_and_local_mem_grads(self, d_combined):
        gate = self.cache['gate']
        has_memory = self.cache['has_memory']
        local_out = self.cache['local_out']
        mem_out = self.cache['mem_out']

        d_local_out = d_combined * (has_memory * gate + (1.0 - has_memory))
        d_mem_out = d_combined * (has_memory * (1.0 - gate))

        d_gate_pre = np.sum(d_combined * (local_out - mem_out) * has_memory, axis=(0, 2, 3))
        gate_flat = gate.reshape(-1)
        d_gate_logit = d_gate_pre * gate_flat * (1.0 - gate_flat)

        return d_local_out, d_mem_out, d_gate_logit
        
    def dynamic_backward(self, d_logits, lr=0.01, max_norm=1.0):
        """Dynamic-alpha backward pass including the memory-gate and memory-attention gradients."""
        d_output = d_logits
        alpha = self.cache.get('alpha', 1.0)

        d_Wo = np.dot(self.cache['x_pooled'].T, d_output)

        d_pooled = np.dot(d_output, self.output.T)

        mask = self.cache['mask']
        if mask is not None:
            token_mask = mask[:, 0, 0, :, np.newaxis]
            lengths = token_mask.sum(axis=1, keepdims=True)
            d_x = (d_pooled[:, np.newaxis, :] / (lengths + 1e-6)) * token_mask
        else:
            d_x = np.repeat(d_pooled[:, np.newaxis, :] / self.cache['seq_len'], self.cache['seq_len'], axis=1)

        d_x = self.layer_norm_backward(d_x, self.cache['x_ln2_input'],
                                        self.ln2_scale, self.ln2_shift,
                                        mean=self.cache.get('ln2_mean'),
                                        var=self.cache.get('ln2_var'))

        d_ffn = d_x
        d_ffn2 = np.sum(np.matmul(self.cache['ffn_act'].transpose(0, 2, 1), d_ffn), axis=0)

        d_ffn_act = np.matmul(d_ffn, self.ffn2.T)
        ffn_drop_mask = self.cache.get('ffn_drop_mask')
        if ffn_drop_mask is not None:
            d_ffn_act = d_ffn_act * ffn_drop_mask / (1.0 - self.ffn_dropout_rate)

        d_ffn_pre = d_ffn_act * (self.cache['ffn_pre'] >= 0)

        d_prev = np.matmul(d_ffn_pre, self.ffn1.T)
        d_ffn1 = np.sum(np.matmul(self.cache['ffn_input'].transpose(0, 2, 1), d_ffn_pre), axis=0)

        d_ln1 = self.layer_norm_backward(
            d_prev + d_x, self.cache['x_ln1_input'], self.ln1_scale, self.ln1_shift
        )

        d_residual = d_ln1
        d_attn = d_ln1
        dx = d_prev + d_residual  # noqa: F841 (kept for parity with your original)

        attn_drop_mask = self.cache.get('attn_drop_mask')
        if attn_drop_mask is not None:
            d_attn = d_attn * attn_drop_mask / (1.0 - self.attn_dropout_rate)

        d_Wo_attn = np.sum(np.matmul(self.cache['attn_concat'].transpose(0, 2, 1), d_attn), axis=0)

        d_attn_concat = np.matmul(d_attn, self.W_o.T)
        batch, seq_len, _ = d_attn_concat.shape
        d_head = self.n_heads
        d_dim = self.d_model // self.n_heads
        d_combined = d_attn_concat.reshape(batch, seq_len, d_head, d_dim).transpose(0, 2, 1, 3)

        # ---- NEW: split across gate, get memory-branch Q gradient ----
        d_local_out, d_mem_out, d_gate_logit = self._gate_and_local_mem_grads(d_combined)

        V = self.cache['V']
        K = self.cache['K']
        Q = self.cache['Q']
        weight = self.cache['attn_weights']

        d_V = np.matmul(weight.transpose(0, 1, 3, 2), d_local_out)
        d_weights = np.matmul(d_local_out, V.transpose(0, 1, 3, 2))
        d_scores = weight * (d_weights - np.sum(d_weights * weight, axis=-1, keepdims=True))
        d_scores /= np.sqrt(Q.shape[-1])
        d_Q_local = np.matmul(d_scores, K)
        d_K = np.matmul(d_scores.transpose(0, 1, 3, 2), Q)

        d_Q_mem = self.memory_attention_backward(
            d_mem_out, Q, self.cache['K_mem'], self.cache['V_mem'], self.cache['mem_weights']
        )
        d_Q = d_Q_local + d_Q_mem
        # ---- end new ----

        x = self.cache['x_attn_input']
        d_W_q = np.einsum('bsd, bhsm->hdm', x, d_Q)
        d_W_k = np.einsum('bsd, bhsm->hdm', x, d_K)
        d_W_v = np.einsum('bsd, bhsm->hdm', x, d_V)

        d_x_q = np.einsum('bhsm, hdm->bsd', d_Q, self.W_q)
        d_x_k = np.einsum('bhsm, hdm->bsd', d_K, self.W_k)
        d_x_v = np.einsum('bhsm, hdm->bsd', d_V, self.W_v)

        d_x_attn_input = d_x_q + d_x_k + d_x_v
        d_x_total = d_x_attn_input + d_residual

        input_ids = self.cache.get('input_ids')

        grads = {
            'output': d_Wo,
            'ffn2': d_ffn2,
            'ffn1': d_ffn1,
            'W_o': d_Wo_attn,
            'W_q': alpha * d_W_q,
            'W_k': alpha * d_W_k,
            'W_v': alpha * d_W_v,
            'mem_gate': d_gate_logit,  # NEW
        }
        grads, norm = self.clip_gradients(grads, max_norm)

        self.output = self.apply_update(self.output, grads['output'], lr)
        self.ffn2 = self.apply_update(self.ffn2, grads['ffn2'], lr)
        self.ffn1 = self.apply_update(self.ffn1, grads['ffn1'], lr)
        self.W_o = self.apply_update(self.W_o, grads['W_o'], lr)
        self.W_q = self.apply_update(self.W_q, grads['W_q'], lr)
        self.W_k = self.apply_update(self.W_k, grads['W_k'], lr)
        self.W_v = self.apply_update(self.W_v, grads['W_v'], lr)
        self.mem_gate = self.apply_update(self.mem_gate, grads['mem_gate'], lr)  # NEW

        if input_ids is not None:
            emb_norm = np.linalg.norm(d_x_total)
            emb_coef = min(1.0, max_norm / (emb_norm + 1e-6))
            flat_ids = input_ids.flatten()
            flat_grads = d_x_total.reshape(-1, self.d_model) / self.cache['seq_len']
            np.add.at(self.token_embedding, flat_ids, -lr * emb_coef * flat_grads)
            self.pos_embedding[:seq_len] -= lr * emb_coef * d_x_total.mean(axis=0)
        else:
            self.pos_embedding[:seq_len] -= lr * d_x_total.mean(axis=0)
            norm = d_x_total

        return norm

    def fixed_attention_backward(self, d_logits, lr=0.01, max_norm=1.0):
        """Fixed-attention backward pass including the memory-gate and memory-attention gradients."""
        d_output = d_logits

        d_Wo = np.dot(self.cache['x_pooled'].T, d_output)
        d_pooled = np.dot(d_output, self.output.T)

        d_x = np.repeat(d_pooled[:, np.newaxis, :] / self.cache['seq_len'], self.cache['seq_len'], axis=1)
        d_x = self.layer_norm_backward(d_x, self.cache['x_ln1_input'],
                                        self.ln1_scale, self.ln1_shift,
                                        mean=self.cache.get('ln1_mean'),
                                        var=self.cache.get('ln1_var'))

        d_ffn = d_x
        d_ffn2 = np.sum(np.matmul(self.cache['ffn_act'].transpose(0, 2, 1), d_ffn), axis=0)

        d_ffn_act = np.matmul(d_ffn, self.ffn2.T)
        ffn_drop_mask = self.cache.get('ffn_drop_mask')
        if ffn_drop_mask is not None:
            d_ffn_act = d_ffn_act * ffn_drop_mask / (1.0 - self.ffn_dropout_rate)

        d_ffn_pre = d_ffn_act * (self.cache['ffn_pre'] >= 0)
        d_prev = np.matmul(d_ffn_pre, self.ffn1.T)
        d_ffn1 = np.sum(np.matmul(self.cache['ffn_input'].transpose(0, 2, 1), d_ffn_pre), axis=0)

        d_x = self.layer_norm_backward(d_x - self.cache['attn_out'],
                                        self.cache['x_ln1_input'],
                                        self.ln1_scale, self.ln1_shift)

        d_ffn = d_x
        d_residual_ffn = d_ffn
        dx = d_prev + d_residual_ffn
        d_attn = dx

        attn_drop_mask = self.cache.get('attn_drop_mask')
        if attn_drop_mask is not None:
            d_attn = d_attn * attn_drop_mask / (1.0 - self.attn_dropout_rate)

        d_Wo_attn = np.sum(np.matmul(self.cache['attn_concat'].transpose(0, 2, 1), d_attn), axis=0)

        # ---- NEW: gate still trains in fixed mode (Q/K/V stay frozen, gate doesn't) ----
        batch, seq_len, _ = d_attn.shape
        d_head = self.n_heads
        d_dim = self.d_model // self.n_heads
        d_combined = d_attn.reshape(batch, seq_len, d_head, d_dim).transpose(0, 2, 1, 3)
        _, _, d_gate_logit = self._gate_and_local_mem_grads(d_combined)
        # ---- end new ----

        grads = {
            'output': d_Wo,
            'ffn2': d_ffn2,
            'ffn1': d_ffn1,
            'W_o': d_Wo_attn,
            'mem_gate': d_gate_logit,  # NEW
        }
        grads, norm = self.clip_gradients(grads, max_norm)

        self.output = self.apply_update(self.output, grads['output'], lr)
        self.ffn2 = self.apply_update(self.ffn2, grads['ffn2'], lr)
        self.ffn1 = self.apply_update(self.ffn1, grads['ffn1'], lr)
        self.W_o = self.apply_update(self.W_o, grads['W_o'], lr)
        self.mem_gate = self.apply_update(self.mem_gate, grads['mem_gate'], lr)  # NEW

        return d_x

    def populate_memory(self, X, batch_size=32, max_vectors_per_head=2000, seed=0):
        """Fill the kNN memory once, after training, from TRAINING windows only."""
        X = np.asarray(X, dtype=np.float64)
        n_use = min(len(X), max(1, max_vectors_per_head // X.shape[1]))
        sel = np.random.default_rng(seed).choice(len(X), n_use, replace=False)
        self.memory = PerHeadMemory(self.n_heads, self.head_dim,
                                    capacity=max_vectors_per_head, memory_k=self.memory_k)
        self.memory_read_enabled = False          # don't search while still filling
        try:
            for s in range(0, n_use, batch_size):
                self.forward_continuous(X[sel[s:s+batch_size]], training=False,
                                        attn_dropout=0.0, ffn_dropout=0.0)
                self.memory.write(self.cache['K'], self.cache['V'])
        finally:
            self.memory_read_enabled = True

    def train(self, input_ids_list, y_true_list, epochs=100, mode=None,
            lr=0.01, embedded=False, max_norm=1.0,
            schedule='cosine_warmup', pad_token_id=0, batch_size=None):

        """Training loop of the parent Transformer, adapted for the memory-augmented model."""
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

                self.memory.write(self.cache['K'], self.cache['V'])
            return losses, accs
        else:
            print(f'[!] Transformer Training not allowed due to low AMR or invalid value, AMR: {AMR}. Skipping training.')
            return None, None



    def train_step(self, input_ids, epoch, y_true, lr=0.01, AME=None,
                mode=None, embedded=False, max_norm=1.0,
                pad_token_id=0, y_true_smooth=None):

        """One training step with the memory-augmented forward/backward passes. Returns the loss."""
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
