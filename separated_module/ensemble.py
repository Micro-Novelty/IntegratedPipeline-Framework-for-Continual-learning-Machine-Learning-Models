"""WeightedEnsemblePredictor: combines MLP, transformer and LSTM predictions."""

import numpy as np
import traceback
from collections import Counter
from .common import _OPT_AVAILABLE, optimized_anisotropy, optimized_dynamic_weighted_ensemble
from .explainability import ExplainabilityModule
from .model_storage import ModelStorage
from .query_node import QueryNode


# weighted ensemble predictor that dynamically adjusts the weights of the transformer and MLP based on the input data's geometric complexity and the attention quality>
# allowing it to leverage the strengths of both models for improved performance across a wider range of data complexities.
class WeightedEnsemblePredictor:
    """Combines MLP, Transformer (and optional LSTM) probabilities into one prediction.

    Supports 'dynamic', attention-weighted and meta ensembles, learns/stores attention memories,
    calibrates the ensemble weight and hands results to the ExplainabilityModule.
    """
    def __init__(self, pipeline, distribution, memory_name):
        self.pipeline = pipeline
        self.storage = ModelStorage(pipeline, memory_name, db_path='activity_log.db')
        self.inference = distribution
        self.query_node = QueryNode(pipeline, memory_name, self.storage)

        self.transformer_weight = 0.5  # Initial equal weight
        self.mlp_weight = 0.5 # initial equal mlp weight
        self.calibration_history = []
        self.explainer = ExplainabilityModule(pipeline, self)
        self.memory_name = memory_name
        self.db_path = 'activity_log.db'

        self.error_counts = None
        self.pred_counts = None
        self.error_decay = 0.85

        self.self_attn_weights = None

        if not self.storage.memory_exists(self.memory_name, type='Transformer'):
            self.memory = {}
        else:
            self.memory = self.storage.memory_retrieval(self.memory_name, type_func='Transformer', verbose=True)

    def _get_lstm_probs(self, input_ids, X_mlp, label_bins=None, confidence_level=0.90):
        """
        Convert LSTMEngine prediction output into a probability
        distribution compatible with trans_probs and mlp_probs
        for use in _dynamic_weighted_ensemble.

        Returns:
            lstm_probs : np.ndarray (batch_size, n_classes) or None if engine not ready
            lstm_weight_hint : float — calibrated confidence scalar for weighting
        """
        engine = self.pipeline.lstm_engine

        # guard — engine must be calibrated before predict() is callable
        if engine is None or engine.residual_std is None:
            print('[-] LSTM engine not ready, skipping LSTM probs')
            return None, 0.0

        # X_mlp is (batch, features) — LSTM expects (T, input_size) per sample
        # treating each feature vector as a single timestep sequence
        if X_mlp.ndim == 1:
            X_mlp = X_mlp.reshape(1, -1)

        batch_size = X_mlp.shape[0]
        n_classes = self.pipeline.model2.output.shape[1]  # align with transformer output

        lstm_probs = np.zeros((batch_size, n_classes))
        overall_confidences = []

        for i in range(batch_size):
            # shape each sample into (T=1, input_size) — single timestep
            x_seq = X_mlp[i].reshape(1, -1)

            try:
                result = engine.predict(
                    x_seq,
                    label_bins=label_bins,
                )
            except AssertionError:
                # calibrate() not called yet — skip gracefully
                print(f'[-] LSTM engine not calibrated for sample {i}, skipping')
                return None, 0.0

            # result['prediction'] is (T,) — take last timestep as scalar score
            raw_score = float(result['prediction'][-1])
            overall_confidences.append(result['overall'])

            # convert scalar score to class probabilities
            # label_confidence gives {label: prob} if label_bins were passed
            if result['label_confidence'] is not None:
                # map label_bins order to class indices
                label_probs = list(result['label_confidence'].values())
                n_label = len(label_probs)

                row = np.zeros(n_classes)
                row[:min(n_label, n_classes)] = label_probs[:n_classes]

                # renormalize in case n_label != n_classes
                row_sum = row.sum()
                if row_sum > 0:
                    row /= row_sum
                else:
                    row[0] = 1.0  # fallback — assign all mass to class 0

            else:
                # no label_bins — use mc_confidence at last timestep as a
                # soft signal: spread probability mass using mc_mean as logit
                mc_mean_last = float(result['mc_mean'][-1])
                mc_conf_last = float(result['mc_confidence'][-1])

                # build a soft peaked distribution using the score as a logit
                logits = np.full(n_classes, -mc_conf_last)
                target_class = int(np.clip(round(raw_score), 0, n_classes - 1))
                logits[target_class] = mc_conf_last

                # softmax here 
                logits -= logits.max()
                row = np.exp(logits)
                row /= row.sum()

            lstm_probs[i] = row

        # weight hint — average overall confidence across batch
        # lower residual_std = engine is well calibrated = higher weight earned
        mean_overall = float(np.mean(overall_confidences))
        lstm_weight_hint = mean_overall / (1.0 + engine.residual_std)

        return lstm_probs, lstm_weight_hint


    def attention_memory_gate(self, probs, x):
        """Look for a stored attention memory matching input x; returns (texts, mlp_probs, trans_probs, attn_weights) when found."""
        memory = self.memory
        cache_attn_memory = [key for key, (_, inp, _, _, _) in memory.items() if key.startswith('TA') and self.pipeline.cosine_similarity(x, inp) >= 0.85]

        if cache_attn_memory:
            print('[+] Found matching attention memory!')
            for memo in cache_attn_memory:
                texts, _, x2, x3, x4 = memory[memo]

            return texts, x2, x3, x4

        else:
            print('🔄 No Matching Attention Weights!')
            if self.self_attn_weights is not None:
                print('|| Using current attention weights because of no matches found.')
                attn_weights = self.self_attn_weights
                return None, None, None, attn_weights

            return None, None, None, None

    def _sanitize_for_storage(self, obj, _depth=0, _max_depth=10):
        """
        Recursively strip Ellipsis objects and '...' string artifacts
        from any structure before saving to database/memory.

        """
        if _depth > _max_depth:
            print(f'[⚠️] _sanitize_for_storage: max depth {_max_depth} reached, '
                f'truncating to avoid infinite recursion')
            return None

        # Case 1 — literal Ellipsis object
        if obj is Ellipsis:
            print('[⚠️] _sanitize_for_storage: found literal Ellipsis, replacing with None')
            return None

        # Case 2 — string containing "..." artifacts
        if isinstance(obj, str):
            if obj.strip() == '...':
                return None
            if '...' in obj:
                cleaned = obj.replace('...', '').strip()
                if cleaned != obj:
                    print(f'[⚠️] _sanitize_for_storage: stripped "..." from string: '
                        f'"{obj[:40]}..." → "{cleaned[:40]}"')
                return cleaned if cleaned else None
            return obj

        # Case 3 — numpy array — check for object dtype containing Ellipsis
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                flat = obj.ravel()
                has_ellipsis = any(v is Ellipsis for v in flat)
                if has_ellipsis:
                    print(f'[⚠️] _sanitize_for_storage: array contains Ellipsis '
                        f'objects, replacing with 0.0')
                    cleaned = np.array([
                        0.0 if v is Ellipsis else v for v in flat
                    ]).reshape(obj.shape)
                    return cleaned
            # numeric arrays never contain Ellipsis
            return obj

        # Case 4 — dict — recurse into keys/values
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                # keys should never legitimately be Ellipsis or "..."
                if k is Ellipsis or (isinstance(k, str) and k.strip() == '...'):
                    print(f'[⚠️] _sanitize_for_storage: dropping key that is Ellipsis/"..."')
                    continue
                cleaned_v = self._sanitize_for_storage(v, _depth + 1, _max_depth)
                if cleaned_v is not None or v is None:
                    cleaned[k] = cleaned_v
            return cleaned

        # Case 5 — list/tuple — recurse into elements
        if isinstance(obj, (list, tuple)):
            cleaned = [
                self._sanitize_for_storage(item, _depth + 1, _max_depth)
                for item in obj
            ]
            # remove None entries that came FROM ellipsis stripping,
            # but preserve legitimately-None entries at the same position
            # by only dropping items that were Ellipsis/"..." originally
            result = [c for c, orig in zip(cleaned, obj)
                    if not (orig is Ellipsis or
                            (isinstance(orig, str) and orig.strip() == '...'))]
            return tuple(result) if isinstance(obj, tuple) else result

        # everything else (int, float, bool, None) — pass through unchanged
        return obj           

    def modular_attention_saving(self, text, X, X2, X3, X4):
        """Sanitise and store the supplied arrays (text, probabilities, attention weights) as an attention memory."""
        memory_name = self.memory_name

        clean_X = self._sanitize_for_storage(X)
        clean_X2 = self._sanitize_for_storage(X2)
        clean_X3 = self._sanitize_for_storage(X3)
        clean_X4 = self._sanitize_for_storage(X4)

        self.memory['TA'] = clean_X, text, clean_X2, clean_X3, clean_X4

        self.storage.save_model_dict(memory_name, self.memory, type='Transformer', model_type='attention')

        print('🚀 Memory Probability Added!')




    def explainability_prediction_batch(self, texts, mlp_probs, trans_probs, attn_weights, show_explanation=False):
        results = []
        for i, text in enumerate(texts):
            text_mlp_probs = mlp_probs[i] if i < len(mlp_probs) else mlp_probs
            text_trans_probs = trans_probs[i] if i < len(trans_probs) else trans_probs
            text_attn_weights = attn_weights[i] if i < len(attn_weights) else attn_weights

            result = self.predict_single(text, text_mlp_probs, text_trans_probs, text_attn_weights, show_explanation)
            results.append(result)
            explanation = result['explanation']
            print(f'[+] Explanation: {explanation}')
        
        return results
    

    def credibility_summarized_prediction(self, input_ids, mlp_probs, trans_probs, attn_weights, type=None):
        """Summarise predictions with their credibility (from the combined MLP/Transformer evidence) and save the attention memory."""
        if type == 'Transformer':
            texts, mlp_probs, trans_probs, attn_weights = self.attention_memory_gate(input_ids)
            if not texts:
                texts = self.pipeline.texts
        else:
            texts = self.pipeline.texts

        results = self.explainability_prediction_batch(texts, mlp_probs, trans_probs, attn_weights)

        # Calculate summary
        predictions = [r['prediction'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        distribution = Counter(predictions)
        
        print("\n📊 Batch Summary:")
        print(f"   Total: {len(results)} predictions")
        print(f"   Avg Confidence: {np.mean(confidences):.1%}")
        print(f"   Distribution: {dict(distribution)}")
        
        self.modular_attention_saving(input_ids, texts, mlp_probs, trans_probs, attn_weights )


    def explain_past_memory(self, probs, input_ids):
        """Retrieve a stored attention memory for the input and explain the corresponding past prediction."""
        _, mlp_probs, trans_probs, attn_weights = self.attention_memory_gate(probs, input_ids)
        self.self_attn_weights = attn_weights

        self_attn_weights = self.self_attn_weights

        if trans_probs is not None:
            print('[+] Attention memory retrieved! ')
            method = 'memory_retrieval'

            self.credibility_summarized_prediction(input_ids, mlp_probs, trans_probs, attn_weights, type='pipeline')
            ensemble_probs = self._dynamic_weighted_ensemble(
                trans_probs, mlp_probs, attn_weights, input_ids
            )

            return ensemble_probs
        else:
            print("[-] Ambiguity present, Requesting peer assistance... ")
        
            try:
                probs = self.inference._handle_peer_agent_request(probs, self_attn_weights, input_ids, type='DevicePeer', agreement=False)
                return probs
            except Exception as e:
                print(f'|| Error initiating peer request: {e}, returning regular probs.')
                self.inference.report_failure(id(self), 'processing', reason=f'{e}')                        

                return probs


    def predict_single(self, text, mlp_probs, trans_probs, attn_weights, show_explanation=True, batch_size=2):
        """Predict one text from its MLP/Transformer probabilities and attention weights, optionally with a printed explanation."""
        # small batch size is used to prevent memory overflow in explanation module when processing large attention weights, as it computes detailed explanations that can be memory intensive.
        cache = self.pipeline.cache
        if cache is not None and "lstm_result" in cache:
            lstm_result = self.pipeline.cache['lstm_result']
            result, explanation = self.explainer._get_prediction_details(text, mlp_probs, trans_probs, attn_weights, lstm_result=lstm_result, batch_size=batch_size)
        else:
            result, explanation = self.explainer._get_prediction_details(text, mlp_probs, trans_probs, attn_weights, batch_size=batch_size)            
        return {
            'prediction': result['final_label'],
            'confidence': result['final_confidence'],
            'explanation': explanation,
            'details': result
        }
    
    

    def explainability_prediction_batch(self, texts, mlp_probs, trans_probs, attn_weights, show_explanation=False):
        """Explain a batch: runs predict_single for each text. (Defined twice in the class; the later definition is the one in effect.)"""
        results = []
        for text in texts:
            result = self.predict_single(text, mlp_probs, trans_probs, attn_weights, show_explanation)
            results.append(result)
            explanation = result['explanation']
            print(f'[+] Explanation: {explanation}')
        
        return results
    


    def predict_ensemble(self, input_ids, X_mlp, y_true, method='dynamic', embedded=False):
        """Run the MLP and Transformer on the inputs and combine them with the chosen method ('dynamic', attention-weighted or meta). Returns (ensemble_probs, info dict)."""
        label_bins=None
        if self.pipeline.cache and 'label_bins' in self.pipeline.cache:
            print('[=] label_bins cache found!')
            label_bins = self.pipeline.cache['label_bins']

        AME = self.pipeline.model2.AME_Encoder(input_ids)

        trans_probs, attn_weights = self.pipeline.model2.forward(input_ids, AME=AME, embedded=embedded)
        mlp_probs = self.pipeline.model3.forward(X_mlp, y=y_true)
        lstm_probs, lstm_weight_hint = self._get_lstm_probs(input_ids, X_mlp, label_bins=label_bins)

        established_agreement = self.query_node._establish_node_connection("PredictEnsemble")
        
        if method == 'equal':
            # Simple average
            ensemble_probs = (trans_probs + mlp_probs) / 2
            
        elif method == 'confidence':
            # Weight by confidence (max probability)
            trans_conf = np.max(trans_probs, axis=1, keepdims=True)
            mlp_conf = np.max(mlp_probs, axis=1, keepdims=True)
            
            # Normalize weights
            total_conf = trans_conf + mlp_conf + 1e-8
            trans_weight = trans_conf / total_conf
            mlp_weight = mlp_conf / total_conf
            
            ensemble_probs = trans_weight * trans_probs + mlp_weight * mlp_probs
            
        elif method == 'dynamic':
            # Dynamic weighting based on agreement and attention
            ensemble_probs = self._dynamic_weighted_ensemble(
                trans_probs, mlp_probs, attn_weights, input_ids,
                lstm_probs=lstm_probs,
                lstm_weight_hint=lstm_weight_hint
            )              
            
        elif method == 'attention':
            # Use attention to weight transformer vs MLP
            ensemble_probs = self._attention_weighted_ensemble(
                trans_probs, mlp_probs, attn_weights
            )
            
        elif method == 'meta':
            # Meta-learner that decides weights
            ensemble_probs = self._meta_ensemble(
                trans_probs, mlp_probs, attn_weights, X_mlp,
                lstm_probs=lstm_probs,
                lstm_weight_hint=lstm_weight_hint
            )
        
        elif method == 'calibration':
            calibrated_weight = self.calibrate_weights(input_ids, X_mlp, y_true, step=3)  
            ensemble_probs = self._attention_weighted_ensemble(
                trans_probs, mlp_probs, calibrated_weight
            )
                                    
        else:
            print(f"[=] Unknown method: {method}")
            raise ValueError("Invalid ensemble method!")            

        
        if established_agreement and self.pipeline.show_explainability_details:
            print('[✅] Agreement established, generating explainability features.')
            try:
                print('=== COMPLETE EXPLAINABILITY PREDICTION ==') 
                self.credibility_summarized_prediction(input_ids, mlp_probs, trans_probs, attn_weights, type='pipeline')
            except Exception as e:
                print(f'[-] Cant get explainability features! : {e}')
                traceback.print_exc()
        else:
            print('[-] No agreement established, skipping explainability features.')


        try:
            ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)
        except:
            ensemble_probs = ensemble_probs / ensemble_probs.sum()

        if ensemble_probs is None or np.isnan(ensemble_probs).any() or np.isinf(ensemble_probs).any():
            print('[-] Ensemble probs is invalid , using MLP probs as.')
            return mlp_probs, {
            'transformer': trans_probs,
            'mlp': mlp_probs,
            'ensemble': mlp_probs,
            'method': None              
            }   

        return ensemble_probs, {
            'transformer': trans_probs,
            'mlp': mlp_probs,
            'ensemble': ensemble_probs,
            'method': method
        }
        
    def anisotropy_measurement(self, x):
        """Measure the anisotropy of x (Cython kernel when available, numpy fallback)."""
        eps = 1e-5
        if _OPT_AVAILABLE:
            x = np.asarray(x)            
            x = x.reshape(x.shape[0], -1)            
            return optimized_anisotropy(np.asarray(x, dtype=np.float64))

        try:
            try:
                grads = np.gradient(x)
                # Stack gradients into a single array of vectors and find the norm of each
                # automatically handles multi-dimensional arrays (e.g., 2D, 3D volumes)
                stacked_grads = np.stack(grads, axis=-1)
                norms = np.linalg.norm(stacked_grads, axis=-1)
                
                # Safely filter out potential NaNs or infs (common at array boundaries)
                valid_norms = norms[np.isfinite(norms)]
                
                if len(valid_norms) == 0:
                    return 0.0 # Return zero or appropriate default if no valid values exist
                    
                # Calculate statistics using the clean data
                std_val = np.std(valid_norms)
                mean_val = np.mean(valid_norms)
                
                anisotropy = std_val / (mean_val + eps)    
                if np.isnan(anisotropy) or np.isinf(anisotropy):
                    anisotropy = self.pipeline.confidence_threshold
            except:
                try:
                    gradient = np.gradient(x)
                except:
                    subnet = x[:min(10, x.shape[0]), :min(10, x.shape[1])]
                    gradient = np.gradient(subnet.flatten())

                val = [np.linalg.norm(v) for v in gradient]
                anisotropy = np.std(val) / np.mean(val) + eps 
        except Exception as e:
            print(f'[!] Cant calculate anisotropy due to: {e}')   
            anisotropy = self.pipeline.confidence_threshold

        return anisotropy
        


    def _dynamic_weighted_ensemble(self, trans_probs, mlp_probs, attn_weights,
                                    input_ids, lstm_probs=None, lstm_weight_hint=0.0):
        # normalize all inputs to guaranteed 2D float64
        mean_pred_counts = np.mean(self.pipeline.model3.pred_counts)
        try:
            trans_probs = np.asarray(trans_probs, dtype=np.float64)
            mlp_probs   = np.asarray(mlp_probs,   dtype=np.float64)

            if trans_probs.ndim == 1: trans_probs = trans_probs[np.newaxis, :]
            if mlp_probs.ndim   == 1: mlp_probs   = mlp_probs[np.newaxis, :]

            B = trans_probs.shape[0]

            n_trans = trans_probs.shape[1]
            n_mlp   = mlp_probs.shape[1]

            has_lstm = lstm_probs is not None
            if has_lstm:
                lstm_probs = np.asarray(lstm_probs, dtype=np.float64)
                if lstm_probs.ndim == 1: lstm_probs = lstm_probs[np.newaxis, :]
                n_lstm = lstm_probs.shape[1]
            else:
                lstm_probs = np.zeros((B, 1), dtype=np.float64)  # dummy, not used
                n_lstm = 0

            n_classes = max(n_trans, n_mlp, n_lstm)
            print(f"🔄 Aligning classes: trans={n_trans} mlp={n_mlp} "
                f"lstm={n_lstm} → {n_classes}")

            if attn_weights is not None:
                attn_arr = np.asarray(attn_weights, dtype=np.float64)
                attn_flat = attn_arr.reshape(B, -1) if attn_arr.ndim > 1 \
                            else np.tile(attn_arr.ravel(), (B, 1))
            else:
                attn_flat = np.full((B, 1), 0.5, dtype=np.float64)

            # per-sample lstm_weight_hints
            if isinstance(lstm_weight_hint, (int, float)):
                lstm_weight_hints = np.full(B, float(lstm_weight_hint), dtype=np.float64)
            else:
                lstm_weight_hints = np.asarray(lstm_weight_hint, dtype=np.float64)
                if lstm_weight_hints.ndim == 0:
                    lstm_weight_hints = np.full(B, float(lstm_weight_hints))

                
            if _OPT_AVAILABLE:
                try:
                    print('[+] Using Optimized Dynamic weighted ensemble Method.')
                    return optimized_dynamic_weighted_ensemble(
                        np.ascontiguousarray(trans_probs),
                        np.ascontiguousarray(mlp_probs),
                        np.ascontiguousarray(attn_flat),
                        np.ascontiguousarray(lstm_probs),
                        np.ascontiguousarray(lstm_weight_hints),
                        float(self.pipeline.confidence_threshold),
                        has_lstm,
                        mean_pred_counts
                    )
                except Exception as e:
                    print(f'[=] Error in optimized dynamic weighted ensemble: {e}, using regular dynamic ensemble method.')
                    pass

            # pure Python fallback 
            ensemble = np.zeros((B, n_classes))
            for i in range(B):
                trans_row = np.zeros(n_classes)
                mlp_row   = np.zeros(n_classes)
                trans_row[:n_trans] = trans_probs[i]
                mlp_row[:n_mlp]     = mlp_probs[i]
                trans_row /= trans_row.sum() + 1e-8
                mlp_row   /= mlp_row.sum()   + 1e-8

                trans_pred = int(np.argmax(trans_probs[i]))
                mlp_pred   = int(np.argmax(mlp_probs[i]))
                agreement  = 1.0 if trans_pred == mlp_pred else 0.3

                attn         = attn_flat[i]
                attn_focus   = float(np.std(attn)) if attn.size > 1 else 0.5
                attn_growth  = 1.0 / (1.0 + np.exp(-attn_focus))
                anisotropy   = self.anisotropy_measurement(attn.reshape(1, -1))
                attn_limit   = (1.0 - attn_focus + attn_growth) * anisotropy
                trans_cf     = attn_growth + attn_limit * attn_focus

                mlp_entropy  = -np.sum(mlp_probs[i] * np.log(mlp_probs[i] + 1e-8))
                mlp_cf       = 1.0 / (1.0 + mlp_entropy)

                tw = trans_cf * (1.0 + agreement) / 2.0
                mw = mlp_cf  * mean_pred_counts * (1.0 + agreement) / 2.0

                if has_lstm:
                    lstm_row = np.zeros(n_classes)
                    lstm_row[:n_lstm] = lstm_probs[i]
                    lstm_row /= lstm_row.sum() + 1e-8
                    lstm_pred = int(np.argmax(lstm_probs[i]))
                    la = 1.0 if (lstm_pred == trans_pred or lstm_pred == mlp_pred) \
                        else self.pipeline.confidence_threshold
                    lw    = float(lstm_weight_hints[i]) * (1.0 + la) / 2.0
                    total = tw + mw + lw + 1e-8
                    ensemble[i] = (tw/total) * trans_row + \
                                (mw/total) * mlp_row   + \
                                (lw/total) * lstm_row
                else:
                    total = tw + mw + 1e-8
                    ensemble[i] = (tw/total) * trans_row + (mw/total) * mlp_row

            return ensemble
        
        except Exception as e:
            print(f'[!] Cant do ensemble prediction due ensemble prediction due to: {e}, returning MLP probabilities')
            return mlp_probs.copy()


    
    def _attention_weighted_ensemble(self, trans_probs, mlp_probs, attn_weights):
        mean_pred_counts = np.mean(self.pipeline.model3.pred_counts)
        if attn_weights is None:
            return (trans_probs + mlp_probs) / 2
        
        batch_size = trans_probs.shape[0]
        ensemble = np.zeros_like(trans_probs)

        n_trans_classes = trans_probs.shape[1]        
        n_mlp_classes = mlp_probs.shape[1]
        
        anisotropy = self.anisotropy_measurement(attn_weights)            
        n_classes = max(n_trans_classes, n_mlp_classes)       
        for i in range(batch_size):
            trans_row = np.zeros(n_classes)
            mlp_row = np.zeros(n_classes)         
            trans_row[:n_trans_classes] = trans_probs[i]
            mlp_row[:n_mlp_classes] = mlp_probs[i]
            
            trans_row = trans_row / (trans_row.sum() + 1e-8)
            mlp_row = mlp_row / (mlp_row.sum() + 1e-8)

            if i < len(attn_weights):
                attn = attn_weights[i]              
                # Attention entropy: lower entropy = more focused = trust transformer more
                if attn.size > 0:
                    attn_flat = attn.flatten()
                    attn_entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-8)) / np.log(len(attn_flat))
                    trans_trust = 1.0 - attn_entropy  # 0 to 1
                else:
                    attn_focus = 1.0 / (1.0 + np.exp(-attn))
                    trans_trust = attn_focus * anisotropy
            else:
                attn_focus = 1.0 / (1.0 + np.exp(-attn))
                attn_limit = 1.0 - np.exp(-attn_focus)
                trans_trust = attn_limit * (1.0 - anisotropy)
            
            # MLP gets the rest
            mlp_trust = (1.0 - trans_trust) * mean_pred_counts

            try:
                ensemble[i] = trans_trust * trans_row + mlp_trust * mlp_row 
            except:
                ensemble = trans_trust * trans_row + mlp_trust * mlp_row 
        
        return ensemble
    
    def _meta_ensemble(self, trans_probs, mlp_probs, attn_weights, X_mlp, 
                         lstm_probs=None, lstm_weight_hint=0.0):
        lstm_row = None
        
        if trans_probs.ndim == 1: trans_probs = trans_probs[np.newaxis, :]
        if mlp_probs.ndim   == 1: mlp_probs   = mlp_probs[np.newaxis, :]

        B = trans_probs.shape[0]

        batch_size = trans_probs.shape[0]
        threshold_feature = 0.1 + self.pipeline.confidence_threshold
    
        n_trans_classes = trans_probs.shape[1]        
        n_mlp_classes = mlp_probs.shape[1]

        has_lstm = lstm_probs is not None
        if has_lstm:
            lstm_probs = np.asarray(lstm_probs, dtype=np.float64)
            if lstm_probs.ndim == 1: lstm_probs = lstm_probs[np.newaxis, :]
            n_lstm = lstm_probs.shape[1]
        else:
            lstm_probs = np.zeros((B, 1), dtype=np.float64)  # dummy, not used
            n_lstm = 0
 
        n_classes = max(n_trans_classes, n_lstm, n_mlp_classes)
        print(f"🔄 Aligning classes: trans={n_trans_classes} mlp={n_mlp_classes} "
            f"lstm={n_lstm} → {n_classes}") 
            
        if isinstance(lstm_weight_hint, (int, float)):
            lstm_weight_hints = np.full(B, float(lstm_weight_hint), dtype=np.float64)
        else:
            lstm_weight_hints = np.asarray(lstm_weight_hint, dtype=np.float64)
            if lstm_weight_hints.ndim == 0:
                lstm_weight_hints = np.full(B, float(lstm_weight_hints))

        # Create meta features
        meta_features = []
        for i in range(batch_size):
            trans_row = np.zeros(n_classes)
            mlp_row = np.zeros(n_classes)
             
            trans_row[:n_trans_classes] = trans_probs[i]
            mlp_row[:n_mlp_classes] = mlp_probs[i]
            if has_lstm:
                lstm_row = np.zeros(n_classes)
                lstm_row[:n_lstm] = lstm_probs[i]
                lstm_row = lstm_row / (lstm_row.sum() + 1e-8)

            trans_row = trans_row / (trans_row.sum() + 1e-8)
            mlp_row = mlp_row / (mlp_row.sum() + 1e-8)

            if lstm_row is None:
                features = [
                    np.max(trans_row),           # Transformer confidence
                    np.max(mlp_row),              # MLP confidence
                    np.std(trans_row),             # Transformer spread
                    np.std(mlp_row),               # MLP spread
                    1.0 if np.argmax(trans_row) == np.argmax(mlp_row) else 0.0,  # Agreement
                ]
            else:
                features = [
                    np.max(trans_row),           # Transformer confidence
                    np.max(mlp_row),              # MLP confidence
                    np.max(lstm_row),
                    np.std(trans_row),             # Transformer spread
                    np.std(mlp_row),               # MLP spread
                    np.std(lstm_row),
                    1.0 if np.argmax(trans_row) == np.argmax(mlp_row) else 0.0,  # Agreement
                ]    

            # Add attention stats if available
            if attn_weights is not None and i < len(attn_weights):
                attn = attn_weights[i]
                if attn.size > 0:
                    features.append(np.std(attn))
                    features.append(np.max(attn))
                else:
                    features.extend([threshold_feature, threshold_feature])
            else:
                features.extend([threshold_feature, threshold_feature])
            
            meta_features.append(features)
        
        meta_features = np.array(meta_features) 
        featured_AME = self.pipeline.AME_Encoder(meta_features) 
        AME_sigmoid = 1.0 / (1.0 + np.exp(-featured_AME))

        ensemble = np.zeros_like(trans_probs)
        
        for i in range(batch_size):
            # Calculate weight based on meta features
            trans_conf = meta_features[i, 0]
            mlp_conf = meta_features[i, 1]
            agreement_idx = 6 if has_lstm else 4
            agreement = meta_features[i, agreement_idx]
            
            trans_pred = int(np.argmax(trans_probs[i]))
            mlp_pred   = int(np.argmax(mlp_probs[i])) 
                                  
            # Boost weight when models agree
            base_weight = threshold_feature + AME_sigmoid * agreement
            
            # Adjust based on relative confidence
            if trans_conf > mlp_conf:
                trans_weight = base_weight
                mlp_weight = 1.0 - base_weight
            else:
                trans_weight = 1.0 - base_weight
                mlp_weight = base_weight
                
            if has_lstm:
                if lstm_row is None:
                    lstm_row = np.zeros(n_classes)
                    lstm_row[:n_lstm] = lstm_probs[i]
                    lstm_row /= lstm_row.sum() + 1e-8

                lstm_pred = int(np.argmax(lstm_probs[i]))
                la = 1.0 if (lstm_pred == trans_pred or lstm_pred == mlp_pred) \
                    else self.pipeline.confidence_threshold
                lw    = float(lstm_weight_hints[i]) * (1.0 + la) / 2.0
                total = trans_weight + mlp_weight + lw + 1e-8
                ensemble[i] = (trans_weight/total) * trans_row + \
                            (mlp_weight/total) * mlp_row   + \
                            (lw/total) * lstm_row
            else:
                try:
                    ensemble[i] = trans_weight * trans_row + mlp_weight * mlp_row 
                except:
                    ensemble = trans_weight * trans_row + mlp_weight * mlp_row                
        
        return ensemble
    
    def calibrate_weights(self, input_ids, X_mlp, y_true, step=3):
        """Search ensemble weights on validation data and return the best-performing weight."""
        print("\n🔧 Calibrating ensemble weights...")
        
        best_weight = self.pipeline.confidence_threshold + self.pipeline.confidence_threshold
        best_accuracy = 0

        AME = self.pipeline.model2.AME_Encoder(input_ids)

        # Try different weights
        for w in np.linspace(0, 1, 11):
            self.transformer_weight = w
            self.mlp_weight = 1 - w
            
            correct = 0
            total = 0
            for i in range(step):
                trans_probs, _ = self.pipeline.model2.forward(input_ids, AME=AME, embedded=True)
                mlp_probs = self.pipeline.mlp.forward(X_mlp)
                
                ensemble = w * trans_probs + (1-w) * mlp_probs
                preds = np.argmax(ensemble, axis=1)
                true = np.argmax(y_true, axis=1)
                
                correct += np.sum(preds == true)
                total += len(preds)
            
            accuracy = correct / total
            print(f" || Weight: {w:.1f}: Accuracy: {accuracy:.2%}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weight = w

        self.transformer_weight = best_weight
        self.mlp_weight = 1.0 - best_weight

        print(f"\n✅ Optimal weights: Transformer: {best_weight:.2f}, MLP={1-best_weight:.2f}")
        print(f"[-] Validation accuracy: {best_accuracy:.2%}")
        
        return best_weight
