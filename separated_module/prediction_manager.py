"""PipelinePredictionManager: regular / advanced prediction entry points."""

import numpy as np
import os
import pandas as pd
import sys
import traceback
from collections import Counter
from .common import APP_BASE_DIR
from .data_utils import NormalizationConfig, dynamic_test_size
from .transformer import Transformer
from .knn_transformer import KNNAugmentedTransformer


class PipelinePredictionManager:
    """User-facing prediction entry points for the IntegratedPipeline.

    Offers regular (MLP-first) and advanced (MLP + Transformer / LSTM / peers) prediction, KNN-transformer
    inference, MC-dropout uncertainty, class-reputation calibration and formatted result display.
    """
    def __init__(self, pipeline, label_csv='labels.csv', target_title='title', label='label'):
        self.pipeline = pipeline
        self.config = None

        try:
            print("📖 Loading labels for Prediction Manager from text file...")
            self.titles, self.y_raw, self.label_map = self.load_labels_from_csv(label_csv, target_title, label)
        except Exception as e:
            print(f"[=] Error loading labels: {e}")
            self.titles, self.y_raw, self.label_map = None, None, None

        if self.label_map is not None:   
            self.error_counts = np.zeros(len(self.label_map), dtype=np.float64)
            self.pred_counts  = np.zeros(len(self.label_map), dtype=np.float64)
            self.decay        = self.pipeline.error_decay   # how fast old errors fade — tunable
        else:
            raise Warning('[!] Label map is None, consider adding label map!')

        print(f"✅ Loaded {len(self.titles)} labeled examples")

    def create_sequences(self, data, window_size, horizon=1, target_col=0):
        """Sliding-window helper (same as data_utils.create_sequences)."""
        X, y = [], []
        for i in range(len(data) - window_size - horizon + 1):
            X.append(data[i : i + window_size])                                       # semua fitur
            y.append(data[i + window_size : i + window_size + horizon, target_col])   # cuma target
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)



    def prepare_dataset(self, df, feature_cols, target_col_name, window_size, horizon=1):
        """Normalise a DataFrame and build windowed (X, y) regression data, storing the NormalizationConfig."""
        if df is None or feature_cols is None or window_size is None:
            raise ValueError(f'[>] Some samples is missing! cannot create Dataset for regression Tasks!')

        data = df[feature_cols].values.astype(np.float32)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        
        self.config = NormalizationConfig(
            feature_cols=feature_cols,
            target_col_name=target_col_name,
            mean=mean,
            std=std
        )
        
        data_norm = self.config.normalize(data)
        X, y = self.create_sequences(data_norm, window_size, horizon, target_col=self.config.target_col_idx)
        
        return X, y, self.config


    def load_labels_from_csv(self, filename, target_title, label):
        """
        Load CSV from multiple common locations — no need to place
        file next to the script.

        Search order:
        1. Absolute path (if filename is already absolute)
        2. Current working directory
        3. Script directory
        4. User home directory
        5. Common data folders (Downloads, Documents, Desktop)
        """

        # build candidate paths
        candidates = []

        # 1 — absolute path as-is
        if os.path.isabs(filename):
            candidates.append(filename)
        else:
            # 2 — current working directory
            candidates.append(os.path.join(os.getcwd(), filename))

            # 3 — script directory
            try:
                script_dir = APP_BASE_DIR
                candidates.append(os.path.join(script_dir, filename))
            except NameError:
                pass

            # 4 — home directory
            candidates.append(os.path.join(os.path.expanduser('~'), filename))

            # 5 — common data folders
            home = os.path.expanduser('~')
            for folder in ['Downloads', 'Documents', 'Desktop', 'Data', 'data']:
                candidates.append(os.path.join(home, folder, filename))

            # 6 — sys.path entries (useful in notebooks)
            for p in sys.path:
                if p:
                    candidates.append(os.path.join(p, filename))

        # find first existing path
        filepath = None
        for candidate in candidates:
            if os.path.exists(candidate):
                filepath = candidate
                break

        if filepath is None:
            print(f"❌ Could not find '{filename}' in any of these locations:")
            for c in candidates[:6]:  # show first 6 only
                print(f"   {c}")
            print(f"\n💡 Tip: place your CSV in one of these folders or pass the full path:")
            print(f"   {os.getcwd()}\\{filename}")
            print(f"   {os.path.expanduser('~')}\\Downloads\\{filename}")
            return [], [], {}

        print(f"✅ Found CSV at: {filepath}")

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"❌ Failed to read CSV: {e}")
            return [], [], {}

        # validate columns exist
        missing = [c for c in [target_title, label] if c not in df.columns]
        if missing:
            print(f"❌ Missing columns: {missing}")
            print(f"   Available columns: {list(df.columns)}")
            return [], [], {}

        print(f"✅ Loaded CSV with columns: {list(df.columns)}")

        # drop rows with missing values in target columns
        before = len(df)
        df = df.dropna(subset=[target_title, label])
        dropped = before - len(df)
        if dropped > 0:
            print(f"⚠️ Dropped {dropped} rows with missing values")

        # extract and clean
        titles       = df[target_title].astype(str).str.strip('"').tolist()
        string_labels = df[label].astype(str).tolist()

        print(f"📊 Found {len(titles)} examples")
        print(f"📊 Labels: {set(string_labels)}")

        # create numeric label map
        unique_labels = sorted(set(string_labels))
        label_map     = {lbl: i for i, lbl in enumerate(unique_labels)}
        y             = [label_map[lbl] for lbl in string_labels]

        return titles, y, label_map


    def initialize_knn_transformer(self, vocab_size=1, task='classification', num_classes=None, d_model=32, n_heads=4, n_features=None, 
                                   memory_capacity=10000,
                                   memory_k=8, memory_metric='euclidean'):
        """Replace the pipeline's Transformer with a memory-augmented one (capacity, k and metric configurable)."""
        self.pipeline.model2 = Transformer(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    n_heads=n_heads,
                    num_classes=num_classes,
                    n_features=n_features,
                    task=task
            )
        self.pipeline.knn_tf = KNNAugmentedTransformer(
                                vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, num_classes=num_classes,
                                n_features=n_features, task=task,
                                memory_capacity=memory_capacity, memory_k=memory_k, 
                                memory_metric=memory_metric
        )

        print(f'[>] kNN Transformer has been initialized with memory: {memory_capacity}')

        return self.pipeline.knn_tf



    def knn_forward_inference(self, X, y, label_map=None, task='classification', training=False, train_epochs=100, batch_size=2, train_mode='fixed_backward', lr=0.1):
        """Run (and optionally train) the KNN-augmented transformer on X/y. Returns (trans_probs, attn_weights, third) where third is the KNN transformer or a regression prediction; ([], [], None) on failure."""
        trans_probs = None
        attn_weights = None
        use_embedded = True

        if label_map and isinstance(label_map, dict):
            reverse_map = {v: k for k, v in label_map.items()}
    
        if self.pipeline.knn_tf is None:
            raise ValueError('[>] kNN Transformer instance is None! consider initializing kNN Transformer Instance first.')

        try:
            if X is None or y is None:
                raise AttributeError('[!] Some Sample is None or missing!')

            if X is not None and y is not None:
                print(f"[🔍] Processing X and y samples for kNN Augmented Transformer...")
                if isinstance(X, (str, np.str_)):
                    clean_str = str(X).replace('[', '').replace(']', '')
                    X = np.fromstring(clean_str, sep=' ')
                if isinstance(X, np.ndarray) and np.issubdtype(X.dtype, np.character):
                    # catches arrays filled with string text
                    clean_str = ' '.join(X.astype(str).flatten()).replace('[', '').replace(']', '')
                    X = np.fromiter(
                            (x for x in clean_str.split() if x != "..."), dtype=float
                        )   

                if isinstance(y, (str, np.str_)):
                    clean_str = str(y).replace('[', '').replace(']', '')
                    y = np.fromstring(clean_str, sep=' ')
                if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.character):
                    # catches arrays filled with string text
                    clean_str = ' '.join(y.astype(str).flatten()).replace('[', '').replace(']', '')
                    y = np.fromiter(
                            (x for x in clean_str.split() if x != "..."), dtype=float
                        )

            if len(X) < 5:
                print(f"[⚠️] Warning: Only {len(X)} samples provided. Consider providing more samples for better prediction accuracy!")

            if 0 in X.shape:
                print(f"[⚠️] Warning: X has zero samples in the total shapes, X shapes: {X.shape}. Creating an empty array with shape (1, n_features) for processing..")
                X = np.empty((1, X.shape[1]))  # Created an empty array with shape (1, n_features)
                X = X.reshape(1, -1)  # Reshape to (1, n_features) if empty but has features
                
            X_train, X_test, y_train, y_test, X_val, y_val = self._prepare_train_val_split(
                X, y, min_val_per_class=5, min_frac=0.1, max_frac=0.3
            )


            AME = self.pipeline.AME_Encoder(X_train)
            if training:
                print('[>] Starting kNN Augmented Transformer Training...')
                sequence_ids_train = self.pipeline._features_to_sequence(X_train)
                if task == 'classification':
                    print('[>] Training kNN Transformer for classification task...')
                    self.pipeline.knn_tf.train(input_ids_list=sequence_ids_train, y_true_list=y, epochs=train_epochs, mode=train_mode, lr=lr, embedded=True, batch_size=batch_size)
                else:
                    print('[>] Training kNN Transformer for regression task...')
                    self.pipeline.knn_tf.train_regression(X_train, y_train, X_val, y_val, epochs=train_epochs, lr=lr, batch_size=batch_size)
                return trans_probs, attn_weights, self.pipeline.knn_tf
            else:
                print('[>] Processing kNN Augmented Transformer probabilities...')
                sequence_ids = self.pipeline._features_to_sequence(X)
                trans_probs, attn_weights = self.pipeline.knn_tf.forward(input_ids=sequence_ids, AME=AME, embedded=use_embedded)

            if label_map is None:
                print('[⚠️] > label_map is None, returning kNN Transformer probabilities without any label mapping...')
                return trans_probs, attn_weights, None
            else:
                trans_pred_indices = np.mean(trans_probs, axis=0)
                trans_pred_idx = np.argmax(trans_pred_indices)
                predict_confidence = trans_pred_indices[trans_pred_idx]
                prediction = reverse_map.get(trans_pred_idx, 'Unknown')
                print(f"[=] kNN Augmented Transformer Prediction: {prediction} with probability {predict_confidence:.2%}")
                return trans_probs, attn_weights, prediction
        
        except Exception as e:
            print(f'[>] Error initating kNN forward Inference: {e}')
            traceback.print_exc()
            return [], [], None



    def regular_prediction_method(self, titles=None, label_map=None, rules=None, X=None, y=None, show_proba=False, top_k=3, batch_size=2, use_transformer=True):
        """Standard prediction flow: encode inputs, run the MLP (optionally the Transformer), apply memory gates and calibration and print/return results."""
        try:
            dataset = None
            X_gen = None
            use_embedded = False
            attn_weights = None

            trans_probs = None
            mlp_probs = None
            target_probs = None

            print(f"\n[🚀] Regular Prediction Initiated...")
            self.pipeline.titles = titles
            self.pipeline.labels = label_map

            reverse_map = {v: k for k, v in label_map.items()}
            num_classes = self.pipeline._get_num_classes(label_map=label_map)

            if titles is not None and rules is not None:
                print(f"[🔍] Preparing data for {len(titles)} titles with {len(rules)} length of rules.")
                if X is None and y is None or X is None or y is None:
                    print('[🔄] Creating automatic X samples because X is not provided manually.')
                    dataset, X_gen = self.pipeline.data_preparation(titles, label_map)  
                    _, y, _, _ = self.pipeline.mlp_training_features(rules, dataset, label_map=label_map)                  
                else:
                    dataset, _ = self.pipeline.data_preparation(titles, label_map)
                    
            if X is not None or y is not None and isinstance(X, (np.ndarray, list)) and isinstance(y, (np.ndarray, list)) and len(X) > 0 and len(y) > 0:
                X_train, X_test, y_train, y_test, X_val, y_val = self._prepare_train_val_split(
                    X, y, min_val_per_class=5, min_frac=0.1, max_frac=0.3
                )
                
                onehot_validation = self.pipeline._validate_onehot(y_train)
                infer_y = self.infer_target_type(self.pipeline.vocab_size, y)
                if onehot_validation and infer_y == 'classification':
                    if num_classes != len(label_map):
                        num_classes = len(label_map)
                    if num_classes > np.max(y_train):
                        y_train = np.eye(num_classes)[np.asarray(y_train).astype(int)]
                        y_val = np.eye(num_classes)[np.asarray(y_val).astype(int)]
                    else:
                        print('[⚠️] Warning: Y onehot encoding fails, Returning y samples as is, This may cause Exploding Gradient in MLP Training!')
                        y_train = y_train
                        y_val = y_val

                X_mean = X_train.mean(axis=0)
                X_std = X_train.std(axis=0) + 1e-8

                X_train = (X_train - X_mean) / X_std
                X = (X_val - X_mean) / X_std
                y = y_val.copy()

            if dataset is not None:
                input_ids, _ = self.pipeline.input_encoding(dataset)
            else:
                input_ids = self.pipeline._features_to_sequence(X)
                
            if X is None and X_gen is not None:
                X = X_gen

            if isinstance(input_ids, (list, np.ndarray)):
                use_embedded = True

            if self.pipeline.cache and 'label_bins' in self.pipeline.cache:
                print('[=] label_bins cache found!')
                label_bins = self.pipeline.cache['label_bins']
                if X_gen is None:
                    X_gen = X

                lstm_probs, _ = self.pipeline.ensemble._get_lstm_probs(input_ids, X_gen, label_bins=label_bins)      
            else:
                lstm_probs = None


            if self.pipeline.use_transformer and self.pipeline.model2 is not None:
                print("[🔄] Using Transformer for probability calibration")

                if titles is not None and len(titles) > 0:
                    # Encode titles for transformer
                    input_ids_list = []
                    for title in titles:
                        # Handle both string and tuple inputs
                        if isinstance(title, tuple):
                            title = title[0]
                        # Encode to token IDs using pipeline's vocabulary
                    ids = self.pipeline.encode(title, self.pipeline.vocab)
                    input_ids_list.append(np.array(ids))
                
                    input_ids = np.array(input_ids_list)
                    AME = self.pipeline.model2.AME_Encoder(input_ids) 
                    try:
                        trans_probs, attn_weights = self.pipeline.model2.forward(input_ids, AME=AME, embedded=use_embedded)
                    except:
                        trans_probs, attn_weights = self.pipeline.model2.forward(input_ids, AME=AME, embedded=False)

                else:
                    AME = self.pipeline.model2.AME_Encoder(input_ids)
                    trans_probs, attn_weights = self.pipeline.model2.forward(input_ids, AME=AME, embedded=use_embedded)
            else:
                print("[⚡] Using MLP only for predictions")
                trans_probs,attn_weights = None, None
        
            if not hasattr(self.pipeline, 'tfidf') or self.pipeline.tfidf is None:
                self.pipeline.initialize_fitting(titles)
            
            # Prepare texts for MLP
            if titles is not None and len(titles) > 0:
                if isinstance(titles[0], tuple):
                    mlp_titles = [t[0] for t in titles]
                else:
                    mlp_titles = titles
                    
                X_tfidf = self.pipeline.tfidf.transform(mlp_titles).toarray() 
            else:
                X_tfidf = X

            # Forward pass through MLP
            mlp_probs = self.pipeline.model3.forward(X, y=y)
           
            # Validate all MLP predictions at once
            mlp_pred_indices = np.argmax(mlp_probs, axis=1)
            if num_classes <= 0:
                num_classes = self.pipeline._get_num_classes(mlp_probs=mlp_probs)

            valid_mask = mlp_pred_indices < num_classes
            if not np.all(valid_mask):
                invalid_count = np.sum(~valid_mask)
                # Replace invalid indices with argmax within valid range
                for i in range(len(mlp_pred_indices)):
                    valid_probs = mlp_probs[i][:num_classes] if num_classes > 0 else mlp_probs[i]
                    if len(valid_probs) > 0:
                        mlp_pred_indices[i] = int(np.argmax(valid_probs))
                    else:
                        mlp_pred_indices[i] = 0  # Default to first class   
                             
            results = []
            if titles is not None and len(titles) > 0:
                for i, title in enumerate(titles):
                    # Handle tuple inputs
                    if isinstance(title, tuple):
                        display_title = title[0]
                        expected_label = title[1] if len(title) > 1 else None
                    else:
                        display_title = title
                        expected_label = None
                    
                    # MLP prediction
                    mlp_class_idx = mlp_pred_indices[i] if i < len(mlp_pred_indices) else mlp_pred_indices[len(mlp_pred_indices)-1]
                    mlp_class_idx = min(mlp_class_idx, num_classes - 1)  # Clamped to valid range
                    
                    mlp_confidence = mlp_probs[i][mlp_class_idx] if i < len(mlp_probs) else mlp_probs[len(mlp_probs)-1]
                    mlp_label = reverse_map.get(mlp_class_idx, f"unknown_{mlp_class_idx}")
                    anisotropy = self.pipeline.anisotropy_measurement(input_ids)
                    anisotropic_rate = 1.0 / (1.0 + np.exp(-anisotropy)) if anisotropy is not None else 1.0

                    # Transformer prediction (if available)
                    if trans_probs is not None:
                        if trans_probs.shape[0] > i:
                            trans_probs_i = trans_probs[i]
                        else:
                            trans_probs_i = trans_probs[-1]  # fallback to last if mismatch
                        
                        trans_class_idx = np.argmax(trans_probs_i)
                        trans_confidence = trans_probs_i[trans_class_idx]
                        trans_label = reverse_map.get(trans_class_idx, f"unknown_{trans_class_idx}")
                        
                        # Calibrated probabilities (blend of MLP and Transformer)
                        if use_transformer:
                            # Boost MLP's prediction in transformer probabilities
                            calibrated = trans_probs_i.copy()
                            try:
                                calibrated[mlp_class_idx] = max(calibrated[mlp_class_idx], anisotropic_rate)
                                calibrated /= calibrated.sum()
                            except Exception as e:
                                calibrated = self.pipeline._calibrate_probs(mlp_probs, mlp_pred_indices, attn_weights, input_ids)
            
                            final_probs = calibrated
                            final_class_idx = mlp_class_idx  # Trust MLP's class decision
                            try:
                                final_confidence = final_probs[final_class_idx]
                            except IndexError:
                                final_confidence = np.max(final_probs) if isinstance(final_probs, np.ndarray) else final_probs

                            if isinstance(final_confidence, np.ndarray):
                                final_confidence = np.max(final_confidence)
                                            
                        else:
                            final_probs = mlp_probs[i]
                            final_class_idx = mlp_class_idx
                            final_confidence = mlp_confidence
                    else:
                        final_probs = mlp_probs[i]
                        final_class_idx = mlp_class_idx
                        final_confidence = mlp_confidence
                        trans_label = reverse_map.get(mlp_class_idx, f"unknown_{mlp_class_idx}")
                        trans_confidence = mlp_confidence
                    
                    final_label = reverse_map.get(final_class_idx, f"unknown_{final_class_idx}")
                    
                    result = {
                        'title': display_title,
                        'expected': expected_label,
                        'predicted': final_label,
                        'confidence': final_confidence,
                        'index': final_class_idx,
                        'mlp_prediction': mlp_label,
                        'mlp_confidence': mlp_confidence,
                    }
                    
                    if trans_label is not None:
                        result['transformer_prediction'] = trans_label
                        result['transformer_confidence'] = trans_confidence

                    agreement = trans_label  == mlp_label
                    
                    
                    # Include top-k predictions if requested
                    if show_proba:
                        top_indices = np.argsort(final_probs)[-top_k:][::-1]
                        top_predictions = []
                        for idx in top_indices:
                            if idx in reverse_map:
                                top_predictions.append({
                                    'label': reverse_map[idx],
                                    'confidence': final_probs[idx]
                                })
                            else:
                                top_predictions.append({
                                    'label': f"unknown_{idx}",
                                    'confidence': final_probs[idx]
                                })
                        result['top_predictions'] = top_predictions
                        
                        mlp_top_indices = np.argsort(mlp_probs[i])[-top_k:][::-1]
                        mlp_top = []
                        for idx in mlp_top_indices:
                            if idx in reverse_map:
                                mlp_top.append({
                                    'label': reverse_map[idx],
                                    'confidence': mlp_probs[i][idx]
                                })
                        result['mlp_top_predictions'] = mlp_top
                        
                        if trans_probs is not None:
                            trans_top_indices = np.argsort(trans_probs[i])[-top_k:][::-1]
                            trans_top = []
                            for idx in trans_top_indices:
                                if idx in reverse_map:
                                    trans_top.append({
                                        'label': reverse_map[idx],
                                        'confidence': trans_probs[i][idx]
                                    })
                            result['transformer_top_predictions'] = trans_top
                    
                    results.append(result)
            else:
                n_samples = mlp_probs.shape[0]

                lstm_pred_indices = np.argmax(lstm_probs, axis=1) if lstm_probs is not None else None
                if mlp_probs is None:
                    logits = self.pipeline.mlp.forward(X) if X is not None else X_tfidf
                    mlp_probs = self.pipeline._softmax(logits)

                target_probs = self.calibration_penalized_check(mlp_probs, mlp_pred_indices)
                target_pred_indices = np.argmax(target_probs, axis=1)

                for i in range(n_samples):
                    outcome = self._compute_sample_prediction(
                        i, mlp_probs, target_probs, target_pred_indices,
                        trans_probs=trans_probs, lstm_probs=lstm_probs,
                        lstm_pred_indices=lstm_pred_indices,
                        attn_weights=attn_weights, input_ids=input_ids,
                        num_classes=num_classes, reverse_map=reverse_map
                    )

                    result = {
                        "title": f"Unknown",
                        "expected": f"Unknown",
                        **outcome
                    }
                    results.append(result)

            # Display results

            if titles is not None and len(titles) > 0:
                print("\n" + "="*70)
                print("🎯 HYBRID PREDICTION RESULTS (MLP + Transformer)")
                print("="*70)
                
                correct_count = 0
                for result in results:
                    print(f"\n📌 '{result['title']}'")
                    
                    if result.get('expected'):
                        status = "✓" if result['predicted'] == result['expected'] else "✗"
                        print(f"   Expected: {result['expected']} {status}")
                    
                    print(f"   🎯 FINAL PREDICTION: {result['predicted']} ({result['confidence']:.1%})")
                    print(f"   ⚡ MLP: {result['mlp_prediction']} ({result['mlp_confidence']:.1%})")
                    
                    if result.get('transformer_prediction'):
                        arrow = "⬆️" if result['transformer_confidence'] > result['mlp_confidence'] else "⬇️"
                        print(f"   🌀 Transformer: {result['transformer_prediction']} ({result['transformer_confidence']:.1%}) {arrow}")
                    
                    if show_proba and 'top_predictions' in result:
                        print("\n   🔍 Top possibilities (calibrated):")
                        for j, pred in enumerate(result['top_predictions'][:top_k], 1):
                            bar = '█' * int(pred['confidence'] * 20)
                            print(f"      {j}. {pred['label']:20s} {bar} {pred['confidence']:.1%}")
                    
                    if result.get('expected') and result['predicted'] == result['expected']:
                        correct_count += 1
                
                if results and results[0].get('expected'):
                    accuracy = correct_count / len(results)
                    print(f"\n📊 Accuracy: {correct_count}/{len(results)} = {accuracy:.1%}")

            verbose = False
            if float(results[0]['confidence']) < self.pipeline.confidence_threshold:
                verbose = True

            payload = {
                'X_samples': X,
                'input_ids': input_ids
            }    

            if titles is not None and len(titles) > 0:
                self.display_hybrid_results(payload, final_class_idx, results, top_k, verbose=verbose)

            # Use results directly - they already contain calibrated predictions
            chosen_label = results[0]['predicted'] if results else None
            confidence = results[0]['confidence'] if results else None

            if isinstance(chosen_label, int) or isinstance(chosen_label, np.integer):
                chosen_label = str(chosen_label)
                
            # Only recalibrate if models disagreed AND we have valid results
            if results and not results[0].get('models_agree', True):
                print("\n[⚠️] Disagreement detected between MLP and Transformer predictions. Using calibrated probabilities for final decision.")
                calibrated_probs = self.pipeline.hybrid_prediction(rules, input_ids, dataset, X=X, y=y, use_embedded=use_embedded)
               
                if calibrated_probs is not None and len(calibrated_probs) > 0:
                    final_idx = int(np.argmax(calibrated_probs[:num_classes]))
                    if final_idx > len(reverse_map):
                        final_idx = int(np.argmax(calibrated_probs[:len(reverse_map)-1]))
                        print(f"[⚠️] Clamping {final_idx} → {final_idx}") 

                    final_idx = int(min(final_idx, num_classes - 1))  # Ensure index is within valid range
                    chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")
                    try:
                        confidence = float(calibrated_probs[0][final_idx])   
                    except:
                        confidence = float(calibrated_probs[0][len(reverse_map)-1]) if isinstance(calibrated_probs[0], (float, int)) else 0.0             
                            
            if isinstance(chosen_label, str) and chosen_label.startswith("unknown") or float(confidence) < self.pipeline.confidence_threshold:
                print(f"\n[⚠️] Final prediction is {chosen_label} with uncertain confidence: {confidence:.1%}. Consider collecting more data or adjusting the model.")
            else:
                print(f"\n[🎯] Final chosen label for input: {chosen_label} || Confidence: {confidence:.1%}")  

            if results and results[0]['confidence'] > self.pipeline.confidence_threshold:
                results[0]['predicted'] = chosen_label
                results[0]['confidence'] = confidence

        except Exception as e:
            print(f"[=] Error during prediction: {e}")
            traceback.print_exc()
            results = []

        return results



    def simple_pass_prediction(self, pipeline, titles=None, label_map=None, rules=None, X=None, X_raw=None, y=None, show_proba=True, top_k=3, batch_size=2):
        """Lightweight single-pass prediction used when the full advanced flow is not needed."""
        self.pipeline.titles = titles
        self.pipeline.labels = label_map   

        num_classes = self.pipeline._get_num_classes(label_map=label_map)
        try:

            if titles is not None and rules is not None:
                print(f"[🔍] Preparing data for {len(titles)} titles with {len(rules)} length of rules.")
                if X is None and y is None or X is None or y is None:
                    print('[🔄] Creating automatic X samples because X Samples is not provided manually.')
                    datasets, X_gen = self.pipeline.data_preparation(titles, label_map)  
                    _, y, _, _ = self.pipeline.mlp_training_features(rules, datasets, label_map=label_map)                  
                else:
                    datasets, _ = self.pipeline.data_preparation(titles, label_map)

            if X is not None or y is not None and len(X) > 0 and len(y) > 0:
                X_train, X_test, y_train, y_test, X_val, y_val = self._prepare_train_val_split(
                    X, y, min_val_per_class=5, min_frac=0.1, max_frac=0.3
                )
                
                onehot_validation = self.pipeline._validate_onehot(y_train)
                infer_y = self.infer_target_type(self.pipeline.vocab_size, y)
                if onehot_validation and infer_y == 'classification':
                    try:
                        if num_classes != len(label_map):
                            num_classes = len(label_map)
                        if num_classes > np.max(y_train):
                            y_train = np.eye(num_classes)[np.asarray(y_train).astype(int)]
                            y_val = np.eye(num_classes)[np.asarray(y_val).astype(int)]
                        else:
                            print('[⚠️] Warning: Y onehot encoding fails, Returning y samples as is, This may cause Exploding Gradient in MLP Training!')
                            y_train = y_train
                            y_val = y_val
                    except Exception as e:
                        print('[⚠️] y-onehot encoding fails, returning y sample as is.')

                X_mean = X_train.mean(axis=0)
                X_std = X_train.std(axis=0) + 1e-8

                X_train = (X_train - X_mean) / X_std
                X = (X_val - X_mean) / X_std
                y = y_val.copy()

            reverse_map = {v: k for k, v in label_map.items()}

            if datasets is not None:
                input_datasets = self.pipeline.transformer_input_encoding(datasets)
            else:
                input_datasets = self.pipeline._features_to_sequence(X)

            pred_probs = self.pipeline.predict_proba(input_datasets, X_raw, type='Hybrid', embedded=True)[0]
            try:
                pred_result = self.pipeline.hybrid_prediction(rules, input_datasets, datasets, X=X, y=y, use_embedded=True)
            except:
                pred_result = self.pipeline.hybrid_prediction(rules, input_datasets, datasets, X=X, y=y, use_embedded=False)

            if X is None and X_gen is not None:
                X = X_gen
            
            if self.pipeline.cache and 'label_bins' in self.pipeline.cache:
                print('[=] label_bins cache found!')
                label_bins = self.pipeline.cache['label_bins']
                lstm_probs, _ = self.pipeline.ensemble._get_lstm_probs(input_datasets, X, label_bins=label_bins)      
            else:
                lstm_probs = None

            print("\n[🔍] Prediction result structure:")
            print(f"[=] Type: {type(pred_result)}")
            print(f"[=] Length: {len(pred_result) if isinstance(pred_result, tuple) else 1}")

            if isinstance(pred_result, tuple):
                if len(pred_result) == 3:
                    pred_indices = pred_result[0]
                    hybrid_probs = pred_result[1]  # Use different variable name
                    attn_weights = pred_result[2]
                    print("✅ Extracted: indices, probs, attention")
                elif len(pred_result) == 2:
                    pred_indices = pred_result[0]
                    hybrid_probs = pred_result[1]  # Use different variable name
                    print("✅ Extracted: indices, probs")
                else:
                    print(f"⚠️ Unknown tuple format with {len(pred_result)} elements")
                    pred_indices = pred_result[0]
                    hybrid_probs = pred_result[1] if len(pred_result) > 1 else None
            else:
                pred_indices = pred_result
                hybrid_probs = None
                print("✅ Single value return")
        
            # Use hybrid_probs if available, otherwise use pred_probs
            final_probs = hybrid_probs if hybrid_probs is not None else pred_probs
        
            if isinstance(pred_indices, (list, tuple)) and len(pred_indices) > 0:
                if isinstance(pred_indices[0], (np.ndarray, list)):

                    pred_indices = np.array([p[0] if isinstance(p, (np.ndarray, list)) else p 
                                        for p in pred_indices])
                else:
                    pred_indices = np.array(pred_indices)
            elif isinstance(pred_indices, np.ndarray):
                if pred_indices.ndim > 1:
                    pred_indices = pred_indices.flatten()
            else:
                pred_indices = np.array([pred_indices])
        
            print(f"\n[📊] Processed predictions:")
            print(f"[=] pred_indices shape: {pred_indices.shape}")
            print(f"[=] pred_indices: {pred_indices}")
        
            if final_probs is not None:
                print(f"[=] final_probs shape: {final_probs.shape if hasattr(final_probs, 'shape') else 'unknown'}")
        
            if final_probs is not None and isinstance(final_probs, np.ndarray) and final_probs.ndim == 1:
                final_probs = final_probs.reshape(1, -1)

            if titles is not None and len(titles) > 0:
                n_samples = len(titles)
            else:
                n_samples = pred_probs.shape[0] if hasattr(pred_probs, 'shape') else 0

            if len(pred_indices) < n_samples:
                print(f"[⚠️] Padding predictions from {len(pred_indices)} to {n_samples}")
                last_idx = pred_indices[-1] if len(pred_indices) > 0 else 0
                pred_indices = np.pad(pred_indices, (0, n_samples - len(pred_indices)), 
                                mode='constant', constant_values=last_idx)

            results = []
            best_idx = -1
            best_confidence = -1
            
            # Determine rows and cols from final_probs
            if final_probs is not None and hasattr(final_probs, 'shape'):
                rows = final_probs.shape[0]
                cols = final_probs.shape[1] if len(final_probs.shape) > 1 else 1
            elif final_probs is not None:
                rows = len(final_probs)
                cols = len(final_probs[0]) if rows > 0 and hasattr(final_probs[0], '__len__') else 1
            else:
                rows, cols = 0, 0
            
            for i in range(n_samples):
                class_idx = int(pred_indices[i]) if i < len(pred_indices) else 0
                    
            if final_probs is not None and i < rows and class_idx < cols:
                if hasattr(final_probs, 'shape'):
                    confidence = final_probs[i, class_idx]
                else:
                    if isinstance(final_probs[i], (list, np.ndarray)):
                        if i < len(final_probs):
                            confidence = final_probs[i][class_idx]
                        else:
                            confidence = np.mean(final_probs[class_idx])
                    else:
                        if i < len(final_probs):
                            confidence = float(final_probs[i])  # Single value
                        else:
                            confidence = float(np.mean(final_probs))

                        
                if confidence > best_confidence:
                    best_idx = i
                    best_confidence = confidence


            if titles is not None and len(titles) > 0:   
                for i, title in enumerate(titles):
                    if i < len(pred_indices):
                        class_idx = int(pred_indices[i])
                    else:
                        class_idx = 0
                        
                    # Get confidence from final_probs
                    if final_probs is not None and i < rows and class_idx < cols:
                        if hasattr(final_probs, 'shape'):
                            confidence = final_probs[i, class_idx]
                        else:  # list
                            if isinstance(final_probs[i], (list, np.ndarray)):
                                confidence = final_probs[i][class_idx]
                            else:
                                confidence = float(final_probs[i])  # Single value
                    else:
                        # Fallback: use max probability instead of min
                        if final_probs is not None and i < len(final_probs):
                            if isinstance(final_probs[i], (list, np.ndarray)):
                                confidence = max(final_probs[i])
                            else:
                                confidence = float(final_probs[i])
                        else:
                            confidence = 0.0
                
                    label = reverse_map.get(class_idx, f"unknown_{class_idx}")

                    result = {
                    'title': title,
                    'predicted': label,
                    'confidence': confidence,
                    'index': class_idx,
                    'is_best': (i == best_idx)
                    }
                    
                    if show_proba and i < rows and cols > 1:
                        if hasattr(final_probs, 'shape'):
                            probs_row = final_probs[i]
                        else:
                            if isinstance(final_probs[i], (list, np.ndarray)):
                                probs_row = np.array(final_probs[i])
                            else:
                                probs_row = np.array([final_probs[i]])
                    
                        if len(probs_row) > 1:
                            top_indices = np.argsort(probs_row)[-top_k:][::-1]
                            top_predictions = []
                            for idx in top_indices:
                                if idx in reverse_map:
                                    top_predictions.append({
                                    'label': reverse_map[idx],
                                    'confidence': float(probs_row[idx])
                                    })
                            result['top_predictions'] = top_predictions
                
                    results.append(result)
            
                print("\n" + "="*70)
                print("[🎯] LABEL PREDICTIONS")
                print("="*70)
            
                for i, result in enumerate(results):
                    print(f"\n[📌] Label: {i+1}. '{result['title']}'")
                
                    best_marker = "[🏆] BEST" if result.get('is_best') else ""
                    print(f"   → {result['predicted']} ({result['confidence']}){best_marker}")
                
                    if show_proba and 'top_predictions' in result:
                        print(" [  Top possibilities:")
                        for j, pred in enumerate(result['top_predictions'][:top_k], 1):
                            bar = '█' * int(pred['confidence'] * 20)
                            print(f"      {j}. {pred['label']} {bar} {pred['confidence']} %")
                
                # Return the best result (not inside loop)
                best_idx = int(np.argmax(final_probs[:, pred_indices] if final_probs is not None and hasattr(final_probs, 'shape') else [r['confidence'] for r in results]))
                if best_idx >= 0:
                    best_result = results[best_idx]
                    if isinstance(best_result['predicted'], str) and best_result['predicted'].startswith("unknown") or best_result['confidence'] < self.pipeline.confidence_threshold:
                        print(f"\n[⚠️] Final prediction is {best_result['predicted']} with uncertain confidence. Consider collecting more data or adjusting the model.")
                    else:
                        print(f"\n✨ Most confident: '{best_result['title']}' → {best_result['predicted']} ({best_result['confidence']:.1%})")
                    return best_result['predicted'], best_result['confidence'], best_result['confidence']
                elif results:
                    # Fallback: return first result if no best found
                    predicted = results[0]['predicted']
                    predicted_confidence = results[0]['confidence']
                    if isinstance(predicted, str) and predicted.startswith("unknown") and predicted_confidence < self.pipeline.confidence_threshold:
                        print(f"\n[⚠️] Final prediction is {predicted} with uncertain confidence: {predicted_confidence:.1%}. Consider more consistent data for the model to learn from.")
                    else:
                        print(f"\n[🎯] Final chosen label for input: {predicted} || Confidence: {predicted_confidence:.1%}")  
                    
                    return predicted, predicted_confidence

            else:
                print("[=] No titles provided for prediction, skipping accuracy validation.")
                
                lstm_pred_indices = np.argmax(lstm_probs, axis=1) if lstm_probs is not None else None
                best_pred_indices = np.argmax(final_probs, axis=1)

                for i in range(n_samples):
                    outcome = self._compute_sample_prediction(
                        i, pred_probs, final_probs, best_pred_indices,
                        trans_probs=trans_probs, lstm_probs=lstm_probs,
                        lstm_pred_indices=lstm_pred_indices,
                        attn_weights=attn_weights, input_ids=input_datasets,
                        num_classes=num_classes, reverse_map=reverse_map
                    )

                predicted = outcome['predicted']
                predicted_confidence = outcome['confidence']

        except Exception as e:
            print(f"[=] Error during robust prediction: {e}")
            predicted = None
            predicted_confidence = None
        return predicted, predicted_confidence
        
    def calculate_entropy(self, probs):
        """Shannon entropy of a probability vector (uncertainty measure)."""
        return -np.sum(probs * np.log(probs + 1e-10), axis=-1)


    def run_mc_dropout(self, model, X, y, input_ids=None, label_bin=None, AME=None, use_embedded=True, n_samples=50):
        """Monte-Carlo dropout: `n_samples` stochastic passes of the 'mlp' or transformer model to estimate predictive uncertainty."""
        predictions = []
        if model == 'mlp':
            for _ in range(n_samples):
                pred = self.pipeline.model3.forward(X, y, mode='regression')
                predictions.append(pred)
        elif model == 'trans':
            for _ in range(n_samples):
                pred, _ = self.pipeline.model2.forward_continuous(
                    X, training=False, attn_dropout=0.1, ffn_dropout=0.1
                )
                predictions.append(pred)

        else:
            print('[>] Invalid model type!')
        
        if len(predictions) > 0:
            return np.stack(predictions, axis=0)
        else:
            return None


    def _compute_sample_prediction_regression(
        self, i, mlp_preds, config,
        trans_preds=None, eps=1e-8,
    ):

        def _mc_stats(mc_preds, sample_idx):
            if mc_preds is None:
                return None

            if mc_preds.ndim == 3:
                # shape: (n_mc_samples, n_data, 1)
                sample_preds = mc_preds[:, sample_idx, :].flatten()   # (n_mc_samples,)
            elif mc_preds.ndim == 2:
                # shape: (n_mc_samples, n_data)
                sample_preds = mc_preds[:, sample_idx]                 # (n_mc_samples,)
            else:
                raise ValueError(f"[>] mc_preds shape unknown: {mc_preds.shape}")

            mean_norm = float(sample_preds.mean())
            std_norm  = float(sample_preds.std())

            mean_original = config.unnormalize_target(np.array([mean_norm]))[0]

            relative_spread = std_norm / (abs(mean_norm) + eps)
            confidence = float(np.clip(1.0 - relative_spread, 0.0, 1.0))

            return {
                "prediction": float(mean_original),
                "confidence": confidence,
                "std_normalized": std_norm,
                "raw_samples": sample_preds.tolist(),
            }

        mlp_result   = _mc_stats(mlp_preds, i)

        if trans_preds is not None:
            trans_result = _mc_stats(trans_preds, i)
        else:
            trans_result = None

        is_valid = mlp_result is not None

        if not is_valid:
            return {
                "is_valid": False,
                "error": "preds None",
                "mlp": None,
                "trans": None,
            }

        return {
            "is_valid": True,
            "mlp": mlp_result,
            "trans": trans_result,
        }

    def _compute_sample_prediction(
        self, i, mlp_probs, target_probs, target_pred_indices,
        trans_probs=None, lstm_probs=None, lstm_pred_indices=None,
        attn_weights=None, input_ids=None, num_classes=None,
        need_ensemble_method=False, reverse_map=None, eps=1e-8,
        mode=None
    ):
        """
        Pure per-sample ensemble computation — no title/label dependency.
        Returns a dict with mlp/trans/lstm/final confidence and class index,
        usable both for titled prediction loops and raw batch prediction.
        """
        reverse_map = reverse_map or {}
        num_classes = num_classes or mlp_probs.shape[1]
        models_agree = False

        # ── MLP layer ─────────────────────────────────────────────
        mlp_class_idx  = int(np.argmax(mlp_probs[i]))
        is_valid_index = 0 <= mlp_class_idx < num_classes

        if not is_valid_index:
            try:
                mlp_class_idx = int(np.argmax(mlp_probs[:len(reverse_map)-1]))
                mlp_confidence = float(mlp_probs[:len(reverse_map)-1][mlp_class_idx]) 
                mlp_label      = reverse_map.get(mlp_class_idx, f"unknown_{mlp_class_idx}")
                return {
                    'predicted'  : mlp_label,
                    'confidence' : mlp_confidence,
                    'mlp_class'  : mlp_class_idx,
                    'predicted_idx'      : mlp_class_idx,
                    'models_agree': models_agree,
                    'final_probs' : mlp_probs[:len(reverse_map)-1],
                    'is_valid'   : True,
                }
            except:
                return {
                    'predicted'  : None,
                    'confidence' : 0.0,
                    'mlp_class'  : 0,
                    'predicted_idx'      : 0,
                    'models_agree': False,
                    'final_probs' : mlp_probs,
                    'is_valid'   : False,
                    'error'      : f'class_index_out_of_range(idx={mlp_class_idx}, num_classes={num_classes})'
                }               

        mlp_confidence = float(mlp_probs[i][mlp_class_idx]) 
        mlp_label      = reverse_map.get(mlp_class_idx, f"unknown_{mlp_class_idx}")

        # ── LSTM ────────────────────── ─────────────────────
        # lstm_pred_indices passed in precomputed, not recomputed per-sample
        if lstm_probs is not None and lstm_pred_indices is not None:
            lstm_class_idx  = int(lstm_pred_indices[i]) 
            lstm_confidence = float(lstm_probs[i][lstm_class_idx])
        else:
            # explicit default so lstm_class_idx isss always defined, even if LSTM is not used hre
            lstm_class_idx  = mlp_class_idx
            lstm_confidence = None

        # ── target/base probs ────────────────────────────────
        target_class_idx = int(target_pred_indices[i])
        target_confidence = float(target_probs[i][target_class_idx])

        # ── Transformer ───────────────────────────────────────
        if trans_probs is not None and attn_weights is not None:
            trans_probs_i = np.asarray(trans_probs[i])

            if trans_probs_i.ndim == 0 or trans_probs_i.size == 1:
                trans_class_idx  = target_class_idx
                trans_confidence = target_confidence
            else:
                trans_class_idx  = int(np.argmax(trans_probs_i))
                trans_confidence = float(trans_probs_i[trans_class_idx])

            trans_label = reverse_map.get(trans_class_idx, f"unknown_{trans_class_idx}")

            if need_ensemble_method:
                calibration = self.pipeline._calibrate_probs(
                    target_probs, target_pred_indices, attn_weights, input_ids
                )

                mlp_weight   = mlp_confidence   / (target_confidence + trans_confidence + eps)
                trans_weight = trans_confidence / (target_confidence + trans_confidence + eps)
                lstm_weight  = None
                if lstm_confidence is not None:
                    lstm_weight = lstm_confidence / (target_confidence + lstm_confidence + eps)

                cal_len = len(calibration[0]) if calibration.ndim > 1 else len(calibration)
                calibration_weighting = (
                    calibration[target_class_idx]
                    if target_class_idx < cal_len else 0.0
                )

                if lstm_confidence is not None and lstm_weight is not None:
                    final_probs = (mlp_weight   * target_probs[i][:cal_len] +
                                trans_weight * calibration[i][:cal_len] +
                                lstm_weight  * calibration[i][:cal_len])
                else:
                    final_probs = (mlp_weight   * target_probs[i][:cal_len] +
                                trans_weight * calibration[i][:cal_len])

                final_probs = self.pipeline._calibrate_probs(final_probs, mlp_class_idx, attn_weights, input_ids)
                final_class_idx = target_class_idx

                try:
                    final_confidence = final_probs[final_class_idx]
                except IndexError:
                    final_confidence = (np.max(final_probs)
                                        if isinstance(final_probs, np.ndarray)
                                        else np.mean(final_probs))

                if isinstance(final_confidence, np.ndarray):
                    final_confidence = float(np.max(final_confidence))

                models_agree = mlp_class_idx == trans_class_idx

            else:
                if lstm_confidence is None:
                    lstm_confidence = mlp_confidence
                models_agree = False

                if (mlp_confidence > trans_confidence and
                    mlp_confidence > lstm_confidence and
                    mlp_confidence <= 0.95):
                    final_probs      = mlp_probs[i]
                    final_class_idx  = mlp_class_idx
                    final_confidence = mlp_confidence

                elif trans_confidence > lstm_confidence:
                    final_probs      = trans_probs[i]
                    final_class_idx  = trans_class_idx
                    final_confidence = trans_confidence

                else:
                    final_probs      = lstm_probs[i] if lstm_probs is not None else mlp_probs[i]
                    final_class_idx  = lstm_class_idx if lstm_probs is not None else mlp_class_idx
                    final_confidence = lstm_confidence if lstm_probs is not None else mlp_confidence

                    if final_confidence > 0.95:
                        if lstm_probs is not None and mlp_probs.shape == lstm_probs.shape:
                            final_probs      = mlp_probs[i] * lstm_probs[i]
                            final_class_idx  = mlp_class_idx
                            final_confidence = mlp_confidence * lstm_confidence
                        else:
                            if lstm_probs is not None:
                                final_probs = (target_probs[i]
                                            if len(target_probs[i]) == num_classes
                                            else lstm_probs[i])
                            else:
                                final_probs = trans_probs[i]
                            final_class_idx  = target_class_idx
                            final_confidence = target_confidence * (lstm_confidence or 1.0)

                models_agree = mlp_class_idx == trans_class_idx

        else:
            # no transformer available — MLP/LSTM only
            trans_class_idx  = None
            trans_confidence = None
            trans_label      = None
            final_probs      = mlp_probs[i]
            final_class_idx  = mlp_class_idx
            final_confidence = mlp_confidence
            agreement        = False
            models_agree     = False

        return {
            'is_valid'          : True,
            'predicted'         : reverse_map.get(final_class_idx, f"unknown_{final_class_idx}"),
            'predicted_idx'             : final_class_idx,
            'confidence'        : float(final_confidence),
            'mlp_class'         : mlp_class_idx,
            'mlp_prediction'    : mlp_label,
            'mlp_confidence'    : mlp_confidence,
            'trans_class'       : trans_class_idx,
            'trans_prediction'  : trans_label,
            'trans_confidence'  : trans_confidence,
            'lstm_class'        : lstm_class_idx if lstm_probs is not None else None,
            'lstm_confidence'   : lstm_confidence,
            'models_agree'      : models_agree,
            'final_probs'       : final_probs,
        }

    def dynamic_test_size(self, y, min_val_per_class=5, min_frac=0.1, max_frac=0.3):
        """
        Pick test_size so the smallest class still gets a usable validation sample,
        without departing too far from typical train/val ratios.
        """
        classes, counts = np.unique(y, return_counts=True)
        n_total = len(y)
        smallest_class_count = counts.min()

        # fraction needed to guarantee min_val_per_class from the rarest class
        needed_frac = min_val_per_class / smallest_class_count

        # clamp to a sane range so it doesn't swing to extremes
        test_size = np.clip(needed_frac, min_frac, max_frac)
        return float(test_size)

    def _prepare_train_val_split(self, X, y, min_val_per_class=5,
                                min_frac=0.1, max_frac=0.3, random_state=42):
        """
        Safely aligns X/y sample counts and splits into train/val,
        never guessing via transpose, always checking stratify viability.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()   # flatten (N,1) labels to (N,) for stratify compatibility

        if X.shape[0] != y.shape[0]:
            print(f'[⚠️] X/y sample count mismatch: X={X.shape[0]}, '
                f'y={y.shape[0]} — aligning to shorter length. '
                f'This likely indicates an upstream bug where X and y '
                f'became desynchronized; investigate the caller.')
            min_samples = min(X.shape[0], y.shape[0])
            X = X[:min_samples]
            y = y[:min_samples]

        if X.shape[0] < 4:
            print(f'[⚠️] Only {X.shape[0]} samples — too few for a train/val '
                f'split, returning all data as training set, no validation')
            return X, np.empty((0,) + X.shape[1:]), y, np.empty((0,) + y.shape[1:])

        dynamic_test_bed = dynamic_test_size(
            y, min_val_per_class=min_val_per_class,
            min_frac=min_frac, max_frac=max_frac
        )

        unique_classes, class_counts = np.unique(y, return_counts=True)
        can_stratify = np.all(class_counts >= 2) and len(unique_classes) > 1

        if not can_stratify:
            sparse_classes = unique_classes[class_counts < 2]
            print(f'[⚠️] Cannot stratify: class(es) with length: {len(sparse_classes.tolist())} '
                f'have fewer than 2 samples — falling back to non-stratified split')

        try:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=dynamic_test_bed,
                stratify=y if can_stratify else None,
                random_state=random_state
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=dynamic_test_bed + min_frac, random_state=42
            )
        except ValueError as e:
            raise Warning(f'[⚠️] train_test_split failed even without stratify: {e} — '
                    'The model cant learn with very Few Y Signals inside the sample,'
                f'Cannot proceed with Training and prediction and make sure to have Y samples with proper shapes!')

        return X_train, X_test, y_train, y_test, X_val, y_val



    def _compute_need_ensemble_method(self, anisotropy, AME, error_counts,
                                    anisotropy_threshold=0.3,
                                    ame_threshold=0.3,
                                    error_threshold=0.3,
                                    min_signals_required=2):
        """
        Ensemble fallback is a safety net — it should trigger if ANY
        single strong risk signal fires, not only when all signals
        happen to align simultaneously. Uses max() over error_counts
        so a single persistently-wrong class triggers assistance even
        when most other classes are healthy.
        """
        signals = []
        score = 0

        if anisotropy is not None and anisotropy > anisotropy_threshold:
            signals.append(f'anisotropy={anisotropy:.3f}')
            score += 1

        if AME is not None and AME > ame_threshold:
            signals.append(f'AME={AME:.3f}')
            score += 1

        if error_counts is not None and len(error_counts) > 0:
            # max: catches localized single-class failure
            # that a flat average would dilute away
            worst_class_idx = int(np.argmax(error_counts))
            max_error = error_counts[worst_class_idx]
            if max_error > error_threshold:
                signals.append(f'errorate(class={worst_class_idx}, '
                            f'value={max_error:.2f})')
                score += 1

        need_ensemble = len(signals) >= min_signals_required and score >= min_signals_required

        if need_ensemble:
            print(f'[=] Ensemble method triggered by: {", ".join(signals)} || Ensemble requirement Score: {score}')

        return need_ensemble


    def _check_for_transformer_sequences(self, sequence_inputs, min_seq_len=3, 
                                min_samples=10,  min_AME=0.5, min_anisotropy=0.5):
        score = 0
        sequence_inputs = np.asarray(sequence_inputs)
        n_samples = sequence_inputs.shape[0] if sequence_inputs.ndim > 0 else 0
        T = sequence_inputs.shape[1] if sequence_inputs.ndim > 1 else 1

        # ── Check 1: sample count ────────────────────────────────
        if n_samples < min_samples:
            score += 1
            print(f'[=] only {n_samples} samples (need >= {min_samples}) — '
                        f'too little data to justify Transformer parameter count')

        # ── Check 2: sequence length ─────────────────────────────
        if T < min_seq_len:
            score += 1
            print(f'[=] sequence length T={T} (need >= {min_seq_len}) — '
                        f'attention has too few positions to model '
                        f'meaningful relationships')

        # ── Check 3: anisotropy   ──
        # ── zero new computation cost                              ──
        anisotropy = self.pipeline.anisotropy_measurement(sequence_inputs)
        if anisotropy < min_anisotropy:
            score += 1
            print(f'[=] anisotropy={anisotropy:.4f} (need >= {min_anisotropy}) — '
                        f'data lacks directional/sequential structure; '
                        f'MLP alone is the appropriate model for this '
                        f'geometry.')  


        AME = self.pipeline.AME_Encoder(sequence_inputs)  
        if AME > min_AME:
            score += 1 
            print(f'[=] Result of Abstract Modelling Error is High, Complexity in samples is High.')

        try:
            cleaned_counts = [x for x in self.pipeline.model3.error_counts if x is not None]
            cleaned_counts = np.asarray(cleaned_counts)
            error_counts = np.mean(cleaned_counts)
            if error_counts > min_AME:
                score += 1
                print('[=] MLP Experienced lots of Errors, Considering to use Transformer Prediction.')
        except Exception as e:
            print(f'[>] Cant calculate total of errors due to: {e}')

        propriate_single_transformer = score == 5
        print(f'[+] Transformer Prediction allowed: {propriate_single_transformer}')
        return propriate_single_transformer

    def _get_voted_results(self, probs, reverse_map, n_samples):
        pred_cache = []
        _cache = {}
        cache = []
        for i in range(n_samples):
            try:
                each_idx = probs[i].argmax()
            except:
                each_idx = np.argmax(probs[:len(reverse_map)-1])
            try:
                confidence = float(probs[each_idx])   
            except:
                confidence = float(probs[i][len(reverse_map)-1]) if isinstance(probs[i], (float, int)) else 0.0
            
            pred_cache.append(each_idx)
            _cache[i] = (each_idx, confidence)
            
        voted_chosen_label = max(set(pred_cache), key=pred_cache.count)
        conf_cache = []
        for i in range(len(_cache)):
            if _cache[i][0] == voted_chosen_label:
                confidence = _cache[i][1] 
                conf_cache.append(confidence)

        float_counts = Counter(conf_cache)
        voted_confidence, _ = float_counts.most_common(1)[0]  
        if voted_confidence < 0.1:
            voted_confidence = self.pipeline.confidence_threshold + voted_confidence

        return voted_chosen_label, voted_confidence


    def infer_target_type(self, vocab_size, y, max_classes_ratio=0.05, max_unique_absolute=50, words=False):
        """Heuristically decide classification vs regression from y and vocabulary size."""
        y = np.asarray(y)
        y_flat = y.flatten()
        score = 0

        if vocab_size is not None or vocab_size < 2:
            score -= 1
        elif words:
            score -= 1
        elif self.pipeline.intents is not None and len(self.pipeline.intents) > 1:
            score -= 1
        if not np.all(np.equal(np.mod(y_flat, 1), 0)):
            score += 1

        unique_vals = np.unique(y_flat)
        n_unique = len(unique_vals)
        n_samples = len(y_flat)

        unique_ratio = n_unique / n_samples
        if n_unique > max_unique_absolute or unique_ratio > max_classes_ratio:
            score += 1

        if score < 1:
            return "classification"
        else:
            return "regression"
    
    def to_onehot(self, y, n_classes, name='y'):
        """
        returns (n, n_classes) float one-hot, or raises with a clear reason.
        - already one-hot with the right width -> returned unchanged
        - 1-D / (n,1) integer ids              -> encoded
        - labels outside 0..n_classes-1        -> ValueError (never silently skipped)
        """
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == n_classes:
            return y.astype(np.float64)
        if y.ndim == 2 and y.shape[1] != 1:
            raise ValueError(f'{name} has {y.shape[1]} columns but there are {n_classes} classes')
        idx = y.reshape(-1).astype(int)
        if idx.size and (idx.min() < 0 or idx.max() >= n_classes):
            raise ValueError(f'{name} labels span {idx.min()}..{idx.max()} '
                            f'but there are only {n_classes} classes')
        out = np.zeros((len(idx), n_classes))
        out[np.arange(len(idx)), idx] = 1.0
        return out


        
    def advanced_prediction_method(self, titles=None, label_map=None, rules=None,
                                X=None, y=None,
                                show_proba=False, top_k=3, 
                                use_transformer=True, return_attention=False, 
                                batch_size=2):
        """Full hybrid prediction flow.

        Encodes the inputs, runs the MLP, Transformer (optional), LSTM and peers, combines them with the
        weighted ensemble, applies calibration, uncertainty handling and feedback hooks, and returns the
        results (optionally with attention weights and top-k probabilities).
        """
        try:
            # ____ init temporary layer ____
            eps = 1e-5
            trans_probs = None
            attn_weights = None
            sequence_ids = None

            X_train = None
            X_val = None
            y_val = None
            y_train = None

            input_ids = None
            anisotropy = None
            final_class_idx = None

            AME = None
            use_embedded = False
            dataset = None
            target_probs = None
            target_pred_indices = None

            X_gen = None
            sec_chosen_label = None
            infer_y = 'classification'

            sec_confidence = 0.0
            final_confidence = 0.0
            min_signal = 1e-3

            correct = 0
            sec_correct= 0
            performance_score = self.pipeline.performance_result

            lstm_pred, lstm_conf, overall_conf, uncertainty = None, 0.0, 0.0, 0.0
            trans_pred, trans_conf = None, 0.0
            mlp_pred, mlp_conf = None, 0.0

            if label_map is None:
                raise ValueError("[!] label_map must be provided for all prediction methods!")
                
            if self.pipeline.use_transformer and self.pipeline.model2 is None:
                raise AttributeError('[>] Transformer Model is Missing! consider using integrated_training() function to initialize Model!')   
            if self.pipeline.model3 is None:
                raise AttributeError("[> MLP Model is Missing! consider using integrated_training() function to initialize Model!")            

            print("\n[🚀] Starting Advanced Hybrid Prediction Method")
            
            reverse_map = {v: k for k, v in label_map.items()}
            num_classes = self.pipeline._get_num_classes(label_map=label_map)

            self.pipeline.titles = titles
            self.pipeline.labels = label_map


            if X is not None and y is not None:
                print(f"[🔍] Processing X and y samples for prediction and Training...")
                if isinstance(X, (str, np.str_)):
                    clean_str = str(X).replace('[', '').replace(']', '')
                    X = np.fromstring(clean_str, sep=' ')
                if isinstance(X, np.ndarray) and np.issubdtype(X.dtype, np.character):
                    # catches arrays filled with string text
                    clean_str = ' '.join(X.astype(str).flatten()).replace('[', '').replace(']', '')
                    X = np.fromiter(
                            (x for x in clean_str.split() if x != "..."), dtype=float
                        )   

                if isinstance(y, (str, np.str_)):
                    clean_str = str(y).replace('[', '').replace(']', '')
                    y = np.fromstring(clean_str, sep=' ')
                if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.character):
                    # catches arrays filled with string text
                    clean_str = ' '.join(y.astype(str).flatten()).replace('[', '').replace(']', '')
                    y = np.fromiter(
                            (x for x in clean_str.split() if x != "..."), dtype=float
                        )
                
                if len(X) < 5:
                    print(f"[⚠️] Warning: Only {len(X)} samples provided. Consider providing more samples for better prediction accuracy!")

                if 0 in X.shape:
                    print(f"[⚠️] Warning: X has zero samples in the total shapes, X shapes: {X.shape}. Creating an empty array with shape (1, n_features) for processing..")
                    X = np.empty((1, X.shape[1]))  # Created an empty array with shape (1, n_features)
                    X = X.reshape(1, -1)  # Reshape to (1, n_features) if empty but has features

                X_train, X_test, y_train, y_test, X_val, y_val = self._prepare_train_val_split(
                    X, y, min_val_per_class=5, min_frac=0.1, max_frac=0.3
                )
                
                onehot_validation = self.pipeline._validate_onehot(y_train)
                infer_y = self.infer_target_type(self.pipeline.vocab_size, y)

                if onehot_validation and infer_y == 'classification':
                    try:
                        y_train = self.to_onehot(y_train, num_classes, 'y_train')
                        y_val = self.to_onehot(y_val, num_classes, 'y_val')
                        print(f'[>] One-hot encoded: y_train {y_train.shape}, y_val {y_val.shape}')
                    except Exception as e:
                        print(f'[⚠️] y-onehot encoding fails: {e}, this might be a regression based type of y samples, passing y values as is without any onehot')

                X_mean = X_train.mean(axis=0)
                X_std = X_train.std(axis=0) + 1e-8

                X_train = (X_train - X_mean) / X_std
                X = (X_val - X_mean) / X_std
                y = y_val.copy()

         
            if titles is not None and rules is not None:
                print(f"[🔍] Preparing data for {len(titles)} titles with {len(rules)} length of rules.")
                if X is None and y is None or X is None or y is None:
                    print('[🔄] Creating automatic X samples because X is not provided manually.')
                    dataset, X_gen = self.pipeline.data_preparation(titles, label_map)  
                    _, y, _, _ = self.pipeline.mlp_training_features(rules, dataset, label_map=label_map)                  
                else:
                    dataset, _ = self.pipeline.data_preparation(titles, label_map)
                    
            if y_train is None:
                y_train = y
            if X_train is None:
                X = X_gen if X_gen is not None else X  

            if y_val is None:
                X_train, X_test, y_train, y_test, X_val, y_val = self._prepare_train_val_split(
                    X, y, min_val_per_class=5, min_frac=0.1, max_frac=0.3
                ) 

            
            if infer_y == 'classification':
                if dataset is not None:
                    input_ids, _ = self.pipeline.input_encoding(dataset)
                elif dataset is not None and X is not None:
                    input_ids, _ = self.pipeline.input_encoding(dataset)
                    row_sums = input_ids.sum(axis=1)
                    weak_rows = np.where(row_sums < min_signal)[0]
                    weak_ratio = len(weak_rows) / len(input_ids)
                    if weak_ratio > 100:
                        print(f'[=] Zero rows abundant in Input indices: {weak_ratio:.1%}, using input indices from X samples...')
                        input_ids = self.pipeline._features_to_sequence(X)
                else:
                    input_ids = self.pipeline._features_to_sequence(X)
            else:
                input_ids = X

            if len(input_ids) < 1:
                print('[⚠️] input_encoding produced no samples — skipping '
                    'transformer training/prediction for this call and transformer results will be Replaced by MLP or LSTM.')
                self.pipeline.use_transformer = False
    
            if self.pipeline.use_transformer and self.pipeline.model2 is not None:
                print("\n[🔄] Running Transformer forward method (Transformer)")
                if X is not None and y is not None:
                    sequence_ids = self.pipeline._features_to_sequence(X_train)
                else:
                    sequence_ids = self.pipeline.sequence_encoding(dataset, label_map=label_map)

                if X is not None:
                    anisotropy = self.pipeline.anisotropy_measurement(X_train)
                else:
                    anisotropy = self.pipeline.anisotropy_measurement(sequence_ids)

                # Get transformer predictions with attention
                print(f"[⚡] anisotropy rate detected on input: {anisotropy:.1%}.")                
                use_embedded = True
                
                AME = self.pipeline.model2.AME_Encoder(sequence_ids) 
                AME = AME if isinstance(AME, (int, float)) else self.pipeline.confidence_threshold

                if self.pipeline.model2 is not None:
                    class_task = self.pipeline.model2.task == 'classification'
                else:
                    class_task = infer_y

                if class_task and self.pipeline.model2.n_features is None:       
                    try:
                        print('[>] Sample is classification type.')
                        trans_probs, attn_weights = self.pipeline.model2.forward(sequence_ids, AME=AME, embedded=use_embedded)
                    except:
                        try:
                            print('[>] Sample turns out to be regression type.')
                            trans_probs, attn_weights = self.pipeline.model2.forward_continuous(X, training=False,
                                                        attn_dropout=self.pipeline.model2.attn_dropout_rate,
                                                        ffn_dropout=self.pipeline.model2.ffn_dropout_rate)
                        except Exception as e:
                            print(f'[>] Transformer forward prediction failed due to: {e}, This may have been caused by different categories of samples length., Skipping Transformer Prediction...')           

                else:
                    try:
                        print('[>] Sample is regression type.')
                        trans_probs, attn_weights = self.pipeline.model2.forward_continuous(X, training=False,
                                                        attn_dropout=self.pipeline.model2.attn_dropout_rate,
                                                        ffn_dropout=self.pipeline.model2.ffn_dropout_rate)
                    except:
                        try:
                            trans_probs, attn_weights = self.pipeline.model2.forward(sequence_ids, AME=AME, embedded=use_embedded)
                            print('[>] Sample turns out to be classification type.')
                        except Exception as e:
                            print(f'[>] Transformer forward prediction failed due to: {e}, This may have been caused by different categories of samples length., Skipping Transformer Prediction...')                            
                    
            else:
                anisotropy = self.pipeline.anisotropy_measurement(X) if X is not None else self.pipeline.confidence_threshold

                try:
                    AME = self.pipeline.model2.AME_Encoder(X) if X is not None else self.pipeline.confidence_threshold
                except:
                    AME = self.pipeline.confidence_threshold


            
            if X is None or len(X) == 0 or isinstance(X, int) or (isinstance(X, np.ndarray) and X.size == 0):
                # Get MLP predictions
                titles = None
                if titles is not None and len(titles) > 0:
                    if isinstance(titles[0], tuple):
                        mlp_titles = [t[0] for t in titles]
                    else:
                        mlp_titles = titles
                    
                    if not hasattr(self.pipeline, 'tfidf') or self.pipeline.tfidf is None:
                        self.pipeline.initialize_fitting(mlp_titles)

                    if isinstance(mlp_titles, (list, tuple, np.ndarray)):   
                        titles = mlp_titles[0]    
                        if isinstance(mlp_titles, (list, tuple, np.ndarray)):     
                            titles = titles[0] 
                    
                    if not isinstance(mlp_titles, list) or isinstance(titles, str):
                        if titles is not None:
                            mlp_titles = titles
                        X = self.pipeline.tfidf.transform(mlp_titles).toarray()
                
                if X_gen is not None:
                    X = X_gen
                if X is not None and len(X) > 0:
                    X = np.asarray(X)
                
            # MLP forward pass
            mlp_probs = self.pipeline.model3.forward(X, y=y_val, mode=infer_y)

             # Validate all MLP predictions at once
            mlp_pred_indices = np.argmax(mlp_probs, axis=1)
            if num_classes <= 0:
                num_classes = self.pipeline._get_num_classes(mlp_probs=mlp_probs)

            valid_mask = mlp_pred_indices < num_classes
            if not np.all(valid_mask):
                invalid_count = np.sum(~valid_mask)
                # Replace invalid indices with argmax within valid range
                for i in range(len(mlp_pred_indices)):
                    valid_probs = mlp_probs[i][:num_classes] if num_classes > 0 else mlp_probs[i]
                    if len(valid_probs) > 0 and i < len(mlp_pred_indices):
                        mlp_pred_indices[i] = int(np.argmax(valid_probs))
                    else:
                        mlp_pred_indices[i] = 0  # Default to first class  

            if sequence_ids is not None:
                print("\n[🔍] Using sequence encoding for transformer input due to low anisotropy.")
                input_ids = sequence_ids.copy()

            # verify samples for accurate answer from cache
            print('[🔍] Verifying Samples for possible predicted output in cache for accurate answer...')
            cached = self.pipeline.accurate_cache_lookup.lookup(
                x_mlp=X, 
                input_ids=input_ids)

            if cached is not None:
                if cached['similarity'] >= 0.95:                
                    print(f"[💎] Using verified cache "
                    f"(combined_sim={cached['similarity']:.1%}, "
                    f"hits={cached['hit_count']})")
                    result = {
                        'predicted': cached['prediction'],
                        'confidence': float(cached['confidence']),
                        'index': int(cached['index']),
                        'models_agree': True,
                        }    

                    print(f"\n[💎] Verified chosen label for samples: {cached['prediction']} || Confidence: {cached['confidence']:.1%}")  
                    
                    return result, cached['prediction'], cached['confidence']
                else:
                    print(f'[!] Similarity: {cached['similarity']} is low, Cannot pick label due to low certainty, Initiating advanced prediction...')
                    target_probs = mlp_probs.copy()
            else:
                print('[=] No verified output from cache available that matched samples, starting advanced prediction...')
                if self.pipeline.use_transformer and infer_y == 'classification':
                    if isinstance(input_ids, np.ndarray):
                        use_embedded = True
                    elif isinstance(input_ids, list):
                        if titles is not None:
                            input_ids_list = []
                            for title in titles:
                                if isinstance(title, tuple):
                                    title = title[0]
                                ids = self.pipeline.encode(title, self.pipeline.vocab)
                                input_ids_list.append(np.array(ids))
                            
                            input_ids = np.array(input_ids_list)
                            input_ids = self.pipeline.sequence_encoding(dataset, label_map=label_map)

                        use_embedded = True

                    target_probs = self.pipeline.predict_proba(input_ids, X, type='Hybrid', embedded=use_embedded)
                else:
                    target_probs = mlp_probs.copy()

                if target_probs is not None:
                    target_probs = target_probs[:mlp_probs.shape[0], :mlp_probs.shape[1]] 
                    if infer_y == 'classification':
                        target_probs = self.pipeline.model3.continuous_predictive_correction(self, target_probs, mlp_pred_indices, AME=AME, anisotropy=anisotropy)  
                    
                    target_pred_indices = np.argmax(target_probs, axis=1)    
                else:
                    target_probs = mlp_probs.copy()
                    target_pred_indices = np.argmax(target_probs, axis=1)

            if self.pipeline.use_lstm:
                label_bins = self.pipeline.scrapper_model.derive_bins_from_data(
                    y_val.ravel(),
                    n_bins=4,
                    labels=["Low", "Moderate", "High", "Extreme"],
                    value_range=(0.0, 1.0)
                )
                lstm_probs, _ = self.pipeline.ensemble._get_lstm_probs(input_ids, X, label_bins=label_bins)      
            else:
                label_bins = None
                lstm_probs = None

            AMR = 1.0 / (1.0 + np.exp(-AME))
            threshold = AMR + (self.pipeline.confidence_threshold + np.mean(self.error_counts)) / 2
            need_ensemble_method = self._compute_need_ensemble_method(anisotropy, AME, self.error_counts,
                                    anisotropy_threshold=threshold,
                                    ame_threshold=threshold,
                                    error_threshold=threshold,
                                    min_signals_required=2)
            
            results = []
            attention_data = [] if return_attention else None

            if target_pred_indices is None:
                target_pred_indices = np.argmax(mlp_probs.copy(), axis=1)

            if infer_y == 'classification' and titles is not None and len(titles) > 0:
                for i, title in enumerate(titles):
                    # Parse input
                    if isinstance(title, tuple):
                        display_title = title[0]
                        expected_label = title[1] if len(title) > 1 else None
                    else:
                        display_title = title
                        expected_label = None
                    
                    # MLP prediction 
                    
                    if i < len(mlp_pred_indices):           
                        mlp_class_idx = int(mlp_pred_indices[i])
                    else:
                        mlp_class_idx = int(target_pred_indices[i] if i < len(target_pred_indices) else target_pred_indices[0])

                    is_valid_index = 0 <= mlp_class_idx < num_classes
                    if not is_valid_index:
                        print(f'[⚠️] Invalid mlp_class_idx={mlp_class_idx} for sample {i} '
                            f'(num_classes={num_classes}, title="{display_title}") '
                            f'— marking as low-confidence unknown, NOT defaulting to class 0')

                        results.append({
                            'title'          : display_title,
                            'expected'       : expected_label,
                            'mlp_prediction' : 'Unknown',
                            'mlp_confidence' : 0.0,
                            'predicted'      : None,            # explicit unknown, not a fake class
                            'confidence'     : 0.0,
                            'mlp_class'      : None,
                            'is_valid'       : False,
                            'error'          : f'class_index_out_of_range(idx={mlp_class_idx}, num_classes={num_classes})'
                        })      
                        continue              

                    
                    if i < len(mlp_probs):
                        mlp_confidence = mlp_probs[i][mlp_class_idx]
                    else:
                        mlp_confidence = target_probs[i][mlp_class_idx] if i < len(target_probs) else target_probs[-1][mlp_class_idx]

                    mlp_label = reverse_map.get(mlp_class_idx, f"unknown_{mlp_class_idx}")

                    if lstm_probs is not None:
                        lstm_pred_indices = np.argmax(lstm_probs, axis=1)
                        lstm_class_idx = lstm_pred_indices[i]              
                        lstm_confidence = lstm_probs[i][lstm_class_idx]
                    else:
                        lstm_confidence = None

                    if i < len(target_pred_indices):
                        target_class_idx = target_pred_indices[i]
                    else:
                        target_class_idx = target_pred_indices[-1]
                    
                    if i < len(target_probs):
                        target_confidence = target_probs[i][target_class_idx]
                    else:
                        target_confidence = target_probs[-1][target_class_idx]
                
                    # Transformer prediction and blending
                    if trans_probs is not None and attn_weights is not None:
                        if i < len(trans_probs):
                            trans_probs_i = trans_probs[i]

                        trans_class_idx = np.argmax(trans_probs_i)
                        if isinstance(trans_probs_i, float):
                            trans_confidence = target_confidence
                        else:
                            trans_confidence = trans_probs_i[trans_class_idx]

                        trans_label = reverse_map.get(trans_class_idx, f"unknown_{trans_class_idx}")

                        if need_ensemble_method:
                            calibration = self.pipeline._calibrate_probs(target_probs, target_pred_indices, attn_weights, input_ids)
                            # Blend predictions (MLP decides class, transformer calibrates confidence)
                            mlp_weight = mlp_confidence / (target_confidence + trans_confidence + eps)
                            trans_weight = trans_confidence / (target_confidence + trans_confidence + eps)
                            if lstm_confidence is not None:
                                lstm_weight = lstm_confidence / (target_confidence + lstm_confidence + eps)
                                
                            calibration_weighting = calibration[target_class_idx] if target_class_idx < len(calibration) else 0.0
                                
                            # Weighted blend: calibration_weighting * calibrated + (1-weight) * mlp
                            if lstm_confidence is not None and lstm_weight is not None:
                                if i < len(target_probs):
                                    final_probs = mlp_weight * target_probs[i][:len(calibration)] + trans_weight * calibration[i][:len(calibration)] + lstm_weight * calibration[i][:len(calibration)]
                                else:
                                    final_probs = mlp_weight * target_probs[-1][:len(calibration)] + trans_weight * calibration[-1][:len(calibration)] + lstm_weight * calibration[-1][:len(calibration)]
                            else:
                                if i < len(target_probs):
                                    final_probs = mlp_weight * target_probs[i][:len(calibration)] + trans_weight * calibration[i][:len(calibration)]
                                else:
                                    final_probs = mlp_weight * target_probs[-1][:len(calibration)] + trans_weight * calibration[-1][:len(calibration)]

                            final_probs = self.pipeline._calibrate_probs(final_probs, mlp_class_idx, attn_weights, input_ids)
                            final_class_idx = target_class_idx
                            try:
                                final_confidence = final_probs[final_class_idx]
                            except IndexError:
                                final_confidence = np.max(final_probs) if isinstance(final_probs, np.ndarray) else np.mean(final_probs)

                            if isinstance(final_confidence, np.ndarray):
                                final_confidence = np.max(final_confidence)

                            # Calculate agreement
                            agreement = mlp_class_idx == trans_class_idx
                        else:
                            if lstm_confidence is None:
                                lstm_confidence = mlp_confidence
                            if mlp_confidence > trans_confidence and mlp_confidence > lstm_confidence and not mlp_confidence > 0.95:
                                final_probs = mlp_probs[i] if i < len(mlp_probs) else mlp_probs[-1]
                                final_class_idx = mlp_class_idx
                                final_confidence = mlp_confidence
                            elif trans_confidence > lstm_confidence:
                                final_probs = trans_probs[i] if i < len(trans_probs) else trans_probs[-1]
                                final_class_idx = trans_class_idx
                                final_confidence = trans_confidence
                            else:
                                chosen_probs = mlp_probs[i] if i < len(mlp_probs) else mlp_probs[-1]

                                final_probs = lstm_probs[i] if lstm_probs is not None else mlp_probs[-1]
                                final_class_idx = lstm_class_idx if lstm_probs is not None else mlp_class_idx
                                final_confidence = lstm_confidence if lstm_probs is not None else mlp_confidence
                                if final_confidence > 0.95:
                                    if lstm_probs is not None and mlp_probs.shape == lstm_probs.shape:
                                        final_probs = chosen_probs * lstm_probs[i]
                                        final_class_idx = mlp_class_idx
                                        final_confidence = mlp_confidence * lstm_confidence
                                    else:
                                        if lstm_probs is not None:
                                            final_probs = target_probs[i] if len(target_probs) == num_classes and i < len(target_probs) else lstm_probs[i]
                                        else:
                                            final_probs = trans_probs[i] if i < len(trans_probs) else trans_probs[:len(trans_probs)-1]
                                            
                                        final_class_idx = target_class_idx
                                        final_confidence = target_confidence * lstm_confidence

                                else:
                                    print(f"[🔄] No model chosen for sample {i} due to low confidence: MLP={mlp_confidence:.1%}, Transformer={trans_confidence:.1%}, LSTM={lstm_confidence:.1%}")

                            agreement = mlp_class_idx == trans_class_idx

                    else:
                        if i < len(mlp_probs):
                            final_probs = mlp_probs[i]
                        else:
                            final_probs = mlp_probs[-1]

                        final_class_idx = target_class_idx
                        final_confidence = target_confidence[-1] if isinstance(target_confidence, np.ndarray) else target_confidence
                        if isinstance(final_confidence, np.ndarray) or isinstance(final_confidence, list):
                            final_confidence = np.max(final_confidence)

                        trans_label = None
                        trans_confidence = 0.0
                        agreement = False
                    
                    final_label = reverse_map.get(final_class_idx, f"unknown_{final_class_idx}")
                    # Build result
                    result = {
                        'title': display_title,
                        'expected': expected_label,
                        'predicted': final_label,
                        'confidence': float(final_confidence),
                        'predicted_idx': int(final_class_idx),
                        'mlp_prediction': mlp_label,
                        'mlp_confidence': float(mlp_confidence),
                        'models_agree': bool(agreement),
                        'sec_predicted': None,
                        'sec_confidence': 0.0,
                        'sec_index': None,
                    }
                    
                    if trans_label is not None:
                        result['transformer_prediction'] = trans_label
                        result['transformer_confidence'] = float(trans_confidence)
                    
                    # Add top-k predictions
                    if isinstance(final_probs, (int, float)):
                        if use_transformer:
                            final_probs = self.pipeline._calibrate_probs(mlp_probs, mlp_class_idx, attn_weights, input_ids)
                        else:
                            final_probs = mlp_probs

                    final_probs = final_probs[:num_classes] if num_classes > 0 else final_probs
                    if show_proba:
                        if len(final_probs.shape) > 1:
                            final_probs = final_probs.flatten()
                        top_indices = np.argsort(final_probs)[-top_k:][::-1]
                        result['top_predictions'] = [
                            {
                                'label': reverse_map.get(idx, f"unknown_{idx}"),
                                'confidence': float(final_probs[idx])
                            }
                            for idx in top_indices.flatten() if idx in reverse_map 
                        ]
                        
                        # MLP top predictions
                        mlp_probs_i = mlp_probs[i][:num_classes] if num_classes > 0 else mlp_probs[i]
                        mlp_top = np.argsort(mlp_probs_i)[-top_k:][::-1]
                        result['mlp_top'] = [
                            {
                                'label': reverse_map.get(idx, f"unknown_{idx}"),
                                'confidence': float(mlp_probs_i[idx])
                            }
                            for idx in mlp_top if idx in reverse_map
                        ]
                        
                        # Transformer top predictions
                        if trans_probs is not None:
                            if trans_probs.ndim > 1:
                                trans_probs = trans_probs[i][:num_classes] if num_classes > 0 else trans_probs[i]
                            else:
                                trans_probs = trans_probs.copy()
                            if trans_probs is not None:
                                trans_top = np.argsort(trans_probs)[-top_k:][::-1]
                                result['transformer_top'] = [
                                    {
                                        'label': reverse_map.get(idx, f"unknown_{idx}"),
                                        'confidence': float(trans_probs[idx])
                                    }
                                    for idx in trans_top if idx in reverse_map
                                ]
                    
                    results.append(result)
                    
                    # Collect attention data if requested
                    if return_attention and attn_weights is not None:
                        attention_data.append({
                            'title': display_title,
                            'attention': attn_weights[i].tolist() if i < len(attn_weights) else None
                        })

            else: 
                print("[=] Initiating Continuous sample prediction.")
                if infer_y == 'classification':
                    transformer_takes = self._check_for_transformer_sequences(input_ids, min_seq_len=3, 
                                    min_samples=10, min_AME=0.5, min_anisotropy=0.5)
                    if not transformer_takes:
                        n_samples = mlp_probs.shape[0]
                        chosen_probs = mlp_probs.copy()
                    else:
                        n_samples = trans_probs.shape[0]
                        chosen_probs = trans_probs.copy()

                    lstm_pred_indices = np.argmax(lstm_probs, axis=1) if lstm_probs is not None else None
                    for i in range(n_samples):
                        outcome = self._compute_sample_prediction(
                            i, chosen_probs, target_probs, target_pred_indices,
                            trans_probs=trans_probs, lstm_probs=lstm_probs,
                            lstm_pred_indices=lstm_pred_indices,
                            attn_weights=attn_weights, input_ids=input_ids,
                            num_classes=num_classes, reverse_map=reverse_map
                        )

                        result = {
                            "title": f"Unknown",
                            "expected": 'Unknown',
                            **outcome
                        }
                        results.append(result)

                    if results is not None and isinstance(results[0], dict):
                        try:
                            final_probs = results[0]['final_probs'] if results else None
                            final_class_idx = results[0]['predicted_idx'] if results else None
                            agreement = results[0]['models_agree'] if results else None
                        except:
                            final_probs = results[0].get('final_probs', None)
                            final_class_idx = results[0].get('predicted_idx', None)
                            agreement = results[0].get('models_agree', None)
                    else:
                        final_probs = mlp_probs
                        final_class_idx = target_pred_indices
                        agreement = False
                else:
                    trans_mc = None
                    mlp_mc   = self.run_mc_dropout('mlp', X_test, y_test, input_ids=None, label_bin=None, AME=None, n_samples=50)

                    if self.pipeline.model2 is not None and self.pipeline.use_transformer:
                        trans_mc = self.run_mc_dropout('trans', X_test, y_test, input_ids=input_ids, label_bin=None, AME=AME, 
                                                        use_embedded=True,
                                                        n_samples=50)   
                    
                    results = []
                    lstm_results = []
                    for i in range(len(X_test)):
                        result = self._compute_sample_prediction_regression(
                            i, mlp_mc, config=self.config,
                            trans_preds=trans_mc
                        )
                        if self.pipeline.use_lstm and self.pipeline.lstm_engine is not None:
                            lstm_result = self.pipeline.predict_unnormalized(self.pipeline.lstm_engine, X_test[i], self.config, confidence_level=0.90)
                            lstm_results.append(lstm_result)
                        
                        results.append(result)
                    

            # Display results

            if infer_y == 'classification':
                verbose = False
                if float(results[0]['confidence']) < self.pipeline.confidence_threshold:
                    verbose = True

                prediction_cache = []
                confidence_cache = {}
                agreement_cache = {}
                _cache = []

                for i in range(len(results)):
                    each_label = results[i].get('predicted') if results else 'Unknown'
                    each_confidence = results[i].get('confidence') if results else 0.0
                    each_agreement = results[i].get('models_agree') if results else False

                    prediction_cache.append(each_label)
                    confidence_cache[i] = (each_label, each_confidence)
                    agreement_cache[i] = (each_label, each_agreement)

                voted_chosen_label = max(set(prediction_cache), key=prediction_cache.count)
                conf_cache = []
                for i in range(len(confidence_cache)):
                    if confidence_cache[i][0] == voted_chosen_label:
                        confidence = confidence_cache[i][1] 
                        conf_cache.append(confidence)

                float_counts = Counter(conf_cache)
                voted_confidence, _ = float_counts.most_common(1)[0]  

                voted_agreement = []
                for i in range(len(agreement_cache)):
                    if agreement_cache[i][0] == voted_chosen_label:
                        get_agreement = agreement_cache[i][1] 
                        voted_agreement.append(get_agreement)

                counts = Counter(voted_agreement)
                agreement_counts = counts[True]
                models_agreement = agreement_counts > counts[False]

                if not isinstance(voted_chosen_label, str):
                    for i in range(num_classes):
                        mlp_class_idx = int(mlp_pred_indices[i])
                        is_valid_index = 0 <= mlp_class_idx < num_classes

                        if mlp_class_idx > len(reverse_map) or not is_valid_index:
                            mlp_class_idx = int(np.argmax(mlp_probs[:len(reverse_map)-1]))
                            print(f"[⚠️] Clamping index {mlp_class_idx} → {mlp_class_idx}")  

                        if is_valid_index:
                            mlp_confidence = float(mlp_probs[i][mlp_class_idx]) 
                        else:
                            mlp_confidence = float(mlp_probs[:len(reverse_map)-1][mlp_class_idx]) 
                            
                        mlp_label      = reverse_map.get(mlp_class_idx, f"unknown_{mlp_class_idx}")
                        results.append({
                            'predicted'  : mlp_label,
                            'confidence' : mlp_confidence,
                            'mlp_class'  : mlp_class_idx,
                            'index'      : mlp_class_idx,
                            'models_agree': False,
                            'final_probs' : mlp_probs,
                            'is_valid'   : True,
                        })
                    
                    voted_chosen_label = results[0].get('predicted') if results else None
                    voted_confidence = results[0].get('confidence') if results else 0.0

                if isinstance(voted_chosen_label, int) or isinstance(voted_chosen_label, np.integer):
                    voted_chosen_label = str(voted_chosen_label)
                    
                if isinstance(voted_confidence, (np.ndarray, list)):
                    voted_confidence = np.mean(voted_confidence)

            if infer_y == 'regression' or not need_ensemble_method:
                if infer_y == 'classification':
                    print('[=] Getting Results...')
                    if not results[0].get('models_agree', True) and self.pipeline.use_transformer:
                        trans_conf_cache = []
                        for i in range(len(results)):
                            trans_confidence = results[i].get('trans_confidence', 1e-8)
                            if trans_confidence is not None and isinstance(trans_confidence, float):
                                trans_conf_cache.append(trans_confidence)
                        
                        if len(trans_conf_cache) > 0:
                            trans_counts = Counter(trans_conf_cache)
                            trans_confidence, _ = trans_counts.most_common(1)[0]
                            confidence = (voted_confidence + trans_confidence) / 2 + 1e-5
                        else:
                            confidence = voted_confidence

                        print(f'[=] Calibrating confidence around frequent Transformer confidence distributions: {confidence:.1%}')
                        
                    payload = {
                        'X_samples': X,
                        'input_ids': input_ids
                    }
                    print('[=] Displaying Test Performance Results....')
                    if titles is not None and len(titles) > 0:
                        correct, sec_correct = self.display_hybrid_results(payload, final_class_idx, results, top_k, verbose=True)
                        
                    if isinstance(voted_chosen_label, str) and voted_chosen_label.startswith("unknown") or float(voted_confidence) < self.pipeline.confidence_threshold:
                        if voted_chosen_label is None or voted_chosen_label.startswith('unknown'):
                            voted_chosen_label = 'Unknown'
                            voted_confidence = 1.0 - voted_confidence  #Invert confidence for unknown class
                            print(f"\n[⚠️] Frequent predicted label is {voted_chosen_label} with uncertain confidence: {voted_confidence:.1%}. Consider more consistent data for the model to learn from.")
                        else:
                            print(f"\n[🎯] Frequent Predicted label: {voted_chosen_label} || With Certain Confidence: {voted_confidence:.1%}")
                    else:
                        print(f'\n[🎯] Frequent Predicted label: {voted_chosen_label} || With uncertain confidence: {voted_confidence:.1%}')

                    return results, voted_chosen_label, voted_confidence

                else:
                    if len(results) > 0:
                        pred_results = []

                        if len(results[0]) > 0 and 'mlp' in results[0]:
                            mlp_pred = results[0]['mlp'].get('prediction', 'Unknown')
                        if len(results[0]) > 0 and 'mlp' in results[0]:
                            mlp_conf = results[0]['mlp'].get('confidence', 'Unknown')

                        if len(results[0]) > 0 and 'trans' in results[0]:
                            trans_pred = results[0]['trans'].get('prediction', 'Unknown')
                        
                        if len(lstm_results) > 1:
                            lstm_pred = np.array([r['prediction_original'][-1] for r in lstm_results])

                        if len(results[0]) > 0 and 'trans' in results[0]:
                            trans_conf = results[0]['trans'].get('confidence', 'Unknown')
                        if len(lstm_results) > 1:
                            lstm_conf = np.mean(np.array([r['mc_confidence'] for r in lstm_results]))
                            overall_conf = np.mean(np.array([r['overall'] for r in lstm_results]))
                            uncertainty = np.mean(np.array([r['gate_uncertainty'] for r in lstm_results]))

                        if mlp_pred is not None and mlp_conf > 0.0:
                            print(f'[🎯] MLP Prediction: {mlp_pred} || with confidence: {mlp_conf:.1%}')
                        if self.pipeline.use_lstm and lstm_pred is not None:
                            print(f'[⚡] LSTM Predictions: {lstm_pred[:top_k]} || with distributed confidence: {lstm_conf:.1%}')
                            print(f'[⚡] LSTM Uncertainty: {uncertainty:.1%}')
                            print(f'[⚡] LSTM Overall Confidence: {overall_conf:.1%}')
                        
                        if len(results) > 0 and 'trans' in results[0]:
                            print(f'[⚡] Transformer Prediction: {trans_pred} || with confidence: {trans_conf:.1%}')

                        prediction = {
                            'predicted': trans_pred,
                            'prediction': trans_pred,
                            'mlp_prediction': mlp_pred,
                            'confidence': trans_conf
                        }
                        pred_results.append(prediction)

                        if lstm_pred is None:
                            return pred_results, trans_pred, trans_conf
                        else:
                            return pred_results, lstm_pred, overall_conf
                    
                    else:
                        print('[>] Prediction results are None, cannot proceed with processing Final prediction')
                        return None, None, 0.0
                        

            else:
                if models_agreement and voted_confidence > self.pipeline.confidence_threshold and not voted_chosen_label.startswith("Unknown"):
                    print(f'[>] Agreement count: {agreement_counts}, Both models agree frequently on predicted label: {voted_chosen_label}')
                    print(f"\n[🎯] Proper Confidence of Final chosen label for input: {voted_chosen_label} || Confidence: {voted_confidence:.1%}")
                    return results, voted_chosen_label, voted_confidence
                    
                # Only recalibrate if models disagreed
                elif results and not models_agreement or not self.pipeline.agreement and self.pipeline.use_transformer:

                    need_peer_condition = not models_agreement and self.pipeline.peer_assistance_threshold > 0.3 and self.pipeline.use_transformer
                    print("\n[⚠️] Disagreement detected between MLP and Transformer predictions. Using calibrated probabilities for final decision.")
                    if voted_confidence < self.pipeline.confidence_threshold and need_peer_condition:
                        print('|| Uncertain advanced prediction, requesting peer assistance if allowed...')
                        final_probs = self.pipeline._handle_distributed_connections(final_probs, attn_weights, input_ids, agreement) 

                        final_idx, voted_confidence = self._get_voted_results(final_probs, reverse_map, len(X))
                        original_idx = final_idx

                        if final_idx > len(reverse_map):
                            final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                            print(f"[⚠️] Clamping {final_idx} → {final_idx}")

                        final_idx = int(final_idx)  
                        voted_chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")
               
                    elif self.pipeline.autonomous and need_peer_condition and attn_weights is not None:
                        if agreement is None:
                            agreement = False

                        print('[||] Iniating local peer output search in database for best output...')
                        final_probs = self.pipeline.distribution._handle_peer_agent_request(final_probs, attn_weights, input_ids, type='DevicePeer', agreement=agreement)

                        final_idx, voted_confidence = self._get_voted_results(final_probs, reverse_map, len(X))
                        original_idx = final_idx              

                        if final_idx > len(reverse_map):
                            final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                            print(f"[⚠️] Clamping {final_idx} → {final_idx}")  

                        final_idx = int(final_idx)  
                        voted_chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")

                    elif not models_agreement and voted_confidence > self.pipeline.confidence_threshold:
                        if final_confidence is not None and confidence < self.pipeline.confidence_threshold:
                            print("\n[⚠️] Low confidence detected, but both models don't agree. Using calibrated probabilities for final decision to ensure robustness.")
                            final_probs = self.pipeline.hybrid_prediction(rules, input_ids, dataset, X=X, y=y, use_embedded=use_embedded)
                        
                            final_idx, voted_confidence = self._get_voted_results(final_probs, reverse_map, len(X))
                            original_idx = final_idx
                            
                            if final_idx > len(reverse_map):
                                final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                                print(f"[⚠️] Clamping {final_idx} → {final_idx}")               
                            final_idx = int(final_idx)  
                        else:
                            print('[🎯] Stable confidence established, But both Models doesnt Agree, Re-evaluating...')   
                            final_probs = self.pipeline.hybrid_prediction(rules, input_ids, dataset, X=X, y=y, use_embedded=use_embedded)
                        
                            final_idx, voted_confidence = self._get_voted_results(final_probs, reverse_map, len(X))
                            original_idx = final_idx
                            if final_idx > len(reverse_map):
                                final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                                print(f"[⚠️] Clamping {final_idx} → {final_idx}")               
                            final_idx = int(final_idx)    

                    else:
                        if self.pipeline.use_transformer and need_ensemble_method:
                            print("\n[⚠️] Uncertain confidence and disagreement detected. Using ensemble method for final decision.")
                            input_forward = sequence_ids if sequence_ids is not None else input_ids
                            final_probs, details = self.pipeline.ensemble.predict_ensemble(input_forward, X, y, method='dynamic', embedded=use_embedded)
                        
                            final_idx, voted_confidence = self._get_voted_results(final_probs, reverse_map, len(X))
                            original_idx = final_idx 

                            if final_idx > len(reverse_map):
                                final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                                print(f"[⚠️] Clamping {final_idx} → {final_idx}")   

                            final_idx = int(final_idx)          
                            voted_chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")
                        else:
                            if final_probs is None:
                                final_probs = mlp_probs

                            target_cache = []
                            for i in range(len(target_pred_indices)):
                                target_cache.append(target_pred_indices[i])
                                
                            counter = Counter(target_cache)
                            predicted_idx, _ = counter.most_common(1)[0]

                            final_probs = self.calibration_penalized_check(target_probs, predicted_idx)
                            final_idx, voted_confidence = self._get_voted_results(target_probs, reverse_map, len(X)) 
                            original_idx = final_idx 

                            if final_idx > len(reverse_map):
                                final_idx = int(np.argmax(target_probs[:len(reverse_map)-1]))
                                print(f"[⚠️] Clamping {final_idx} → {final_idx}")

                            final_idx = int(final_idx)          
                            voted_chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")

                elif voted_confidence < self.pipeline.confidence_threshold and not self.pipeline.agreement and not results[0].get('models_agree', True):
                    if trans_probs is not None:
                        prob_entropy = self.calculate_entropy(final_probs)
                        normalized_entropy = prob_entropy / np.log(prob_entropy.shape[-1]) if prob_entropy.shape[-1] > 1 else 0
                        attn_quality = 1.0 / (1.0 + np.exp(-attn_weights.mean()) + eps) if attn_weights is not None else 0.5
                        anisotropy = self.pipeline.anisotropy_measurement(attn_weights.mean() if attn_weights is not None else 0.5)

                    else:
                        normalized_entropy = self.calculate_entropy(input_ids)  # Max entropy for uniform distribution
                        attn_quality = self.pipeline.confidence_threshold
                        anisotropy = self.anisotropy_measurement(input_ids) if hasattr(self.pipeline, 'anisotropy_measurement') else 0.5

                    mean_entropy = np.mean(normalized_entropy)

                    use_simple_prediction = (
                    anisotropy < 0.3 or
                    mean_entropy < 0.5 or  # High uncertainty
                    voted_confidence < 0.4 or  # Low confidence
                    not models_agreement or  # Disagreement
                    attn_quality < 0.4
                    )

                    if use_simple_prediction:
                        print("\n[⚡] Condition is poorly unviable to handle agreement. Using robust prediction method for better reliability.")
                        predicted_label, confidence = self.simple_pass_prediction(self.pipeline, titles=titles, label_map=label_map, X_raw=X, y=y, show_proba=show_proba, top_k=top_k)
                        if predicted_label is not None:
                            print(f"\n[🎯] Robust prediction result: {predicted_label} with confidence {confidence:.1%}")
                            return _, predicted_label, confidence

                    else:
                        final_idx, confidence =self._get_voted_results(final_probs, reverse_map, len(X)) 
                        original_idx = final_idx

                        if final_idx > len(reverse_map):
                            final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                            print(f"[⚠️] Clamping {final_idx} → {final_idx}")                      
                        final_idx = int(final_idx) 
                        voted_chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")

                else:
                    print("\n[🎯] Using initial Regular final prediction as final decision.")
                    final_idx, voted_confidence = self._get_voted_results(final_probs, reverse_map, len(X))

                    if final_idx > len(reverse_map):
                        final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                        print(f"[⚠️] Clamping {final_idx} → {final_idx}")                    
                    final_idx = int(final_idx)
                    voted_chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")

            if voted_confidence > 0.8:
                voted_confidence = (voted_confidence + performance_score) / 2 
                print(f'[⚡] Confidence calibrated to be around: {confidence}')

            if isinstance(voted_chosen_label, str) and voted_chosen_label.startswith("unknown") or float(voted_confidence) < self.pipeline.confidence_threshold:
                if voted_chosen_label.startswith("unknown"):
                    voted_chosen_label = 'Unknown'
                    voted_confidence = 1.0 - voted_confidence  #Invert confidence for unknown class

                print(f"\n[⚠️] Final prediction is {voted_chosen_label} with uncertain confidence: {voted_confidence:.1%}. Consider more consistent data for the model to learn from.")
            else:
                print(f"\n[🎯] Final chosen label for input: {voted_chosen_label} || Confidence: {voted_confidence:.1%}")  

            try:
                consecutive_probs = self.pipeline.distribution._handle_peer_agent_request(target_probs, attn_weights, input_ids, type='DevicePeer', agreement=agreement)
                sec_final_idx, sec_confidence = self._get_voted_results(consecutive_probs, reverse_map, len(X))

                if sec_final_idx > len(reverse_map):
                    sec_final_idx = int(np.argmax(consecutive_probs[:len(reverse_map)-1]))
                    print(f"[⚠️] Clamping {sec_final_idx} → {sec_final_idx}")            

                sec_final_idx = int(sec_final_idx)  
                if sec_confidence > 0.8:
                    sec_confidence = (sec_confidence + performance_score) + eps / 2 

                if 'sec_predicted' in results and results['sec_predicted'] is None and sec_chosen_label:
                    print('========== Second Prediction Initiative ==========')
                    print(f'[⚡] My Second Prediction: {sec_chosen_label}') 
                    print(f'[⚡] Confidence: {sec_confidence:.1%}')  

                    results[0]['sec_predicted'] = sec_chosen_label
                    results[0]['sec_confidence'] = sec_confidence
                    results[0]['sec_index'] = sec_final_idx

                    if confidence > results[0]['confidence']:
                        results[0]['predicted'] = voted_chosen_label
                        results[0]['confidence'] = voted_confidence
                        results[0]['predicted_idx'] = final_idx
                else:
                    print('[!] No prediction in results cache are found!')

            except Exception as e:
                print(f'[!] Error initiating second prediction in Advanced prediction method: {e} ')

                results[0]['sec_predicted'] = voted_chosen_label
                results[0]['sec_confidence'] = voted_confidence
                results[0]['sec_index'] = final_idx


        except Exception as e:
            print(f"[!] Error in advanced prediction method: {e}, Initiating regular prediction method...")
            traceback.print_exc()            
            try:
                if infer_y == 'classification':
                    results = self.regular_prediction_method(titles=titles, label_map=label_map, rules=rules, X=X, y=y, show_proba=False, top_k=3, batch_size=2, use_transformer=self.pipeline.use_transformer)
                    voted_chosen_label = results[0]['predicted']
                    voted_confidence = results[0]['confidence']
                else:
                    raise Warning('[>] Regression tasks cannot proceed with simple prediction pipeline used for classification!')
                    return None, None, None

            except Exception as error:
                print(f'[= ! =] Error in all prediction method: {error}')
                traceback.print_exc()
                results, voted_chosen_label, voted_confidence = None, None, 0.0

        print('[=] Displaying Results....')
        payload = {
            'X_samples': X,
            'input_ids': input_ids
        }
        if titles is not None and len(titles) > 0:
            correct, sec_correct = self.display_hybrid_results(payload, final_class_idx, results, top_k, verbose=True)

        if sec_chosen_label and sec_correct > correct:
            print(f'[⚡] Second Prediction: {sec_chosen_label} has higher accuracies, relying on: {sec_chosen_label} as final label.')
            chosen_label = sec_chosen_label # overrides previous chosen label if accuracy is higher
        elif self.pipeline.autonomous and sec_confidence > self.pipeline.confidence_threshold:
            print(f'[⚡] Autonomous Prediction used second predicted label: {sec_chosen_label}')
            voted_chosen_label = sec_chosen_label
        else:
            print(f'[⚡] Final Prediction: {voted_chosen_label} with confidence: {voted_confidence:.2%}')
            voted_chosen_label = voted_chosen_label

        # delete pipelines cache
        print('[🔍] Pipelines Cache Cleaned!')
        self.pipeline.cache.clear()

        return results, voted_chosen_label, voted_confidence

    def _repair_class_counters(self, n_classes):
        """Counters are ALWAYS float64 1-D of length n_classes. Anything else is reset loudly."""
        for name in ('pred_counts', 'error_counts'):
            arr = getattr(self, name, None)
            try:
                arr = None if arr is None else np.asarray(arr, dtype=np.float64)
            except (TypeError, ValueError):          # e.g. a "[0. 0. ...]" string from a bad restore
                arr = None
            if arr is None or arr.ndim != 1 or arr.shape[0] != n_classes:
                if arr is not None:
                    print(f'[⚠️] {name} had shape {arr.shape}, expected ({n_classes},) - resetting it.')
                arr = np.zeros(n_classes, dtype=np.float64)
            setattr(self, name, arr)

    def calibration_penalized_check(self, final_probs, predicted_index):
        """Update per-class error counters (class reputation) and return the final probabilities adjusted for classes that have been wrong often."""
        # update class reputation.
        n_classes = len(self.label_map)
        decay = self.pipeline.error_decay
        self._repair_class_counters(n_classes)
        self.error_counts *= decay
        self.pred_counts  *= decay
    
        if final_probs is None:
            print('[!] Warning final probabilities is None! returning the None probabilities...')
            return final_probs
    
        try:
            # predicted_index can be a scalar (majority vote) or one index per sample
            idx = np.atleast_1d(np.asarray(predicted_index)).astype(int).ravel()
            idx = idx[(idx >= 0) & (idx < n_classes)]
            np.add.at(self.pred_counts, idx, 1.0)         # each predicted sample counts once
    
            for c in range(n_classes):
                if len(self.pred_counts) < c:             
                    if isinstance(self.pred_counts[c], (int, float)) and self.pred_counts[c] > 0:
                        error_rate = self.error_counts[c] / (self.pred_counts[c] + 1e-8)
                        reputation = 1.0 / (1.0 + error_rate)
                        if isinstance(error_rate, (list, np.ndarray)):
                            if len(error_rate) != len(final_probs):
                                reputation = 1.0 / (1.0 + np.mean(error_rate))
                        if c < len(final_probs):
                            final_probs[c] *= reputation
    
            prob_sum = final_probs.sum()                  
            if prob_sum > 1e-8:
                final_probs /= prob_sum
    
        except Exception as e:
            print(f'[!] Cant check and calibrate probs based on penalty due to: {e}')
    
        return final_probs

        
    def display_hybrid_results(self, payload, predicted_index, results, top_k=3, verbose=False, acc_threshold=0.3):
        """Print a formatted table of predictions, confidences, top-k labels and accuracy against expected labels."""
        print("\n" + "="*80)
        print("[🎯] === PREDICTION RESULTS === ")
        print("="*80)

        correct = 0
        sec_correct = 0
        error = 0
        total_with_expected = 0
        X_samples, input_ids = payload['X_samples'], payload['input_ids']   

        if len(self.error_counts) < 1:
            self.error_counts = np.zeros(len(self.label_map), dtype=np.float32)

        for idx, result in enumerate(results):
            print(f"\n{idx+1}. 📌 '{result['title']}'")   
            if result.get('expected'):
                total_with_expected += 1
                status = ": ✅" if result['predicted'] == result['expected'] else ": ❌"
                print(f"[=] First Expectation: {result['expected']} || Model Answer: {status}")

                if 'sec_predicted' in result:
                    sec_status = ": ✅" if result['sec_predicted'] == result['expected'] else ": ❌"                
                    print(f"[=] Second Expectation: {result['expected']} || Model Answer: {sec_status}")    
                    if 'sec_predicted' in result and 'sec_confidence' in result and result['sec_predicted'] == result['expected']:
                        self.pipeline.accurate_cache_lookup.add_verified(
                            X_samples, input_ids, 
                            result['sec_predicted'], result['sec_confidence'], result['sec_index'],
                            source='automatic_verified')
                        
                        sec_correct += 1
                    else:
                        error += 1
                        if result.get('predicted_idx') < len(self.error_counts):
                            if isinstance(self.error_counts[result.get('predicted_idx')], (int, float)):
                                self.error_counts[result.get('predicted_idx')] += 1.0

                if result['predicted'] == result['expected']:
                    if 'sec_index' in result and result['sec_index'] is None:
                        result['sec_index'] = None

                    if 'predicted_idx' in result and 'confidence' in result and 'predicted' in result:
                        self.pipeline.accurate_cache_lookup.add_verified(
                            X_samples, input_ids, 
                            result['predicted'], result['confidence'], result['predicted_idx'],
                            source='automatic_verified')              
                        correct += 1      
                else:
                    error += 1
                    if result.get('predicted_idx') < len(self.error_counts):
                        if len(self.error_counts) > 0 and isinstance(self.error_counts[result.get('predicted_idx')], (int, float)):
                            self.error_counts[result.get('predicted_idx')] += 1.0
        
            # Agreement indicator
            agree_symbol = "✓" if result.get('models_agree', True) else "⚠️"
            print(f"[=] {agree_symbol} FINAL: {result['predicted']} ({result['confidence']:.1%})")

            # MLP vs Transformer
            print(f"      ├─ [⚡] MLP: {result['mlp_prediction']} ({result['mlp_confidence']:.1%})")
            if result.get('transformer_prediction'):
                arrow = "⬆️" if result['transformer_confidence'] > result['mlp_confidence'] else "⬇️"
                print(f"      └─ [🌀] Transformer: {result['transformer_prediction']} ({result['transformer_confidence']:.1%}) {arrow}")
            
            # Top predictions
            if 'top_predictions' in result:
                print(f"\n [🔍] Top {top_k} possibilities:")
                for j, pred in enumerate(result['top_predictions'][:top_k], 1):
                    bar = '█' * int(pred['confidence'] * 20)
                    print(f"         {j}. {pred['label']:20s} {bar} {pred['confidence']:.1%}")

        if total_with_expected > 0:
            accuracy = correct / total_with_expected
            print(f"\n📊 Accuracy: {correct}/{total_with_expected} = {accuracy:.1%}")
            if accuracy > acc_threshold:
                if self.pipeline.lstm_engine:
                    self.pipeline.storage.save_weights(self.pipeline.memory_name, model_type='Pipeline') 

        return correct, sec_correct
