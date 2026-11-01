# AbstractIntegratedModule (split package)

The original single 26,500-line `AbstractIntegratedModule.py` split into one file per topic.
Code was moved verbatim (checked block-by-block against the original); the only code changes are listed below.

## Layout

| File | What it contains | Main contents |
|---|---|---|
| `common.py` | Shared setup: optional Cython / Rust accelerators, availability flags, pipeline lock, `APP_BASE_DIR` | `_OPT_AVAILABLE`, `_RUST_MODULE_AVAILABLE`, `wc`, `optimized_*` |
| `messaging_types.py` | Message, queue and request data types, enums and priorities shared across the async/agent layers. | `MessagePriority`, `WrapperState`, `AsyncTask`, `TrustLevel`, `RequestStatus`, `AsyncRequest` … |
| `security_config.py` | Security configuration objects: security levels, admin roles, SecurityConfig and SecurityError. | `SecurityConfig`, `SecurityLevel`, `SecurityError`, `AdminRole` |
| `singleton.py` | Singleton metaclass / base class used to guarantee a single shared instance. | `SingletonMeta`, `Singleton` |
| `data_utils.py` | Dataset helpers: DB-string repair, normalisation config, sliding-window sequences, train/val splitting, scalar conversion. | `fix_serialized_string`, `NormalizationConfig`, `create_sequences`, `get_dataset`, `dynamic_test_size` |
| `memory_store.py` | SQLite-backed "model memory" primitives: schema, connection helpers, CKA similarity, set/save memory. | `get_database_path`, `setup_memory_table`, `linear_cka`, `set_memory`, `save_memory` |
| `quantization.py` | Quantisation-aware-training primitives: QuantConfig, fake weight quantisation and activation observers. | `QuantConfig`, `fake_quant_weight`, `ActObserver` |
| `calibrated_init.py` | Data-aware ("calibrated") weight initialisation and loading of calibrated weights into QAT layers. | `calibrated_init`, `describe`, `load_into_qat_layers` |
| `metrics.py` | Calibration metrics (Expected Calibration Error). | `calculate_ece`, `expected_calibration_error` |
| `geometric_weight_shaping.py` | GeometricWeightShaping: geometry-aware weight shaping / regularisation used by Dense, Transformer and LSTM layers. | `GeometricWeightShaping` |
| `activations.py` | Activation functions and their derivatives. | `sigmoid`, `sigmoid_deriv`, `tanh_deriv`, `Activation` |
| `losses.py` | Loss functions and the shape-alignment helper they rely on. | `Loss` |
| `hnsw.py` | HNSW approximate nearest-neighbour index. | `HNSW` |
| `per_head_memory.py` | Per-attention-head k-NN memory used by the KNN-augmented transformer. | `PerHeadMemory` |
| `snn_encoders.py` | Spike encoders for the spiking neural network (Poisson, rate, latency). | `PoissonEncoder`, `RateEncoder`, `LatencyEncoder` |
| `snn_layers.py` | Spiking-network building blocks: surrogate spike function, LIF layer, readout, dropout, timing memory, STDP, homeostasis. | `HeaviSide`, `LIFDenseLayer`, `ReadoutLayer`, `SpikeDropout`, `PerSpikeTimingMemory`, `STDPLearner` … |
| `snn_utility.py` | SNNUtility: seeding / RNG helpers for the SNN code. | `SNNUtility` |
| `optimizers.py` | Optimisers: SGD, Adam and AdamOptimizer. | `SGD`, `Adam`, `AdamOptimizer` |
| `snn_network.py` | SNN loss helpers and the SNNNetwork container (forward / backward / train-eval switching). | `softmax_cross_entropy`, `mse_loss`, `SNNNetwork` |
| `snn_diagnostics.py` | SNN diagnostics: firing/weight/gradient statistics, health checks, reports and DiagnosticsTracker. | `layer_firing_stats`, `weight_stats`, `grad_norm_stats`, `network_report`, `print_report`, `check_health` … |
| `snn_calibration.py` | SNN confidence utilities: softmax probabilities, temperature scaling, Monte-Carlo confidence. | `softmax_probs`, `predict_with_confidence`, `TemperatureScaler`, `mc_confidence`, `print_confidence_report` |
| `snn_training.py` | SNN construction, mini-batching, STDP pre-training, training and prediction entry points. | `build_snn_network`, `iterate_minibatches`, `stdp_pretrain`, `initialize_snn_network`, `train_snn_network`, `snn_predict` |
| `target_memory.py` | TargetMemory and TargetMemoryMixin: memory-blended predictions for the transformer. | `TargetMemory`, `blend`, `TargetMemoryMixin` |
| `transformer.py` | Transformer model. | `Transformer` |
| `knn_transformer.py` | KNNAugmentedTransformer: transformer with per-head k-NN memory. | `KNNAugmentedTransformer` |
| `dense_layers.py` | Dense, QATDense and SoftmaxOutput layers. | `Dense`, `QATDense`, `SoftmaxOutput` |
| `mlp.py` | MLP classifier/regressor. | `MLP` |
| `lstm.py` | LSTM cell, network and training engine. | `LSTMCell`, `LSTMNetwork`, `LSTMEngine` |
| `cross_session.py` | CrossSessionAutomation: automation that spans training sessions. | `CrossSessionAutomation` |
| `explainability.py` | ExplainabilityModule: human-readable explanations of predictions. | `ExplainabilityModule` |
| `model_storage.py` | ModelStorage: persistence of models, weights and memories (SQLite / Rust backend). | `ModelStorage` |
| `api_security.py` | Request-level security helpers: RateLimiter, InputSanitizer, ApiKeyManager. | `RateLimiter`, `InputSanitizer`, `ApiKeyManager` |
| `message_queues.py` | Async and threaded message queues for the agent layer. | `AsyncMessageQueue`, `ThreadedMessageQueue` |
| `query_node.py` | QueryNode: query-handling node. | `QueryNode` |
| `ensemble.py` | WeightedEnsemblePredictor: combines MLP, transformer and LSTM predictions. | `WeightedEnsemblePredictor` |
| `agent_distributed_inference.py` | AgentDistributedInference: distributed / secure inference between agents. | `AgentDistributedInference` |
| `auto_batcher.py` | AutoBatcherAutomation: automatic batch-size selection. | `AutoBatcherAutomation` |
| `answer_cache.py` | AccurateAnswerCache: cache of verified answers. | `AccurateAnswerCache` |
| `integrated_pipeline.py` | IntegratedPipeline: end-to-end training pipeline tying all models together. | `IntegratedPipeline` |
| `async_manager.py` | Async result queue, worker pool and PipelineAsyncManager. | `AsyncResultQueue`, `WorkerPool`, `PipelineAsyncManager` |
| `prediction_manager.py` | PipelinePredictionManager: regular / advanced prediction entry points. | `PipelinePredictionManager` |
| `peer_agent.py` | ConsecutivePeerAgent. | `ConsecutivePeerAgent` |
| `cohesive_deployment.py` | CohesiveAgentDeployment: orchestrates agents, queues and pipelines. | `CohesiveAgentDeployment` |
| `cluster_runners.py` | Async runners for secure agent clusters. | `run_secure_agent_cluster`, `example_async_with_result_queue` |
| `cli_tests.py` | Interactive smoke-test helpers (PermissiveTest etc.). | `initiate_cohesive_agent_deployment_test`, `initiate_prediction_usage`, `initiate_with_retries`, `initiate_graceful_shutdown`, `AsyncWrappertest`, `PermissiveTest` |
| `__main__.py` | `python -m AbstractIntegratedModule` runs the interactive smoke test (`PermissiveTest`) | |
| `__init__.py` | Re-exports every public class/function, so `from AbstractIntegratedModule import X` still works | |

Dependencies only point "downwards" (no circular imports); `__init__.py` imports modules in dependency order.

## Changes beyond moving code

1. Each module has the imports it needs, and `common.py` holds the optional Cython/Rust loading.
   If an accelerator is missing, its names are `None` (previously undefined) and the existing
   `_OPT_AVAILABLE` / `_RUST_MODULE_AVAILABLE` checks still guard every use.
2. Five places located files with `os.path.dirname(os.path.abspath(__file__))`. They now use
   `APP_BASE_DIR` (the folder that *contains* the package, i.e. where the old file lived),
   so databases / certificates are found in the same place as before.
3. `logger = logging.getLogger(__name__)` is created per module (names like `AbstractIntegratedModule.model_storage`).
4. Docstrings were added; no logic was changed.

## Usage

Put the `AbstractIntegratedModule/` folder where the old `.py` file was (and remove or rename the old file,
otherwise the two share one import name). `AbstractOptimizedModules` and `abstract_weights_core` are still
imported from the same place as before (they must be importable, e.g. in that same folder).

    from AbstractIntegratedModule import IntegratedPipeline, MLP, Transformer
    python -m AbstractIntegratedModule     # interactive smoke test
