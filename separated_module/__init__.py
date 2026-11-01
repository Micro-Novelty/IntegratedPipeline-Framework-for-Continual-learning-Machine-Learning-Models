"""AbstractIntegratedModule package (split from the original single-file module).

Every public class/function that used to live in AbstractIntegratedModule.py is
re-exported here, so ``from AbstractIntegratedModule import IntegratedPipeline``
keeps working.
"""

from .messaging_types import MessagePriority, WrapperState, AsyncTask, TrustLevel, RequestStatus, AsyncRequest, SecureMessage, Message
from .security_config import SecurityConfig, SecurityLevel, SecurityError, AdminRole
from .singleton import SingletonMeta, Singleton
from .data_utils import fix_serialized_string, NormalizationConfig, create_sequences, get_dataset, dynamic_test_size
from .memory_store import get_database_path, setup_memory_table, linear_cka, set_memory, save_memory
from .quantization import QuantConfig, fake_quant_weight, ActObserver
from .calibrated_init import calibrated_init, describe, load_into_qat_layers
from .metrics import calculate_ece, expected_calibration_error
from .geometric_weight_shaping import GeometricWeightShaping
from .activations import sigmoid, sigmoid_deriv, tanh_deriv, Activation
from .losses import Loss
from .hnsw import HNSW
from .per_head_memory import PerHeadMemory
from .snn_encoders import PoissonEncoder, RateEncoder, LatencyEncoder
from .snn_layers import HeaviSide, LIFDenseLayer, ReadoutLayer, SpikeDropout, PerSpikeTimingMemory, STDPLearner, ThresholdHomeostasis
from .snn_utility import SNNUtility
from .optimizers import SGD, Adam, AdamOptimizer
from .snn_network import softmax_cross_entropy, mse_loss, SNNNetwork
from .snn_diagnostics import layer_firing_stats, weight_stats, grad_norm_stats, network_report, print_report, check_health, DiagnosticsTracker
from .snn_calibration import softmax_probs, predict_with_confidence, TemperatureScaler, mc_confidence, print_confidence_report
from .snn_training import build_snn_network, iterate_minibatches, stdp_pretrain, initialize_snn_network, train_snn_network, snn_predict
from .target_memory import TargetMemory, blend, TargetMemoryMixin
from .transformer import Transformer
from .knn_transformer import KNNAugmentedTransformer
from .dense_layers import Dense, QATDense, SoftmaxOutput
from .mlp import MLP
from .lstm import LSTMCell, LSTMNetwork, LSTMEngine
from .cross_session import CrossSessionAutomation
from .explainability import ExplainabilityModule
from .model_storage import ModelStorage
from .api_security import RateLimiter, InputSanitizer, ApiKeyManager
from .message_queues import AsyncMessageQueue, ThreadedMessageQueue
from .query_node import QueryNode
from .ensemble import WeightedEnsemblePredictor
from .agent_distributed_inference import AgentDistributedInference
from .auto_batcher import AutoBatcherAutomation
from .answer_cache import AccurateAnswerCache
from .integrated_pipeline import IntegratedPipeline
from .async_manager import AsyncResultQueue, WorkerPool, PipelineAsyncManager
from .prediction_manager import PipelinePredictionManager
from .peer_agent import ConsecutivePeerAgent
from .cohesive_deployment import CohesiveAgentDeployment
from .cluster_runners import run_secure_agent_cluster, example_async_with_result_queue
from .cli_tests import initiate_cohesive_agent_deployment_test, initiate_prediction_usage, initiate_with_retries, initiate_graceful_shutdown, AsyncWrappertest, PermissiveTest
