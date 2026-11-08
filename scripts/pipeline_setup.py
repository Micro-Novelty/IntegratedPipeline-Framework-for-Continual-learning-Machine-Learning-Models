# save as pipeline_setup.py (or change it with other names)
# NOTE: this workflow is customizable specify to your needs.

from AbstractIntegratedModule import IntegratedPipeline
from AbstractIntegratedModule import PipelinePredictionManager
from AbstractIntegratedModule import Transformer
from AbstractIntegratedModule import QuantConfig
from AbstractIntegratedModule import set_memory
from AbstractIntegratedModule import save_memory
import numpy as np

# db_path is your custom initialized sqlite database name (ex: data_storage.db)
# label_csv is the name of your .txt file that contains labels with csv format used for Prediction.
def init_setup(db_path, X, y, memory_name=None, csv=None, new_task=None, new_config=None, cols=None, port=None):
    # set_memory() is used to load the Model's Memory, csv fie for Training, task and label map using you previouslyr given X and y samples
    # new_memory argument is used to declare a new memory name if the searched memory is not present.
    # port is used for P2P (Optional)
    if port is None:
        port = 5001

    memory_name, label_csv, task, label_map, feature_cols, config = set_memory(db_path, X, y, new_memory=memory_name)

    if label_csv is None:
        label_csv = csv
    if task is None:
        task = new_task
    if feature_cols is None:
        feature_cols = cols
    if config is None:
        config = new_config

    # if task and csv training file is None in the database, set your own csv training file and task that matches the actual X and y samples category (classification or regression)

    main_model = IntegratedPipeline(
    memory_name=memory_name,  # memory name for the Model you already initialized
    use_async=True, # local asynchronous prediction is permitted, if not PipelineAsyncManager wont start asynchronous prediction.
    agent_port=port, # this port is used to set AgentDistributedInference server (optional)
    ssl_cert_file=None, ssl_key_file=None,# provide your cert_file path or key_file path (optional)
    ssl_context=None, # used by the Agent server. (optional)
    client_ssl_context=None # used by the client. (optional)
    ) 
    
    main_prediction = PipelinePredictionManager(
        main_model, # your initialized pipeline
        label_csv=label_csv, 
        # your filename that contains the csv and contains the CSV format.
        # the Agent will automatically searched the nearby folder like: downloads, data, and desktop folder.
        target_title='window_title', label='label')

    if config is not None:
        # set regression config to PredictionManager for prediction on regression task:
        main_prediction.config = config

    # initialize label map name if its None in the memory database
    if label_map is None and not label_csv is None:
        titles, _, label_map = main_prediction.load_labels_from_csv(
            label_csv,  # the name of your .txt file with CSV format.
            'window_title', 'label')
            

    # Quantization config is optional
    qcfg = QuantConfig(enabled=True, w_bits=8, a_bits=8)

    # Set up the main model with the specified configurations you wants, this will override the default configurations of the IntegratedPipeline, 
    # you can change the parameters to suit your dataset and requirements.
    main_model.model_setup(mlp_train_epochs=1000, tf_train_epochs=10, tf_lr=0.1, mlp_lr=0.5,
                            lstm_train_epochs=50, lstm_lr=5e-1, lstm_hidden_dim=64, tf_heads=4, tf_d_model=32,
                            error_decay=0.75)

    return memory_name, main_model, main_prediction, task, label_csv, label_map, qcfg, feature_cols

# handle training round for one task at a time:
def one_train_round(memory_name, main_model, db_path, X, y, with_training=False, titles=None, rules=None, 
                task=None, label_map=None, feature_cols=None, label_csv=None, use_qat=True, qcfg=None, config=None):
    # TRAINING BLOCK: This is where you train the MLP, Transformer, and LSTM models with the specified configurations you want.

    if task == 'classification':
        # NOTE:
        # set input_dim and num_classes based on the shape of X and the number of unique labels in label_map.
        # this are important for the MLP and Transformer model to know how many input features and output classes it has to handle.
        input_dim = X.shape[1] if X.ndim > 1 else 1
        num_classes = max(len(label_map), int(np.max(y)) + 1)

        # Re initialize all models everytime a new Training is started, each weights will be saved to database later.
        main_model.initialize_mlp_model(titles=titles, rules=rules, label_map=label_map, input_dim=input_dim, output_dim=num_classes, hidden_layers=[12, 17, 35], use_qat=use_qat, qcfg=qcfg)

        # vocab size set to 1 if titles and rules are None.
        main_model.initialize_transformer_model(titles=titles, rules=rules, label_map=label_map, vocab_size=1, num_classes=num_classes, d_model=64, n_heads=4, n_features=None,
                                                        attn_dropout=0.1, ffn_dropout=0.05, weight_decay=1e-2, task=task)
        main_model.initialize_lstm_model(input_size=1, hidden_size=32, output_size=1, dropout_rate=0.1, seed=42)

        # Memory Loading Block: This is where you load the model memory if you have previously saved it, this will allow you to continue training or prediction from where you left off.
        # This function will automatically adds the loaded weights to the current model, so you can continue training or prediction without losing any progress.
        memory_loaded, lstm_weights, transformer_weights, mlp_weights, = main_model.load_model_memory()

        if not memory_loaded and with_training:
            main_model.train_mlp_model(titles=titles, rules=rules, label_map=label_map, X=X, y=y, task=task, train_epochs=500, momentum=0.9, lr=0.01)
            main_model.train_transformer(titles=titles, rules=rules, label_map=label_map, X=X, y=y,  task=task, d_model=64, train_epochs=10, 
                                            mode='dynamic_backward', lr=0.5, batch_size=4)
            main_model.train_lstm(titles=titles, rules=rules, label_map=label_map, raw_X=X, raw_Y=y, n_train=0.8, n_display=5, train_epochs=10, lr=0.1)
            
            # save memory right after training to your custom initialized database: (ex: to your data_storage.db)
            save_memory(db_path, memory_name, label_csv, X, y, task, label_map)

            # save weights to database
            main_model.save_weights(memory_name, model_type='Pipeline') 
        else:
            print('[>] Memory is here, ignoring training')

    elif task == 'regression':
        input_dim = X.shape[1] if X.ndim > 1 else 1
        num_classes = y.shape[1]

        main_model.initialize_mlp_model(titles=titles, rules=rules, label_map=label_map, input_dim=input_dim, output_dim=num_classes, hidden_layers=[12, 17, 35], use_qat=True, qcfg=qcfg)

        # vocab size is always set to 1 in regression samples.
        main_model.initialize_transformer_model(titles=titles, rules=rules, label_map=label_map, vocab_size=1, num_classes=num_classes, d_model=64, n_heads=4, n_features=len(feature_cols),
                                                        attn_dropout=0.1, ffn_dropout=0.05, weight_decay=1e-2, task=task)
        main_model.initialize_lstm_model(input_size=1, hidden_size=32, output_size=1, dropout_rate=0.1, seed=42)
        memory_loaded, lstm_weights, transformer_weights, mlp_weights, = main_model.load_model_memory()

        if not memory_loaded and with_training:
            main_model.train_mlp_model(titles=titles, rules=rules, label_map=label_map, X=X, y=y, task=task, train_epochs=500, momentum=0.9, lr=0.01)
            main_model.train_transformer(titles=titles, rules=rules, label_map=label_map, X=X, y=y,  task=task, d_model=64, train_epochs=10, 
                                                mode='dynamic_backward', lr=0.5, batch_size=4)
            main_model.train_lstm(titles=titles, rules=rules, label_map=label_map, raw_X=X, raw_Y=y, n_train=0.8, n_display=5, train_epochs=10, lr=0.1)

            # save memory right after training to your custom initialized database: (ex: to your data_storage.db)
            save_memory(db_path, memory_name, label_csv, X, y, task, label_map, feature_cols=feature_cols, config=config)

            # save weights to database
            main_model.save_weights(memory_name, model_type='Pipeline') 

        else:
            print('[>] Regression training is skipped')

    else:
        print('[>] Invalid task name')

# for prediction only.
def predict(main_prediction, X, y, titles=None, rules=None, label_map=None):
    results, chosen_label, confidence = main_prediction.advanced_prediction_method(
    titles=None, label_map=label_map, rules=None, # titles and rules can be set to None (Optional samples), but label_map must NOT be None.
        X=X, y=y, # you could create your own X and y samples and put it here (Optional, y sample must already be one hot encoded first).
        top_k=3, batch_size=4)

    return results, chosen_label, confidence

# for training only
    # if new_memory is set to None, save_memory() AbstractIntegratedModule function will automatically create a random memory name (e.g: memory-07g6g3g2vgg).
    # set new_memory to None if you are planning to do continuous learning where creating a different memory name is inefficient and took time.
def train_run(db_path, X, y, task, pred_config=None, label_csv=None, feature_cols=None, 
            new_memory=None, with_training=True, port=None):
    memory_name, main_model, main_prediction, task, label_csv, label_map, qcfg, feature_cols = init_setup(db_path, X, y, memory_name=new_memory, csv=label_csv, new_task=task, new_config=pred_config, cols=feature_cols, port=port)

    one_train_round(memory_name, main_model, db_path, X, y, with_training,
        task=task, label_map=label_map, feature_cols=feature_cols, label_csv=label_csv, use_qat=True, qcfg=qcfg, config=pred_config)

    # return main_prediction for later usage (like P2P or asynchronous prediction)
    return main_prediction, main_model
    
# one full train -> predict.
def one_run(db_path, X, y, task=None, pred_config=None, label_csv=None, feature_cols=None, 
            new_memory=None, with_training=True, port=None):
    memory_name, main_model, main_prediction, task, label_csv, label_map, qcfg, feature_cols = init_setup(db_path, X, y, memory_name=new_memory, csv=label_csv, new_task=task, new_config=pred_config, cols=feature_cols, port=None)
    one_train_round(memory_name, main_model, db_path, X, y, with_training,
        task=task, label_map=label_map, feature_cols=feature_cols, label_csv=label_csv, use_qat=True, qcfg=qcfg, config=pred_config)
    
    results, chosen_label, confidence = predict(main_prediction, X, y, label_map=label_map)

    return results, chosen_label, confidence

# ================== Continual learning setup for kNN Transformer (OPTIONAL, can be removed) =======================
def knn_transformer_round(memory_name, main_prediction, db_path, X, y, with_training=False, titles=None, rules=None, 
            task=None, label_map=None, feature_cols=None, label_csv=None, use_qat=True, qcfg=None, config=None):
    if task == 'classification':
        # NOTE:
        # set input_dim and num_classes based on the shape of X and the number of unique labels in label_map.
        # this are important for the MLP and Transformer model to know how many input features and output classes it has to handle.
        input_dim = X.shape[1] if X.ndim > 1 else 1
        if label_map is not None:
            num_classes = max(len(label_map), int(np.max(y)) + 1)
        else:
            num_classes = int(np.max(y)) + 1

        knn_tf = main_prediction.initialize_knn_transformer(
            vocab_size=1, task='classification', num_classes=num_classes, d_model=32, n_heads=4, n_features=None, 
            memory_capacity=10000,
            memory_k=5, memory_metric='euclidean')  

        loaded = main_prediction.load_memory_head_setup(memory_name) # This function is used to load the saved HNSW Memory inside your (activity_log.db) database, and automatically apply the Old Memory back to the knn-augmented Transformer.
        if not loaded and with_training:
            losses, accs, _ = main_prediction.knn_forward_inference(X, y,
            label_map=label_map, # Label map can be safely set to None during Training phase.
            training=True, train_epochs=10, task='classification', batch_size=64,
            train_mode='dynamic_backward', lr=0.1) # for Training kNN Transformer only

            # save memory right after training to your custom initialized database: (ex: to your data_storage.db)
            save_memory(db_path, memory_name, label_csv, X, y, task, label_map, feature_cols=feature_cols, config=config)
            main_prediction.save_memory_head_setup(memory_name) # This function saves the Per-Head memory setup inside KNNAugmentedTransformer class to the the same database (activity_log.db) so it can be retrieved as Retrievable Memory later.

    elif task == 'regression':
        input_dim = X.shape[1] if X.ndim > 1 else 1
        num_classes = y.shape[1]

        knn_tf = main_prediction.initialize_knn_transformer(
            vocab_size=1, task='regression', num_classes=num_classes, d_model=32, n_heads=4, n_features=len(feature_cols), 
            memory_capacity=10000,
            memory_k=5, memory_metric='euclidean')  

        loaded = main_prediction.load_memory_head_setup(memory_name) # This function is used to load the saved HNSW Memory inside your (activity_log.db) database, and automatically apply the Old Memory back to the knn-augmented Transformer.
        if not loaded and with_training:
            losses, accs, _ = main_prediction.knn_forward_inference(X, y,
            label_map=label_map, # Label map can be safely set to None during Training phase.
            training=True, train_epochs=10, task='regression', batch_size=64,
            train_mode='dynamic_backward', lr=0.1) # for Training kNN Transformer only

            # save memory right after training to your custom initialized database: (ex: to your data_storage.db)
            save_memory(db_path, memory_name, label_csv, X, y, task, label_map, feature_cols=feature_cols, config=config)
            main_prediction.save_memory_head_setup(memory_name) # This function saves the Per-Head memory setup inside KNNAugmentedTransformer class to the the same database (activity_log.db) so it can be retrieved as Retrievable Memory later.

    else:
        print('[>] Invalid task name')

def knn_train_run(db_path, X, y, task, pred_config=None, label_csv=None, feature_cols=None, 
        new_memory=None, with_training=True):
    memory_name, main_model, main_prediction, task, label_csv, label_map, qcfg, feature_cols = init_setup(db_path, X, y, memory_name=new_memory, csv=label_csv, new_task=task, new_config=pred_config, cols=feature_cols)

    knn_transformer_round(memory_name, main_prediction, db_path, X, y, with_training,
    task=task, label_map=label_map, feature_cols=feature_cols, label_csv=label_csv, use_qat=True, qcfg=qcfg, config=pred_config)

    return main_prediction, label_map

def knn_predict_run(main_prediction, X, y, label_map=None):
    transformer_probs, attn_weights, prediction = main_prediction.knn_forward_inference(X, y,
    label_map=label_map, # Label map can be inserted in here, or you Can set this to None,
                        # if set to None this function will only returns Transformer prob and attn_weights.
    task='classification', training=False)
