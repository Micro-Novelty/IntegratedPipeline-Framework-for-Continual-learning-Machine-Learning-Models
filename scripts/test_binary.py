
from AbstractIntegratedModule import IntegratedPipeline
from AbstractIntegratedModule import PipelinePredictionManager
from AbstractIntegratedModule import Dense
from AbstractIntegratedModule import Transformer
from AbstractIntegratedModule import SecurityConfig
from AbstractIntegratedModule import PipelineAsyncManager
from AbstractIntegratedModule import SecurityLevel
from AbstractIntegratedModule import CohesiveAgentDeployment
from AbstractIntegratedModule import QuantConfig
import numpy as np
import pandas as pd
import asyncio
import traceback
from pathlib import Path

# pipeline_setup.py
from AbstractIntegratedModule import set_memory
from AbstractIntegratedModule import save_memory
from AbstractIntegratedModule import get_dataset
from sklearn.datasets import make_classification

# db_path is your custom initialized sqlite database name (ex: data_storage.db)
def init_setup(db_path, X, y, memory_name=None, csv=None, new_task=None, new_config=None, cols=None, port=None):
    # set_memory() is used to load the Model's Memory, csv fie for Training, task and label map using you previouslyr given X and y samples
    # new_memory argument is used to declare a new memory name if the searched memory is not present.\
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
    agent_port=5001, # this port is used to set AgentDistributedInference server (optional)
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
    if label_map is None and label_csv is not None:
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


# handle training round
def one_train_round(memory_name, main_model, db_path, X, y, with_training=False, titles=None, rules=None, 
                task=None, label_map=None, feature_cols=None, label_csv=None, use_qat=True, qcfg=None, config=None):
    # TRAINING BLOCK: This is where you train the MLP, Transformer, and LSTM models with the specified configurations you want.

    if task == 'classification':
        # NOTE:
        # set input_dim and num_classes based on the shape of X and the number of unique labels in label_map.
        # this are important for the MLP and Transformer model to know how many input features and output classes it has to handle.
        input_dim = X.shape[1] if X.ndim > 1 else 1
        num_classes = max(len(label_map), int(np.max(y)) + 1)

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


def predict(main_prediction, X, y, titles=None, rules=None, label_map=None):
    results, chosen_label, confidence = main_prediction.advanced_prediction_method(
    titles=None, label_map=label_map, rules=None, # titles and rules can be set to None (Optional samples), but label_map must NOT be None.
        X=X, y=y, # you could create your own X and y samples and put it here (Optional, y sample must already be one hot encoded first).
        top_k=3, batch_size=4)

    return results, chosen_label, confidence


def train_run(db_path, X, y, task, pred_config=None, label_csv=None, feature_cols=None, 
            new_memory=None, with_training=True, port=None):
    memory_name, main_model, main_prediction, task, label_csv, label_map, qcfg, feature_cols = init_setup(db_path, X, y, memory_name=new_memory, csv=label_csv, new_task=task, new_config=pred_config, cols=feature_cols, port=port)

    one_train_round(memory_name, main_model, db_path, X, y, with_training,
        task=task, label_map=label_map, feature_cols=feature_cols, label_csv=label_csv, use_qat=True, qcfg=qcfg, config=pred_config)
    return main_prediction, main_model
    

def one_run(db_path, X, y, task=None, pred_config=None, label_csv=None, feature_cols=None, 
            new_memory=None, with_training=True, port=None):
    memory_name, main_model, main_prediction, task, label_csv, label_map, qcfg, feature_cols = init_setup(db_path, X, y, memory_name=new_memory, csv=label_csv, new_task=task, new_config=pred_config, cols=feature_cols, port=port)
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

def knn_train_run(db_path, X, y, task, pred_config=None, label_csv=None, feature_cols=None, new_memory=None, with_training=True):
    memory_name, main_model, main_prediction, task, label_csv, label_map, qcfg, feature_cols = init_setup(db_path, X, y, memory_name=new_memory, csv=label_csv, new_task=task, new_config=pred_config, cols=feature_cols)
    
    knn_transformer_round(memory_name, main_prediction, db_path, X, y, with_training,
       task=task, label_map=label_map, feature_cols=feature_cols, label_csv=label_csv, use_qat=True, qcfg=qcfg, config=pred_config)

    return main_prediction, label_map

def knn_predict_run(main_prediction, X, y, label_map=None):
    transformer_probs, attn_weights, prediction = main_prediction.knn_forward_inference(X, y,
    label_map=label_map, # Label map can be inserted in here, or you Can set this to None,
                        # if set to None this function will only returns Transformer prob and attn_weights.
    task='classification', training=False)


# Load dataset (Change to Your own dataset here, just make sure to change the column names and features accordingly)
df = pd.read_csv("PJME_hourly.csv", parse_dates=["Datetime"], index_col="Datetime")
df = df.sort_index()
daily = df.resample("D").sum()


# calender features (day of week, month, is weekend) (suit to what your dataset actually has)
daily["dayofweek"] = daily.index.dayofweek     # 0=Senin, 6=Minggu
daily["month"] = daily.index.month
daily["is_weekend"] = (daily.index.dayofweek >= 5).astype(int)

# lag features (in days, not hours) (suit to what your dataset actually has)
daily["lag_1"] = daily["PJME_MW"].shift(1)      # yesterday (lag_1 will be used in feature_cols)
daily["lag_7"] = daily["PJME_MW"].shift(7)      # last week (same day) (lag_7 will be used in feature_cols)
daily["lag_14"] = daily["PJME_MW"].shift(14)    # 2 weeks ago (lag_14 will be used in feature_cols)

# Rolling statistics (suit to what your dataset actually has)
daily["rolling_mean_7"] = daily["PJME_MW"].rolling(7).mean() # (rolling_mean_7 will be used in feature_cols)
daily["rolling_std_7"] = daily["PJME_MW"].rolling(7).std()   # (rolling_std_7 will be used in feature_cols)

# delete the first row that contains NaN values due to lag and rolling calculations
daily = daily.dropna()

# Feature columns: a set of list thatcontains the features you want to use for training and prediction, 
# you can add or remove features based on your own dataset and requirements.
# so change this list to suit your dataset and requirements, but make sure the features you choose are actually present in your dataset.
feature_cols = ["PJME_MW", "dayofweek", "month", "is_weekend",
                "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_std_7"]
# example of other feature_cols:
# ["Global_active_power", "Global_reactive_power", "Voltage",
            #    "Global_intensity", "Sub_metering_1", "Sub_metering_2",
            #   "Sub_metering_3", "dow_sin", "dow_cos"]
# feature_cols = ["daily_sales", "is_promo", "is_holiday", "dow_sin", "dow_cos",
               #  "lag_7", "lag_14", "rolling_mean_7"]

X, y, config = get_dataset(daily, feature_cols, target_col_name="PJME_MW",
                                      window_size=30, horizon=8)

s_X, s_y = make_classification(
    n_samples=500,  # Number of data points
    n_features=5,  # Total features
    n_informative=3,  # Useful features
    n_redundant=2,  # Redundant linear features
    n_classes=2,  # Binary classification (0 or 1)
    random_state=42,  # For reproducibility
)

f_db_path = 'storage_holder.db'
mode = 'train_run'


if mode == 'one_run':
    results, chosen_label, confidence = one_run(f_db_path, s_X, s_y, task='classification', pred_config=None, 
                                            label_csv='ManualsTraining.txt', feature_cols=None)
    results, chosen_label, confidence = one_run(f_db_path, X, y, task='regression', pred_config=config, 
                                label_csv=None, feature_cols=feature_cols)
else:
    class_main, class_model = train_run(f_db_path, s_X, s_y, task='classification', pred_config=None, 
                                            label_csv='ManualsTraining.txt', feature_cols=None)
    reg_main, reg_model = train_run(f_db_path, X, y, task='regression', pred_config=config, 
                                label_csv=None, feature_cols=feature_cols, port=5008)
    
    sec_class_main, sec_class_model = train_run(f_db_path, s_X, s_y, task='classification', pred_config=None, 
                                            label_csv='ManualsTraining.txt', feature_cols=None, port=5009)
    sec_reg_main, sec_reg_model = train_run(f_db_path, X, y, task='regression', pred_config=config, 
                                label_csv=None, feature_cols=feature_cols, port=5007)

    results, chosen_label, confidence = one_run(f_db_path, s_X, s_y, with_training=False)
    results, chosen_label, confidence = one_run(f_db_path, X, y, with_training=False)
    


# ==== Use pipeline_setup.py continual learning workflow ======
 # = for classification:
 # use any X and y for classification (plus label map using label_csv)
main_prediction, label_map = knn_train_run(f_db_path, s_X, s_y, task='classification', pred_config=config, 
                                 label_csv="ManualsTraining.txt", feature_cols=None)
knn_predict_run(main_prediction, s_X, s_y, label_map=label_map)

 # for regression:
 # use the previous example of regression sample (PJME_MW Sample).
 # label_map is None for regression sample, labels for prediction will use prediction config              
main_prediction, _ = knn_train_run(f_db_path, X, y, task='regression', pred_config=config, 
                                 label_csv=None, feature_cols=feature_cols)
knn_predict_run(main_prediction, X, y, label_map=None)


test_titles = [
    ("Opening Thesis.docx", "slight_work"),
    ("Watching YouTube and Google Chrome", "distracted"),
    ("Watching Slack", "communication"),
    ("Programming in Visual Studio Code", "focused_work"),
    ("Watching netflix.com - Chrome", "break"),
    ]

rules = [
        # === WORK / PRODUCTIVITY ===
        (r'code|programming|develop|debug|compile|script', 'focused_work'),
        (r'vscode|visual_studio|ide|terminal|shell', 'focused_work'),
        (r'notion|evernote|onenote|notes|todo|task', 'productive'),
        (r'slack|teams|discord|zoom|meeting|call', 'communication'),
        (r'email|gmail|outlook|inbox|mail', 'communication'),

        # === ENTERTAINMENT ===
         (r'youtube|netflix|twitch|stream|video', 'entertainment'),
         (r'music|spotify|soundcloud|audio|player', 'entertainment'),
         (r'game|gaming|steam|epic|play', 'gaming'),
         (r'facebook|instagram|tiktok|social|post', 'social_media'),
         
         # === BROWSING ===
         (r'chrome|firefox|edge|safari|browser', 'browsing'),
         (r'google|search|wiki|wiki|article', 'information'),
         (r'stackoverflow|github|docs|documentation', 'research'),
    ]

# SECURITY BLOCK: This is where you set up the security configurations for the PipelineAsyncManager, this will allow you to control the access to the model and its predictions.
# change this api key into a more secure and complex key, this will be used for authentication when you want to do asynchronous predictions with the model.
api_key = 'my-ultra-safe-secret-key-for-authentication'

# max_text_length: the maximum length of the text you want to allow for prediction, in this case we use 10000 (changeable).
# max_queue_size: the maximum number of requests you want to allow in the queue, in this case we use 100 (changeable).
# rate_limit_requests: the maximum number of requests you want to allow per minute, in this case we use 60 (changeable).
# require_api_key: whether to require an API key for prediction or not, in this case we use True for safety (changeable).

config = SecurityConfig(
      max_text_length=10000,
      max_queue_size=100,
      rate_limit_requests=60,  # 60 per minute
      require_api_key=True,
      max_pending_tasks=50,
      request_timeout=30.0,

      # Start with no IP restrictions, add via admin API
      allowed_ips=[],
      blocklisted_ips=[],
      require_bootstrap_auth = False
   )

# NOTE:
# Asynchronous Prediction Block: This is where you set up the PipelineAsyncManager for asynchronous predictions, this will allow you to make predictions without blocking the main thread and allows you to handle multiple requests concurrently.
# pass the PipelinePredictionManager of yours (main_prediction in this case), and the IntegratedPipeline of yours (here its main_model), 
# the SecurityConfig as config, and the api_key as api_key, and the max_workers of 4, more worker, means more concurrent requests (changeable), 
# the task_timeout as 30 (changeable), and the max_retries as 3 (changeable).
sec_memory_name = 'agent-2'
sec_main_model = IntegratedPipeline(
   memory_name=sec_memory_name,  # memory name for the AI you already initialized
   use_async=True, # local asynchronous prediction is permitted, if not PipelineAsyncManager wont start asynchronous prediction.
   agent_port=5002, # this port is used to set AgentDistributedInference server (optional)
   ssl_cert_file=None, ssl_key_file=None,# provide your cert_file path or key_file path (optional)
   ssl_context=None, # used by the Agent server. (optional)
   client_ssl_context=None # used by the client. (optional)
   ) 


sec_main_prediction = PipelinePredictionManager(
   sec_main_model, # your initialized pipeline
   label_csv=None, 
   # your filename that contains the csv and contains the CSV format.
   # the Agent will automatically searched the nearby folder like: downloads, data, and desktop folder.
   target_title='window_title', label='label')


sec_main_model.initialize_mlp_model(titles=test_titles, rules=rules, label_map=label_map, input_dim=None, output_dim=None, hidden_layers=[12, 17, 35], use_qat=False, qcfg=None)

sec_main_model.initialize_transformer_model(titles=test_titles, rules=rules, label_map=label_map, vocab_size=1, num_classes=None, d_model=128, n_heads=4, n_features=None,
                                                attn_dropout=0.1, ffn_dropout=0.05, weight_decay=1e-4, task='classification')

sec_main_model.train_lstm(titles=test_titles, rules=rules, label_map=label_map, raw_X=s_X, raw_Y=s_y, n_train=0.8, n_display=5, train_epochs=10, lr=0.1)
sec_main_model.train_mlp_model(titles=test_titles, rules=rules, label_map=label_map, X=s_X, y=s_y, task='classification', train_epochs=500, momentum=0.9, lr=0.01)
sec_main_model.train_transformer(titles=test_titles, rules=rules, label_map=label_map, X=s_X, y=s_y,  task='classification', d_model=128, train_epochs=10, 
                                    mode='dynamic_backward', lr=0.5, batch_size=4)
sec_main_model.train_lstm(titles=test_titles, rules=rules, label_map=label_map, raw_X=s_X, raw_Y=s_y, n_train=0.8, n_display=5, train_epochs=10, lr=0.1)


wrapper = PipelineAsyncManager(sec_main_model, 
         sec_main_prediction, 
         config=config, 
         state_file=None, 
         security_level=SecurityLevel.PRODUCTION,
         api_key=api_key, 
         max_workers=4, 
         task_timeout=30, 
         max_retries=3 )

# NOTE:
# start the PipelineAsyncManager event loop, this will start the asynchronous prediction event loop and allow you to make predictions without blocking the main thread.
# set method to 'Transformer_included' if you want to use the Transformer model for prediction (Only one option, not changeable).
# set bootstrap_token to None if you don't want to use bootstrap authentication, otherwise set it to your own bootstrap token (changeable).
wrapper.start(method='Transformer_included', bootstrap_token=None)

# NOTE:
# Asynchronous Prediction Example: This is where you make an asynchronous prediction using the PipelineAsyncManager, this will allow you to make predictions without blocking the main thread and allows you to handle multiple requests concurrently.
# payloads: this is the dictionary of payloads you want to send to the PipelineAsyncManager for prediction, in this case we use the test_titles, label_map, rules, and use_transformer as True (changeable).
# pass X and y samples for prediction, in this case we use the X and y we created earlier (Not changeable and X and y samples are must be passed here).
# api_key can be passed or set to None, if you set require_api_key to True in the SecurityConfig, then you must pass the api_key here.
# chunk_size: this is the chunk size you want to use for the prediction, in this case we use 1000 because our sample is large, and we require larger chunks to save memory and prevent memory overhead (changeable).
# client_ip (optional): this is the client IP you want to use for the prediction, in this case we set it to None (changeable, client_ip is a string, not listsor dicts).
# max_predictions: this is the maximum number of predictions you want to allow for the model to predict on each samples you have provided, in this case we use 3 max predictions, after that the model will abort all predictions (changeable).
# stop_on_error: this is the boolean flag you want to use for the prediction, in this case we set it to False, meaning if its False, when the model encounters an error during prediction, it will continue to predict the next sample, set to True if you want the model to stop on the first error (changeable).
# overall_timeout: this is the overall timeout you want to use for the prediction, in this case we set it to 20000 seconds, meaning if the model takes longer than 20000 seconds to predict, it will abort the prediction (changeable).

# IMPORTANT NOTE: the async_predict variable will return all the prediction results, prediction success, failures, so you can directly see how the model predicts each batch of the given sample.

payloads = {'test_titles': test_titles, 'label_map': label_map, 'rules': rules, 'use_transformer': True}
async_predict = wrapper.prediction_method(payloads, X, y_samples=y, chunk_size=1000,
                 max_predictions=3, task="auto", timeout=1000,
                 api_key=api_key, client_ip=None, overall_timeout=20000,
                 stop_on_error=False)

print(f'[=] Results: {async_predict}')


# Peer-to-Peer (P2P) Deployment Block: This is where you set up the CohesiveAgentDeployment for peer-to-peer deployment, this will allow you to deploy the model on multiple agents and allows them to communicate with each other for distributed inference.

# NOTE:
# secret_key: this is the secret key you want to use for authentication between agents, in
# filename: this is the filename you want the model to access, in this case we use 'ManualsTraining.txt' (changeable).
# timeout: this is the timeout you want to use for the prediction, in this case we set it to 20000 seconds (changeable).
# agent_1_id: this is the agent id you want to use for the first agent, in this case we use 'ID-01' (changeable).
# agent_2_id: this is the agent id you want to use for the second agent, in this case we use 'ID-02' (changeable).
secret_key = 'my-ultra-safe-secret-key-for-authentication'
filename = 'ManualsTraining.txt'
agent_1_id = 'ID-01'
agent_2_id = 'ID-02'


# function must be 'async' to use the await syntaxes below:
async def agent_test(task='classification', x=None, y=None, titles=None, rules=None, label_map=None, main_prediction=None, main_model=None, sec_main_model=None):

   # NOTE:
   # shared_auth_token : the secret_key you have alread made
   # trusted networks is a list (must be list), that contain allowed IPs for networking.
   # pass IntegratedPipeline (here its main_model), and PipelinePredictionManager (here its main_prediction).
   # agent memory_name can be anything you want.
   # pass target_titles and label_name (matches the target_title you used during loading your label_map inside your .txt file.)

   # enable_peers : allow an agent to communicate with other agent.
   # security_level:
      # PRODUCTION: secure for P2P networking,
      # DEVELOPMENT: not-secure for P2P networking
   main_model.timeout = 20000
    
   # Agent 1 - Primary (Port 5555)
   agent1 = CohesiveAgentDeployment(
      pipeline=main_model,
      memory_name="agent_primary",
      filename=filename,
      target_title='window_title',
      label_name='label',
      security_level="DEVELOPMENT",
      enable_peers=True,
      trusted_networks=['127.0.0.1/32', '192.168.1.0/24'],
      peer_discovery_port=5555,
      shared_auth_token=secret_key,
      predict_manager=main_prediction
   )

   # Agent 2 - Secondary (Port 5556)
   # same integratedpipeline model and main_prediction model, or you can set a different/second IntegratedPipeline and PipelinePredictionManager here (Changeable).
   agent2 = CohesiveAgentDeployment(
      pipeline=sec_main_model,
      memory_name="agent_secondary",
      filename=filename,
      target_title='window_title',
      label_name='label',
      security_level="DEVELOPMENT",
      enable_peers=True,
      trusted_networks=['127.0.0.1/32', '192.168.1.0/24'],
      peer_discovery_port=5556,
      shared_auth_token=secret_key,
      predict_manager=main_prediction
   )

   try:
      # Start both agents
      print("\n🚀 Starting Agent 1...")
      await agent1.start()
      print("✅ Agent 1 started on port 5555")
      
      print("\n🚀 Starting Agent 2...")
      await agent2.start()
      print("✅ Agent 2 started on port 5556")
      
      # Give servers time to fully bind.
      await asyncio.sleep(2)
      
      # Get API keys
      api_key = agent1.get_api_key()
      print(f"\n🔑 Using API Key: {api_key[:20]}...")

      # first payload for agent-1:
      payloads_1 = {"test_titles": titles, "label_map": label_map, "rules": rules, "X":X, "y":y, "use_transformer": True, "agent_id": agent_1_id}
      # second payload for agent-2:
      payloads_2 = {"test_titles": titles, "label_map": label_map, "rules": rules, "X":X, "y":y, "use_transformer": True, "agent_id": agent_2_id}

      # NOTE:
      # daily: is the dataframe/dataset you have loaded, if its present meaning You can use it in P2P using this setup for regression prediction:
      if daily is not None and task == 'regression':
         regression_payload = {
               'context': daily,
               'feature_columns': feature_cols,
               'target_col_name': 'PJME_MW',
               'window_size': 30,
               'horizon': 1
         }
      else:
         # set to None for classification tasks.
         regression_payload = None

      # Make prediction with peer ensemble
      # Peer Connection will be ensured successfull during P2P search. 
      result = await agent1.multi_modal_peer_ensemble_prediction(
         payloads=payloads_1,
         api_key=api_key,
         method='advanced',
         disable_sync=True,
         regression_payload=regression_payload
      )

      result2 = await agent2.multi_modal_peer_ensemble_prediction(
         payloads=payloads_2,
         api_key=api_key,
         method='advanced',
         disable_sync=True,
         regression_payload=None
      ) 
      
      print(f"\n📊 Ensemble Result for Agent 1:")
      print(f"   Prediction: {result.get('prediction', 'N/A')}")
      print(f"   Confidence: {result.get('confidence', 0):.2%}")

      print(f"   Second Prediction: {result2.get('prediction', 'N/A')}")
      print(f"   Second Confidence: {result2.get('confidence', 0):.2%}")

      # Keep running briefly.
      print("\n⏳ Cluster stable. Waiting 30 seconds before shutdown...")
      await asyncio.sleep(30)
      agent1._peer_agent.stop_server()
      agent2._peer_agent.stop_server()
      
   except Exception as e:
      print(f"\n❌ Error in cluster: {e}")
      traceback.print_exc()

   # NOTE: shutdown is important for each agent. (IMPORTANT)
   print("\n🛑 Shutting down cluster...")
   await agent1.shutdown()
   await agent2.shutdown()
   print("✅ Cluster shutdown complete")


# Initiate agent_test
asyncio.run(agent_test(task='classification', x=s_X, y=s_y, titles=test_titles, rules=rules, label_map=label_map, 
                       main_prediction=class_main, main_model=class_model, sec_main_model=sec_class_model))
asyncio.run(agent_test(task='regression', x=X, y=y, main_prediction=reg_main, main_model=reg_model, sec_main_model=sec_reg_model))

# Step 2
from AbstractIntegratedModule import PipelineAsyncManager
from AbstractIntegratedModule import SecurityConfig
from AbstractIntegratedModule import SecurityLevel

# NOTE:
# All the three Models (MLP, Transformer, LSTM MUST be Initialized, Trained before initiating Asynchronous Prediction, the Initialization workflow can be seen above.)
# Replace the regular advanced_prediction_method() with this asynchronous setups:


print(" = TESTING ASYNCHRONOUS PREDICTION MANAGER = ")
# Set discovery secret (in production, use environment variable)
secret_key = 'my-ultra-safe-secret-key-for-authentication' # you can customize this key

# max_text_length: the maximum length of the text you want to allow for prediction, in this case we use 10000 (changeable).
# max_queue_size: the maximum number of requests you want to allow in the queue, in this case we use 100 (changeable).
# rate_limit_requests: the maximum number of requests you want to allow per minute, in this case we use 60 (changeable).
# require_api_key: whether to require an API key for prediction or not, in this case we use True for safety (changeable).
security_config = SecurityConfig(
      max_text_length=10000, # can be extended
      max_queue_size=100, # can be extended
      rate_limit_requests=60,  # 60 per minute
      require_api_key=True, #
      max_pending_tasks=50,
      request_timeout=60.0,

      # Start with no IP restrictions, you can add allowed IPs for asynchronous prediction externally, boothstrap_auth for better security
      allowed_ips=[],
      blocklisted_ips=[],
      require_bootstrap_auth = False # true for better security (Not recommended, cause less flexibility)
  )

# NOTE:
# Asynchronous Prediction Block: This is where you set up the PipelineAsyncManager for asynchronous predictions, this will allow you to make predictions without blocking the main thread and allows you to handle multiple requests concurrently.
# pass the PipelinePredictionManager of yours (main_prediction in this case), and the IntegratedPipeline of yours (here its main_model), 
# the SecurityConfig as config, and the api_key as api_key, and the max_workers of 4, more worker, means more concurrent requests (changeable), 
# the task_timeout as 30 (changeable), and the max_retries as 3 (changeable).
async_manager = PipelineAsyncManager(class_model, 
        class_main, # your previous initialized PipelinePredictionManager
        config=security_config, 
        state_file=None, # state file is used to load known security logs ex: ip used, ip blacklisted, etc.
        security_level=SecurityLevel.PRODUCTION, # production level security initiated
        api_key=secret_key,  #set secret key you initialized
        max_workers=4, # workers to initiate asynchronous tasks, more workers, lower memory usage.
        task_timeout=30, 
        max_retries=3) # retries after failure during prediction

async_manager.start(method='Transformer_included', bootstrap_token=None) # boothstrap token is optional for better security


# NOTE:
# Asynchronous Prediction Example: This is where you make an asynchronous prediction using the PipelineAsyncManager, this will allow you to make predictions without blocking the main thread and allows you to handle multiple requests concurrently.
# payloads: this is the dictionary of payloads you want to send to the PipelineAsyncManager for prediction, in this case we use the test_titles, label_map, rules, and use_transformer as True (changeable).
# pass X and y samples for prediction, in this case we use the X and y we created earlier (Not changeable and X and y samples are must be passed here).
# api_key can be passed or set to None, if you set require_api_key to True in the SecurityConfig, then you must pass the api_key here.
# chunk_size: this is the chunk size you want to use for the prediction, in this case we use 1000 because our sample is large, and we require larger chunks to save memory and prevent memory overhead (changeable).
# client_ip (optional): this is the client IP you want to use for the prediction, in this case we set it to None (changeable, client_ip is a string, not listsor dicts).
# max_predictions: this is the maximum number of predictions you want to allow for the model to predict on each samples you have provided, in this case we use 3 max predictions, after that the model will abort all predictions (changeable).
# stop_on_error: this is the boolean flag you want to use for the prediction, in this case we set it to False, meaning if its False, when the model encounters an error during prediction, it will continue to predict the next sample, set to True if you want the model to stop on the first error (changeable).
# overall_timeout: this is the overall timeout you want to use for the prediction, in this case we set it to 20000 seconds, meaning if the model takes longer than 20000 seconds to predict, it will abort the prediction (changeable).

# IMPORTANT NOTE: the async_predict variable will return all the prediction results, prediction success, failures, so you can directly see how the model predicts each batch of the given sample.

payloads = {'test_titles': test_titles, 'label_map': label_map, 'rules': rules, 'use_transformer': True}
async_predict = wrapper.prediction_method(payloads, s_X, y_samples=s_y, chunk_size=1000,
                 max_predictions=3, task="auto", timeout=1000,
                 api_key=api_key, client_ip=None, overall_timeout=20000,
                 stop_on_error=False)

print(f'[=] Results: {async_predict}')


# NOTE: This function below MUST require test_titles or titles, label_map and example rules:
print('[==] Initiating advanced batch prediction')
predicted_output = async_manager.advanced_batch_prediction(test_titles, label_map, rules, 
X=None, y=None, # provide your initialized X and y samples (Also Optional, can be set to None)
api_key=secret_key, client_ip=None) # you can add client_ip to provide a robust authentication paired with secret_key
# for better and faster advanced prediction when using titles and rules, consider using advanced batch prediction like in the above example


from AbstractIntegratedModule import train_snn_network
from AbstractIntegratedModule import snn_predict
from AbstractIntegratedModule import initialize_snn_network

# if your X samples (for regression and classification) has large values, consider re-scaling your X samples to lower range to [-1, 1] to prevent Exploding gradients.
# X = main_model._safe_to_2d_float(X) if your X samples has len(shape) > 2

# n_hidden and seed can be changed and flexible.
# X, y, and label map is a must samples to pass.
net, hidden_layer, utility = initialize_snn_network(X, y, sec_label_map, n_hidden=64, seed=42)

encoder, net, reports, warnings = train_snn_network(X=X, y=y, epochs=50,
      net=net, hidden_layer=hidden_layer, task='regression', # pass your net, hidden_layer and task (can be classification and regression)
      batch_size=32, n_steps=20, # n number of steps for Input encoding for Noise robustness, higher steps, higher robustness from Noise.
      n_hidden=64, lr=2e-3,  # pass hidden layers (n_hidden) for the SNN Network (n_hidden must match the previous initialized n_hidden), and learning rate for the SNN network.
      pretrain_steps=200, unsupervised_stdp=False # pretrain steps can be set to 0 if unsupervised_std is set to False (This is for STDP Unsupervised leearning for the SNN before BPTT Training in order to shape better Weights)
      )
# This function already includes diagnostic modules for the Whole SNN network to give you a Clearer picture of the SNN Condition before Prediction.
# Very High Dead neurons is a sign of Bad development, consider to Increase hidden layers or learning rate in order for the Model to improve.

# x must be a two dimensional sample, if X.ndim > 2, consider reshaping the X to be a 2D sample safely.
results = snn_predict(X=X, y=y, task='regression', label_map=sec_label_map, encoder=encoder, net=net)
# net is the SNNNetwork Class, and Encoder is PoissonEncoder, both are required for Prediction and must be passed in this function, consider saving the SNN net and the encoder using json for later use.
# Small Note:
# - This architecture is not guaranteed to work best for Classifying tabular datas, its good for time series event based activity classification.
# net is the SNNNetwork Class, and Encoder is PoissonEncoder, both are required for Prediction and must be passed in this function, consider saving the SNN net and the encoder using json for later use.
# Small Note:
# - This architecture is not guaranteed to work best for Classifying tabular datas, its good for time series event based activity classification.
