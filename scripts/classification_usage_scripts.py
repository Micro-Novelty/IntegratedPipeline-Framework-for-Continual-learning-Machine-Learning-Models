from AbstractIntegratedModule import IntegratedPipeline
from AbstractIntegratedModule import PipelinePredictionManager
import numpy as np

# SETUP BLOCK.
memory_name = 'agent_memory'
cert_file = <your_cert_file_dir> # your .crt file
key_file = <your_key_file_dir> # your .key file

# SSL Setup for users who used Lets encrypt / Public CA cert:
# Server context
server_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
server_ctx.load_cert_chain('your_server.crt', 'your_server.key') # No load_verify_locations needed — OS trust store handles public CA

# Client context  
client_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
client_ctx.load_cert_chain('your_client.crt', 'your_client.key') # no load_verify_locations — OS trust store handles it

# for enterprise and internal CA:
'''
server_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
server_ctx.load_cert_chain('their.crt', 'their.key')
server_ctx.load_verify_locations('company_ca.crt')  # internal CA not in OS store
server_ctx.verify_mode   = ssl.CERT_REQUIRED
server_ctx.check_hostname = False

client_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
client_ctx.load_cert_chain('their.crt', 'their.key')
client_ctx.load_verify_locations('company_ca.crt')
client_ctx.verify_mode   = ssl.CERT_REQUIRED
client_ctx.check_hostname = False
''' # uncomment to use

# Note: The Above setup is Important for Users who wants to do Secure P2P In deployment case, for local P2P the above setup is optional and automatic self-signed CERT will be used for local-device P2P.


main_model = IntegratedPipeline(
   memory_name=memory_name,  # memory name for the AI you already initialized
   use_async=True, # local asynchronous prediction is permitted, if not PipelineAsyncManager wont start asynchronous prediction.
   agent_port=5001, # this port is used to set AgentDistributedInference server (optional)
   ssl_cert_file=cert_file, ssl_key_file=key_file,# provide your cert_file path or key_file path (optional)
   ssl_context=server_ctx, # used by the Agent server. (optional)
   client_ssl_context=client_ctx # used by the client. (optional)
   ) 

main_prediction = PipelinePredictionManager(
   main_model, # your initialized pipeline
   label_csv='example_manual_training.txt', 
   # your filename that contains the .txt file and contains the CSV format.
   # the Agent will automatically searched the nearby folder like: downloads, data, and desktop folder.
   target_title='window_title', label='label')

# example_manual_training is a .txt file that contain csv format like above example.

# rules will be used to create automatic dataset for IntegratedPipeline.
example_rules = [
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

                     # more rules
                 ]

# Samples:
X = None # <your_x_samples>
y = None # <your_y_samples>


# activate explainability capability to explain uncertainty:
main_model.show_explainability_details = True
main_model.distribution.predict_manager = main_prediction # set PipelinePredictionManager to AgentDistributedInference for asynchronous prediction later (Very important for asynchronous prediction)
# main_model.use_transformer = True if you want to use transformer, this will notify all modules that used advanced_prediction_method will initiate prediction with both transformer and MLP.

# set IntegratedPipeline Penalty rate when it output wrong answer:
main_model.error_decay = 0.75
# error_rate > 0.5 means old errors fade quickly — a class that was wrong 3 predictions ago matters less than one wrong just now, making the model less likely to output repetitive wrong answer.
# this a flexible tunable-knob for the model judgement regarding wrong answer, this will propagate through prediction layers to inform about the model repetitive answer and calibrate it immediately.

# NOTE:
# (OPTIONAL) You can also define your own QuantConfig for quantization-aware training (QAT), 
# this is useful for reducing the model size and improving inference speed, 
# especially for deployment on edge devices.
qcfg = QuantConfig(enabled=True, w_bits=7, a_bits=7)

# Set up the main model with the specified configurations you wants, this will override the default configurations of the IntegratedPipeline, 
# you can change the parameters to suit your dataset and requirements.
# tf -> Transformer, lr -> learning rate.
main_model.model_setup(mlp_train_epochs=1000, tf_train_epochs=10, tf_lr=0.1, mlp_lr=0.5,
                     lstm_train_epochs=50, lstm_lr=5e-1, lstm_hidden_dim=64, tf_heads=4, tf_d_model=32,
                     error_decay=0.75)

# test samples with more sophisticated rules and more complex titles for prediction
# (title, intent)
test_titles = [
 ("Opening Thesis.docx", "slight_work"),
 ("Watching YouTube and Google Chrome", "distracted"),
 ("Watching Slack", "communication"),
 ("Programming in Visual Studio Code", "focused_work"),
 ("Watching netflix.com - Chrome", "break"),
# more titles 
 ]  

# Load label_map from your .txt file for Training and Prediction (Important)      
titles, _, label_map = main_prediction.load_labels_from_csv(
   <your_filename.>,  # the name of your .txt file with CSV format.
   <target_title>, <target_label>)

# NOTE:
# INITIALIZATION BLOCK: This is where you initialize the MLP, Transformer, and LSTM models with the specified configurations you want.

# MLP Initialization: This is where you initialize the MLP model with the specified configurations you want.
# titles: the titles you want to use for training and prediction, in this case we use None (Optional).
# rules: the rules you want to use for training and prediction, in this case we use None (Optional).
# label_map: the label map you want to use for training and prediction, in this case we use the label_map we created earlier.
# input_dim: the input dimension you want to use for creating MLP Dense layers, in this case we use the input_dim we created earlier (can be set to None if task is classification, or you can set input_dim=X.shape[-1]).
# output_dim: the output dimension you want to use for creating MLP Dense layers, in this case we use the num_classes we created earlier(can be set to None if your task is classification, if regression you must pass the output dim (ex: output_dim=num_classes))
# hidden_layers: the hidden layers you want to use for creating MLP Dense layers, The layers are free to change to how much you want. (changeable).
# use_qat: whether to use quantization-aware training (QAT) or not, in this case we use True (can be set to False).
# qcfg: the quantization configuration you want to use for QAT, in this case we use the qcfg we created earlier (can be set to None).

# NOTE: input_dim and output dim can be set to None if your samples are only titles, rules and label_map, but you can still set your hidden_layers however you want.
main_model.initialize_mlp_model(titles=titles, rules=rules, label_map=label_map, input_dim=None, output_dim=None, hidden_layers=[26, 64, 85], use_qat=True, qcfg=qcfg)

# Transformer Initialization: This is where you initialize the Transformer model with the specified configurations you want.
# vocab_size: the vocabulary size you want to use for the Transformer, set vocab_size to 1 if your task is regression, set to your own vocab size / set to None if your goal is classification (if set to None, this function below will automatically determine your vocab size from the given test_titles).
# num_classes: the number of classes you want to use for creating Transformer weights, in this case we use the num_classes we created earlier.
# d_model: the total dimension for creating Transformer layers, this layer is crucial for the Models learning capacity, in this case we use 128 (changeable).
# n_heads: the number of attention heads you want to use for creating Transformer layers, in this case we use 4 (Lower head means less computational power and complexity) (changeable).
# n_features: the number/length of features you have, if your task is classification, n_features MUST be None, if your task is regression, you MUST need to use the len(feature_cols) of your dataset and pass it as n_features=len(feature_cols) here.
# attn_dropout: the dropout rate you want to use for the Transformer layers, in this case we use 0.1 (changeable).
# ffn_dropout: the dropout rate you want to use for the Transformer feed-forward network, in this case we use 0.05 (changeable).
# weight_decay: the weight decay you want to use for the Transformer layers, in this case we use 1e-4 for smaller and more deterministic decay (changeableand useful for testing the model capability).
# task: the task you want to use for the Transformer model, set to 'regression' for regression task, or "classification" for classification task (changeable).
main_model.initialize_transformer_model(titles=test_titles, rules=rules, label_map=label_map, vocab_size=None, num_classes=None, d_model=128, n_heads=4, n_features=None,
attn_dropout=0.1, ffn_dropout=0.05, weight_decay=1e-4, task='classification')

# LSTM Initialization: This is where you initialize the LSTM model with the specified configurations you want.
# input_size: the input size you want to use for creating LSTM layers, in this case we use 1 (changeable and flexible, meaning you can set to whatever input size you need).
# hidden_size: the hidden size you want to use for creating LSTM layers, in this case we use 32 (changeable and flexible, meaning you can set to whatever hidden size you need).
# output_size: the output size you want to use for creating LSTM layers, in this case we use 1 (changeable and flexible, meaning you can set to whatever output size you need).
# dropout_rate: the dropout rate you want to use for the LSTM layers, in this case we use 0.1 (changeable and flexible, meaning you can set to whatever dropout rate you need).
# seed: the random seed you want to use for the LSTM layers, in this case we use 42 (changeable and flexible, meaning you can set to whatever seed you need).
main_model.initialize_lstm_model(input_size=1, hidden_size=32, output_size=1, dropout_rate=0.1, seed=42)


# Memory Loading Block: This is where you load the model memory if you have previously saved it, this will allow you to continue training or prediction from where you left off.
# This function will automatically adds the loaded weights to the current model, so you can continue training or prediction without losing any progress.
memory_loaded, lstm_weights, transformer_weights, mlp_weights, = main_model.load_model_memory()

# TRAINING BLOCK: This is where you train the MLP, Transformer, and LSTM models with the specified configurations you want.

# MLP Training: This is where you train the MLP model with the specified configurations you want.
# titles and rules are optional, you can set them to None if you don't want to use them for training, but label_map must NOT be None.
# X and y are the training samples you want to use for training the MLP model, in this case we use the X and y we created earlier.
# pass your task, train_epochs however you like.
# momentum: the momentum you want to use for the MLP optimizer, higher momentum leads to faster convergence/better learning, in this case we use 0.75 (changeable).
# learning rate (lr): the learning rate you want to use for the MLP optimizer, higher learning rate leads to faster convergence/better learning, in this case we use 0.5 (changeable).

# EXAMPLE WORKFLOW:
# Train when weight memory is not loaded (fresh start)
if not memory_loaded:
   main_model.train_mlp_model(titles=test_titles, rules=rules, label_map=label_map, X=X, y=y, task='classification', train_epochs=500, momentum=0.9, lr=0.01)

   # Transformer Training: This is where you train the Transformer model with the specified configurations you want.
   # titles and rules are optional, you can set them to None if you don't want to use them for training, but label_map must NOT be None.
   # X and y are the training samples you want to use for training the Transformer model, in this case we use the X and y we created earlier.
   # pass your task, train_epochs however you like.
   # mode: the training mode you want to use for the Transformer model : 
      # in this case we use 'dynamic_backward': use dynamic_backward for training if you have large samples,
      # use 'fixed_backward' for training if you have small samples (such as only titles and rules are present, or very small length of X samples (ex: <100)),
   # batch_size: the batch size you want to use for the Transformer training, in this case we use 2 (changeable, based on how many batch you want, smaller batch means more memory usage).
   # learning rate (lr): the learning rate you want to use for the Transformer optimizer, higher learning rate leads to faster convergence/better learning, in this case we use 0.5 (changeable).
   main_model.train_transformer(titles=test_titles, rules=rules, label_map=label_map, X=X, y=y,  task='classification', d_model=128, train_epochs=10, 
      mode='fixed_backward', lr=0.5, batch_size=2)

   # LSTM Training: This is where you train the LSTM model with the specified configurations you want.
   # titles and rules are optional, you can set them to None if you don't want to use them for training, but label_map must NOT be None.
   # raw_X and raw_Y are the training samples you want to use for training the LSTM model, in this case we use the X and y we created earlier.
   # n_display: display what the LSTM previously thoughts about each sample after training, this will inform the LSTM uncertainty, intervals.
   # n_train: the percentage of training samples you want to use for training the LSTM model, in this case we use 0.8 or 80% of Training samples is used (changeable).
   main_model.train_lstm(titles=test_titles, rules=rules, label_map=label_map, raw_X=X, raw_Y=y, n_train=0.8, n_display=5, train_epochs=10, lr=0.1)
else:
   print('[>] memory is loaded, skipping Training...)')
         

# NOTE:
# PREDICTION BLOCK:
# the below section called advanced_prediction_method(...) is a prediction method that will output a single answer based from numerous predictions and how frequent it was predicted by a single model (This is called majority-based answer choosing method), it will output a single answer from the label_map you have given as its final prediction.
# meaning advanced_prediction method is only used to predict an answer based on the given label_map and only output a single answer, the entire predictions the model has made was saved in 'results' variable below, so you can see how the model predicts each batch of the given sample.
# titles and rules are optional, label map must not be None, but both label_map, X, y, titles and rules cannot be all empty.
# Use case: - classification and both regression problems that requires a model to only output a single answer.
# top_k : meaning, the model will only prints 3 labels it has predicted in the terminal (This will be used when titles and rules is not None, otherwise this argument will not be used.)
# batch_size : the batch size you want to use for the prediction, in this case we use 2 (changeable, based on how many batch you want, smaller batch means more memory usage).
results, chosen_label, confidence = main_prediction.advanced_prediction_method(
   titles=test_titles, label_map=label_map, rules=rules, # titles and rules can be set to None (Optional samples), but label_map must NOT be None.
      X=X, y=y, # you could create your own X and y samples and put it here (Optional, y sample must already be one hot encoded first).
      top_k=3, batch_size=2)

# Important Note: If you set titles and rules to None, you must provide X and y samples for prediction, otherwise the models cant predict anything.
# batch size=2 is needed during transformer training for batching, if you have larger samples consider using batch_size > 8, for medium amount of samples (>10 -> <50 samples) consider using 2 or 4 batch_size.
# the returned 'results' here returns all the predicted labels over each batch inside the given sample, so you can get the full picture of How the models predicts label over each batch in X sample.

# save Weights as a memory when previous memory isn't present to the SQLite database, this will allow you to load the model memory later for further prediction.
# model_type must be 'Pipeline'.
if not memory_loaded:
   main_model.save_weights(memory_name, model_type='Pipeline') 


# SIMULATION BLOCK (OPTIONAL):
# This function below allows you to create A continuous Training simulations using K-fold split to evaluate MLP Training performance using your provided X and Y samples, This function helps to diagnose if Your samples are appropriate for MLP and make It generalize instead of severely Overfitting.
# This function will create a fresh instance of MLP every k-amount you have set (the k=5), and also prints a Confusion matrix that helps diagnose Your MLP Training Accuracy Performance.
# This function is great to help you diagnose Further degradation of MLP performance and also Its consistency, And also helps validate our AWE Method at how well it makes MLP Training performance Consistent Accross different Seeds for better Reliability performance.
main_model.evaluate_mlp_performance(X, y, label_map, k=5, seed=42)

# (OPTIONAL):
# This setup below would allow you to save the Accurate answer (if the model guessed a specific problem correct) directly to the database,
# without initiating prediction over the sample repeatedly.
input_ids = main_model.model2.cache['input_ids'] # input indices the transformer used as input (model2 is Transformer).
index = results.get('index') # predicted index where the Model have chosen the label.
confidence = results.get('confidence') # the model confidence over the predicted Output from advanced prediction method.
main_model.accurate_cache_lookup.add_verified(X, input_ids, chosen_label, confidence, index,
                  source='Correctly-Answered') # You can modify the the source with "Answered-Correctly" (fit to your needs)

# ... more features you can add
