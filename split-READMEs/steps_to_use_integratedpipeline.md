______________________________________________________________________________________________________________________
## [=] Step's for library in-depth Usage
0. Download via PIP:
   - Clone repository first:
     ```bash
      # Clone immediately for Windows and x86_64 only without prerequisites          
      git clone https://github.com/Micro-Novelty/IntegratedPipeline-Continous-Learning-AI-Agent-library-framework.git
      cd IntegratedPipeline-Continous-Learning-AI-Agent-library-framework     
      ```

   - Install the library via pip:
     ```bash
     pip install AbstractIntegratedModule #or
     python -m pip install AbstractIntegratedModule
     ```

1. Clone repository:

 - ```
   # prerequisites (for Raspberry pi OS Only)
   # Update system
   sudo apt-get update
   sudo apt-get upgrade -y

   # Install Python 3.x and development tools
   # Version 3.x means you can install python 3.10 to 3.13 only,
   # choose one version specified for your needs (3.10 or 3.11 or 3.12 ...).
   sudo apt-get install python3.x python3.x-dev python3.x-venv -y

   # Install additional build tools
   sudo apt-get install build-essential libatlas-base-dev libjasper-dev -y

   # Clone immediately for Windows and x86_64 only without prerequisites          
   git clone https://github.com/Micro-Novelty/IntegratedPipeline-Continous-Learning-AI-Agent-library-framework.git
   cd IntegratedPipeline-Continous-Learning-AI-Agent-library-framework     
   ```   
 2. Install System Dependencies (for x86_64 installation):
    ```
    # Ubuntu/Debian
    # example with Version 3.10:
    sudo apt-get update
    sudo apt-get install python3.10 python3.10-dev python3.10-venv

    # CentOS/RHEL
    sudo yum install python310 python310-devel

    # Fedora
    sudo dnf install python3.10 python3.10-devel
    ```
    
 3. Create a virtual environment:
     - ```
       # Create virtual environment (windows)
       # example with python 3.10
       python -m venv venv
       # Activate virtual environment
       venv\Scripts\activate
       
       # Create virtual environment (x86_64) (ARM64 / raspberry pi)
       python3.10 -m venv venv
       # Activate virtual environment
       source venv/bin/activate
       ```
         
   
 4. Verify Installation:
    - ```
      python -c "from AbstractIntegratedModule import IntegratedPipeline; print('✓ Installation successful!')"
      ```
      
 5. Run main.py for quick test of successful imports:
    - Download main.py in our repository code section: [main.py](main.py)
    - ```
      # run this for quick import test.
      python main.py
      ```
           
         
     
3. Create CSV file that contains training labels and titles (Optional for training using texts):
   -  Example format:
      ```txt
      window_title,label
      "Thesis.docx",focused_work
      "Microsoft Excel",work
      "YouTube -> Google Chrome",distracted
      "Slack",communication
      "VSCode", focused_work
      "netflix.com -> Google Chrome",break
      "Outlook",work
      "System Settings",system designing
      "GitHub",creating-and-editing-repo
      "README.md -> VS Code",focused_work
      "Amazon.com -> Chrome",personal-work
      ```
      Note = window_title is target_title and label is target_label, check step below to use it.
      ### NOTE:
      - There must be no space between ',' and words to prevent missing labels during label_map extraction.


4. A. Use IntegratedPipeline as in this example:
   -  ### [CLASSIFICATION TASK ONLY] :
   ```python
   from AbstractIntegratedModule import IntegratedPipeline
   from AbstractIntegratedModule import PipelinePredictionManager
   import numpy as np

   # SETUP BLOCK.
   # NOTE:
   # if you want the Model to continuously learn from every input, Consider changing the memory name to match the exact input where the Model has been trained on that specific input, This allows Flexibility while making the Model deterministic in behavior.
   memory_name = 'agent_memory'
   # example: 
   # if model is trained in Input B, change memory name to 'sec_agent_memory', or whatever memory name you want.
   
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
   
   main_model.autonomous = True
   # Autonomous mode makes the Model autonomous to prevent any EOF Error during Github CI workflow usage, or etc, this will prevent the Model from requesting input from you. 

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
   ```

   - ### [REGRESSION TASK ONLY] :
   ```python

   # Here we used a dataset specifically predicting how much Electricity is used (You can see and inspect this .csv in kagglehub):
   # the dataset MUST be in .csv format like in example below, The Framework doesnt accept .txt, or any other format (Safety restriction to avoid Data Scraping):

   # Load dataset (Change to Your own dataset here, just make sure to change the column names and features accordingly)
   df = pd.read_csv("PJME_hourly.csv", parse_dates=["Datetime"], index_col="Datetime")
   df = df.sort_index() #s sort index
   daily = df.resample("D").sum()


   # calender features (day of week, month, is weekend) (suit to what your dataset actually has)
   daily["dayofweek"] = daily.index.dayofweek     # 0=Monday, 6=Sunday
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

   main_model = IntegratedPipeline(
   memory_name=memory_name,  # memory name for the AI you already initialized
   use_async=True, # local asynchronous prediction is permitted, if not PipelineAsyncManager wont start asynchronous prediction.
   agent_port=5001, # this port is used to set AgentDistributedInference server (optional)
   ssl_cert_file=None, ssl_key_file=None,# provide your cert_file path or key_file path (optional)
   ssl_context=None, # used by the Agent server. (optional)
   client_ssl_context=None # used by the client. (optional)
   ) 


   main_prediction = PipelinePredictionManager(
      main_model, # your initialized pipeline
      label_csv='energy_training.txt', 
      # your filename that contains the csv and contains the CSV format.
      # the Agent will automatically searched the nearby folder like: downloads, data, and desktop folder.
      target_title='window_title', label='label')

   # Prepare dataset using prepare_dataset() method, this will create the X and y samples for training and prediction.
   # argument description: 
   # daily: the dataset you want to use for training and prediction, in this case we use the daily dataset/dataframe we created earlier.
   # feature_cols: the list of features you want to use for training and prediction, in this case we use the feature_cols list we created earlier.
   # target_col_name: the name of the target column you want to predict, in this case we use "PJME_MW" as the target column.
   # window_size: the number of past days you want to use for training and prediction, in this case we use 30 days.
   # horizon: the number of days you want to predict into the future, in this case we use 5 days.
   X, y, config = main_prediction.prepare_dataset(daily, feature_cols, target_col_name="PJME_MW",
                                       window_size=30, horizon=5)
   # NOTE:
   # the total horizon must match the total labels you have in your label_map, if your label_map is 5, you need to set the horizon=5.

   # activate explainability capability to explain uncertainty:
   main_model.show_explainability_details = True
   main_model.distribution.predict_manager = main_prediction # set PipelinePredictionManager to AgentDistributedInference for asynchronous prediction later (Very important for asynchronous prediction)
   # main_model.use_transformer = True if you want to use transformer, this will notify all modules that used advanced_prediction_method will initiate prediction with both transformer and MLP.

   main_model.autonomous = True
   # Autonomous mode makes the Model autonomous to prevent any EOF Error during Github CI workflow usage, or etc, this will prevent the Model from requesting input from you. 

   # set IntegratedPipeline Penalty rate when it output wrong answer:
   main_model.error_decay = 0.75
   main_model.use_lstm = True
   main_model.use_transformer = True
   # error_decay > 0.5 means old errors fade quickly — a class that was wrong 3 predictions ago matters less than one wrong just now, making the model less likely to output repetitive wrong answer.
   # this a flexible tunable-knob for the model recovery after error regarding wrong answers, this will propagate through prediction layers to inform about the model repetitive answer and the model calibrate its prediction immediately.


   # load your label_map from your .txt file that contains CSV format, this will be used for training and prediction.
   # window_title: the column name in your .txt file that contains the window title, 
   # label: the column name in your .txt file that contains the label.
   titles, _, label_map = main_prediction.load_labels_from_csv(
      'energy_training.txt',  # the name of your .txt file with CSV format.
      'window_title', 'label')

   # NOTE:
   # set input_dim and num_classes based on the shape of X and the number of unique labels in label_map.
   # this are important for the MLP and Transformer model to know how many input features and output classes it has to handle.
   input_dim = X.shape[1] if X.ndim > 1 else 1
   num_classes = max(len(label_map), int(np.max(y)) + 1)

   main_model.model_setup(mlp_train_epochs=1000, tf_train_epochs=10, tf_lr=0.1, mlp_lr=0.5,
   lstm_train_epochs=50, lstm_lr=5e-1, lstm_hidden_dim=64, tf_heads=4, tf_d_model=32,
   error_decay=0.75)

   # use_qat argument set to False because Quantization distorts regression samples.
   # MLP Initialization is important.
   main_model.initialize_mlp_model(titles=None, rules=None, label_map=label_map, input_dim=input_dim, output_dim=num_classes, hidden_layers=[12, 17, 35], use_qat=False, qcfg=None)

   # set task to regression
   # vocab_size MUST be 1,
   # n_features is passed with len(feature_cols) of your dataset
   main_model.initialize_transformer_model(titles=None, rules=None, label_map=label_map, vocab_size=1, num_classes=num_classes, d_model=128, n_heads=4, n_features=len(feature_cols),
         attn_dropout=0.1, ffn_dropout=0.05, weight_decay=1e-4, task='regression')
   
   # LSTM input, hidden sizes and output remains flexible in regression task.
   main_model.initialize_lstm_model(input_size=1, hidden_size=32, output_size=1, dropout_rate=0.1, seed=42)

   # Memory Loading Block: This is where you load the model memory if you have previously saved it, this will allow you to continue training or prediction from where you left off.
   # This function will automatically adds the loaded weights to the current model, so you can continue training or prediction without losing any progress.
   memory_loaded, lstm_weights, transformer_weights, mlp_weights, = main_model.load_model_memory()

   # EXAMPLE WORKFLOW
   if not memory_loaded:
      # MLP Training can be skipped
      main_model.train_mlp_model(titles=None, rules=None, label_map=label_map, X=X, y=y, task='regression', train_epochs=500, momentum=0.9, lr=0.01)

      # Use dynamic_backward for regression task for better learning capabilities.
      main_model.train_transformer(titles=None, rules=None, label_map=label_map, X=X, y=y,  task='regression', d_model=128, train_epochs=10, 
         mode='dynamic_backward', lr=0.5, batch_size=4)

      main_model.train_lstm(titles=None, rules=None, label_map=label_map, raw_X=X, raw_Y=y, n_train=0.8, n_display=5, train_epochs=10, lr=0.1)
   else:
      print('[>] Memory is loaded -> skipped Training')

   results, chosen_label, confidence = main_prediction.advanced_prediction_method(
   titles=None, label_map=label_map, rules=None, # titles and rules can be set to None (Optional samples), but label_map must NOT be None.
      X=X, y=y, # you could create your own X and y samples and put it here (Optional, y sample must already be one hot encoded first).
      top_k=3, batch_size=4)
   # NOTE:
   # MLP Prediction results will not be Showed in the Terminal, MLP Training can be skipped, but MLP MUST be initialized first to avoid any Errors.

   # save Weights as a memory to the SQLite database, this will allow you to load the model memory later for further prediction.
   main_model.save_weights(memory_name, model_type='Pipeline') 


  - Note:
    - Classification Task workflow script setup can be downloaded here: [classification_usage_script](scripts/classification_usage_scripts.py)
    - Regression task workflow example can be downloaded here: [regression_usage_script](scripts/regression_usage_scripts.py)

   
4. B. Using Standalone IntegratedPipeline Transformer:
   ```python
   # if you want to use the IntegratedPipeline Transformer only, you can use this setup:
   # ideal when you want blazing Fast Transformer Training and prediction, and the Transformer weights are loaded using JSON, this made Transformer Training much more feasible and easily intuitive for Beginners who wants Complex abstraction models that can run on cheap Edge devices.

   vocab_size = main_model.vocab_size# <your actual vocab size>.
   model_transformer = Transformer(
                  vocab_size=vocab_size
                  d_model=main_model.transformer_d_model,
                  n_heads=pipeline.transformer_heads,
                  num_classes=num_classes
                  n_features=None, # set to len(feature_cols) for regression.
                  task='classification'
              )
   # or just get your old Transformer:
   # model_transformer = main_model.model2

   sequence_inputs = main_model._features_to_sequence(X, d_model=pipeline.transformer_d_model) # Converts X samples into sequences that the Transformer can recognize.
   # embedded=True will ensure that Transformer forward method will put a correct mask for the samples.
   # mode='dynamic_backward' helps Transformer to gains better accuracy for Complex vocabularies.
   model_transformer.train(sequence_inputs, y, epochs=100, mode='dynamic_backward', lr=0.1, embedded=True, batch_size=2)

   # save Transformer weights
   tf = model_transformer
   json_data = {
        'token_embedding': tf.token_embedding,
        'pos_embedding': tf.pos_embedding,
        'W_q': tf.W_q,
        'W_k': tf.W_k,
        'W_v': tf.W_v,
        'W_q_fixed': tf.W_q_fixed,
        'W_k_fixed': tf.W_k_fixed,
        'W_v_fixed': tf.W_v_fixed,
        'W_o': tf.W_o,
        'ffn1': tf.ffn1,
        'ffn2': tf.ffn2,
        'ln1_scale': tf.ln1_scale,
        'ln1_shift': tf.ln1_shift,
        'ln2_scale': tf.ln2_scale,
        'ln2_shift': tf.ln2_shift,
        'output': tf.output,
        'output_bias': tf.output_bias
    }
    serializable_data = {key: val.tolist() for key, val in json_data.items()}
    with open("transformer_weights.json", "w") as file:
        json.dump(serializable_data, file, indent=4)  # indent adds readable formatting

    #load Transformer weights
    # in here transformer_model a new IntegratedPipeline Transformer instance:
    with open('transformer_weights.json', 'r') as file:
        loaded_data = json.load(file)
    transformer_model.W_q = loaded_data['token_embedding']
    ... # and so on until output_bias.
   ```
   - Note: script setup can be downloaded here: [transformer_usage](scripts/transformer_usage.py)
   
5. C. Use kNN-Augmented Transformer:
   Note: kNN-Augmented Transformer doesn't have its own Prediction pipeline flow, meaning after the Transformer forward method is called, the probabilities of the kNN Transformer will be returned and no prediction is made, We decided to be better this way because it grants Flexibility over how Prediction will be made in your favour:
   - Example Code:
   - ```python

      # Initialize kNN Transformer:
      knn_tf = sec_main_prediction.initialize_knn_transformer(
       vocab_size=1, task='<your_task>', num_classes=num_classes, d_model=32,
       n_heads=4, n_features=len(feature_cols), # pass len(feature_cols) if your task is regression, set to None if classification
       memory_capacity=10000,
       memory_k=5, memory_metric='euclidian')
     # NOTE:
     - vocab_size and num_classes is a MUST, meaning you have to pass the number of classes of your samples, and vocab_size of your data (can be set to vocab_size=1 for regression task)
     - memory_capacity = the amount of memory for the kNN to hold
     - memory_k = "K" is the central hyperparameter that defines the number of nearby training data points the algorithm looks at to make a prediction, more memory_k => more nearby training points the model can look up.
     
     
     # knn_forward_inference() lives inside PipelinePredictionManager, so you need to call this class to initiate the knn_forward_inference() function.
     losses, accs, _ = main_prediction.knn_forward_inference(X, y,
     label_map=None, # Label map can be safely set to None during Training phase.
     memory_metric='euclidean', training=True, batch_size=2,
     train_mode='dynamic_backward', lr=0.1) # for Training kNN Transformer only
     
     transformer_probs, attn_weights, prediction = main_prediction.knn_forward_inference(X, y,
     label_map=label_map, # Label map can be inserted in here, or you Can set this to None,
                            if set to None this function will only returns Transformer prob and attn_weights.
     memory_metric='euclidean',
     training=False) # for returning kNN Transformer probabilities and attention weights only.
     # Note: - train_mode can be set to 'dynamic_backward' if you have very large dataset, this makes Transformer Q, K, V be much more dynamic and grants flexible learning behavior for large dataset.
            # - train_mode can be set to 'fixed_backward' if you have small dataset, this makes Transformer Q, K, V to stay frozen so the FFN flow will handle the Training, making Learning in very little samples possible and deterministic in behavior.
            # - y sample must be one-hot encoded manually before its passed to the function, since the function above will not automatically one-hot encode the y-sample.

      # ==== SAVING MEMORY BLOCK =====
      main_prediction.pipeline.storage.save_hnsw_memory(memory_name) # This function saves the HNSW setup inside KNNAugmentedTransformer class to the the same database (activity_log.db) so it can be retrieved as Retrievable Memory later.
      main_prediction.pipeline.storage.save_memory_head(memory_name) # This function saves the Per-Head memory setup inside KNNAugmentedTransformer class to the the same database (activity_log.db) so it can be retrieved as Retrievable Memory later.

      # Get binary blob to save target memory:
      blob = main_model.model2.target_memory_to_bytes()
      main_model.storage.save_target_memory_row(memory_name, blob) # Save target memory row for kNN Transformer usage later


      # === LOADING MEMORY BLOCK ===
      main_prediction.pipeline.storage.load_hnsw_setup(memory_name) # This function is used to load the saved HNSW Memory inside your (activity_log.db) database, and automatically apply the Old Memory back to the knn-augmented Transformer.
      main_prediction.pipeline.storage.load_head_memory_setup(memory_name) # This function is used to load the saved HNSW Memory inside your (activity_log.db) database, and automatically apply the Old Memory back to the knn-augmented Transformer.
      main_model.storage.load_target_memory_row(memory_name) # load target memory row for kNN Transformer

     # .... # your own custom kNN prediction block.
     ```
      
     
7. To use IntegratedPipeline prediction without Transformer, Only Specialized MLP:
      Note: IntegratedPipeline without Transformer is'nt recommended due to it being weak at certain contextual prediction's, excel's at classification task's.
      - Example without transformer:
   ```python
   prediction_result = main_prediction.advanced_prediction_method( 
            [t[0] for t in test_titles],  # titles is enough for MLP Classification / or can be set to None.
            label_map,
            example_rules, # can be set to None
            X=X, y=y, # X, and Y samples can be passed here if titles and rules is None.
            top_k=3, batch_size=4
            )
   
   ```
8. Asynchronous prediction:
  - Asynchronous prediction request is important and is critical because it keeps prediction interfaces responsive, maximizes local hardware efficiency, and enables apps to handle background tasks seamlessly without waiting on remote server responses,
  - for asynchronous prediction handling, consider using this setup
```python
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
async_manager = PipelineAsyncManager(main_model, 
        main_prediction, # your previous initialized PipelinePredictionManager
        config=security_config, 
        state_file=None, # state file is used to load known security logs ex: ip used, ip blacklisted, etc.
        security_level=SecurityLevel.PRODUCTION, # production level security initiated
        api_key=secret_key  #set secret key you initialized
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
# test_titles and rules are can be set to None if not needed.
payloads = {'test_titles': test_titles, 'label_map': label_map, 'rules': rules, 'use_transformer': True}
async_predict = wrapper.prediction_method(payloads, X, y_samples=y, chunk_size=1000,
                 max_predictions=3, task="auto", timeout=1000,
                 api_key=api_key, client_ip=None, overall_timeout=20000,
                 stop_on_error=False)

print(f'[=] Results: {async_predict}')


# NOTE: This function below MUST require test_titles or titles, label_map and example rules:
print('[==] Initiating advanced batch prediction')
         predicted_output = async_manager.advanced_batch_prediction(test_titles, label_map, example_rules, 
         X=None, y=None, # provide your initialized X and y samples (Also Optional, can be set to None)
         secret_key=secret_key, client_ip=None) # you can add client_ip to provide a robust authentication paired with secret_key
# for better and faster advanced prediction when using titles and rules, consider using advanced batch prediction like in the above example

```
[=] Note:
 - Asynchronous prediction used Event loop that handles incoming request, There are conditions where event loop will not start and can't accet requests:
   - CPU Above > 95%    - Disk space is < 100 MB
   - RAM above > 95%
 - When event loop is not triggered, Asynchronous prediction can't be initiated and must be restarted/retried.
 - Script setup can be downloaded here: [async_script](scripts/async_script.py)

9. Peer-to-Peer Probability coordination:
   - Each peer is both server and client simultaneously for robustness and resilience during during P2P.
   - To Make the Agent cooperate with other peers, consider using this setup:
   - [=] for ensemble prediction from multiple peers, exchanging predicted label with each other, consider using this setup:
```python

# step 3

# NOTE:
# All three Models (MLP, Transformer and LSTM) must be Initialized and Trained first before initiating P2P.
from AbstractIntegratedModule import CohesiveAgentDeployment
from AbstractIntegratedModule import PipelinePredictionManager
import asyncio
import traceback

prediction_manager = PipelinePredictionManager(main_model, label_csv=<your_training_labels.txt>, target_title=<target_title>, label=<target_label>)

secondary_model = IntegratedPipeline(memory_name=memory_name, 
                use_async=True, agent_port=8080, 
                ssl_cert_file=cert_file, ssl_key_file=key_file) # provide cert_file path or key_file path (optional)
# secondary model of integrated pipeline is critical for ARM64 environment to prevent socket conflict during P2P with the first Integrated pipeline instance.
# make sure that the agent_port in secondary_model must be different from agent_port in first IntegratedPipeline instance you initialized to prevent address already in use error [Errno 48].

print("=== SECURE PEER-TO-PEER CLUSTER ===")

# CohesiveAgentDeployment is deeply tied and coupled with AgentDistributedInference,
# if you already set an SSL cert and key, CohesiveAgentDeployment will use the SSL directly from AgentDistributedInference
# allowing secure socket to be used directly by CohesiveAgentDeployment


# example task (change to regression if yours are regression task)
# function must be async to use the await syntax (Non-blocking)
async def p2p_tasks(task='classification')
   main_model.distribution.enable_ssl = False # set to false if you dont have SSL key and CERT, this code would instruct AgentDistributedInference that you don't have SSL, and provide you a regular unsecured socket (Not necessary for production)
   secondary_model.distribution.enable_ssl = False

   # NOTE:
      # shared_auth_token : the secret_key you have alread made
      # trusted networks is a list (must be list), that contain allowed IPs for networking.
      # pass IntegratedPipeline (here its main_model), and PipelinePredictionManager (here its main_prediction).
      # agent memory_name can be anything you want.
      # pass target_titles and label_name (matches the target_title you used during loading your label_map inside your .txt file.)

      # enable_peers : allow an agent to communicate with other agent.
      # security_level:
         # PRODUCTION: secure for P2P networking,
         # DEVELOPMENT: not-secure for P2P networking (use this for testing)
   # Agent 1 - Primary (Port 5555)
   agent1 = CohesiveAgentDeployment(
      pipeline=main_model, # main_model is your initialized integrated pipeline
      memory_name="agent_primary", # any name you want for the agent.
      filename=<filename>, # name of your .txt file that contains the CSV format and training labels
      target_title=<title_name>, 
      label_name=<label_name>,
      security_level="PRODUCTION", # production level security
      enable_peers=True, # allow peer discovery
      trusted_networks=['127.0.0.1/32', '192.168.1.0/24'], # for trusted networks, you need to provide the list of IPs of your peers.
      peer_discovery_port=5555, # peer port to start P2P
      shared_auth_token=secret_key, # your previous initialized secret_key
      predict_manager=prediction_manager, # your prediction manager
      peer_config = <'your_peer_ip_lists.json'> # you need to create .json file that contains your peer IP and Port lists
      consecutive_peer_config = <'your_second_fallback_peer_ip_lists.json'> # same for this one too, but for fallback.
      )
   
   # Agent 2 - Secondary (Port 5556)
   agent2 = CohesiveAgentDeployment(
      pipeline=secondary_model,
      memory_name="agent_secondary",
      filename=<filename>,
      target_title=<title_name>,
      label_name=<label_name>,
      security_level="PRODUCTION",
      enable_peers=True, # agent is allowed to find peers
      trusted_networks=['127.0.0.1/32', '192.168.1.0/24'],
      peer_discovery_port=5556,
      shared_auth_token=secret_key,
      predict_manager=prediction_manager,
      peer_config = <'your_peer_ip_lists.json'> # you need to create .json file that contains your peer IP and Port lists
      consecutive_peer_config = <'your_second_fallback_peer_ip_lists.json'> # same for this one too, but for fallback.
      )

   # Note: CohesiveAgentDeployment contains ConsecutivePeerAgent that can start a server once ensemble prediction from peer is started
   # be advised to stop the server too before shutdown-ing CohesiveAgentDeployment cluster

   # example peer_Ip_lists_config.json (de-comment to use)
   # {
         # you must put "known_peers" in the config so python can identify the list of IPs and Ports 
         #  "known_peers": [ 
         #    ["127.0.0.1", 5555], can be modified using real IP or local IP.
         #    ["127.0.0.1", 5556],
               # more ip and port lists...
      #   ]
      # }


   try:
      # Start both agents
      print("\n🚀 Starting Agent 1...")
      await agent1.start()
      print("✅ Agent 1 started on port 5555")
      
      print("\n🚀 Starting Agent 2...")
      await agent2.start()
      print("✅ Agent 2 started on port 5556")
      
      # Give servers time to fully bind
      await asyncio.sleep(2)
      
      # Get API keys
      api_key = agent1.get_api_key()
      print(f"\n🔑 Using API Key: {api_key[:20]}...")

      texts = {"test_titles": test_titles, "label_map": label_map, "rules": rules, # test_titles and rules are Optional samples, can be set to None.
   'X': None, "y":None, "use_transformer": True, "agent_id": agent_id} # (X and y are optional samples here too)
      # Important Note: If you set titles and rules to None, you must provide X and y samples for prediction, otherwise the models cant predict anything.

      # texts dictionary must contain test_titles, label_map, and rules that you can assign,
      # agent ID can be strings, int, or floats, recommendded to make it long for better security.
      agent1.pipeline.use_transformer = False # set agent1 pipeline to not use transformer for efficient processing in ARM64 env.
      agent2.pipeline.use_transformer = False
         
         # Make prediction with peer ensemble
         # Connection will be guaranteed successfull during discovery.
      result = await agent1.multi_modal_peer_ensemble_prediction(
            texts=texts,
            api_key=api_key,
            method='advanced',
            disable_sync=True
         )  # await using asyncio, multi_modal_peer_ensemble is already async by design (Inside ConsecutivePeerAgent), no need to put asyncio.run()

      # must have agent2 peer ensemble function after agent1 peer ensemble prediction function, 
      # so agent1 can receive agent2 peer ensemble prediction request.
      result2 = await agent2.multi_modal_peer_ensemble_prediction(
            texts=texts,
            api_key=api_key,
            method='advanced',
            disable_sync=True
         )

      print(f"\n📊 Ensemble Result for Agent 1:")
      print(f"   Prediction: {result.get('prediction', 'N/A')}")
      print(f"   Confidence: {result.get('confidence', 0):.2%}")

      print(f"\n📊 Ensemble Result for Agent 2:")
      print(f"   Prediction: {result2.get('prediction', 'N/A')}")
      print(f"   Confidence: {result2.get('confidence', 0):.2%}")

      # Keep running briefly
      print("\n⏳ Cluster stable. Waiting 5 seconds before shutdown...") # 5 seconds before shutdown.

      # stop ConsecutivePeerAgent servers inside CohesiveAgentDeployment.
      agent1._peer_agent.stop_server() # ._peer_agent is ConsecutivePeerAgent
      agent2._peer_agent.stop_server()

      await asyncio.sleep(5)

   except Exception as e:
      print(f"\n❌ Error in cluster: {e}")
      traceback.print_exc()
      
   finally:
      print("\n🛑 Shutting down cluster...")
      await agent1.shutdown()
      await agent2.shutdown()
      print("✅ Cluster shutdown complete")

# Initiate tasks:
if __name__ == '__main__':
   # match to your own task (classification or regression based on your samples)
   p2p_tasks(task='classification')

```
[=] Important Note: 
    - This setup outputs the given predicted label of an input directly, making P2P more flexible and fast.
    - This setup used Hybrid feature in prediction handling, Asynchronous prediction request, and Synchronous prediction handling. Synchronous prediction does block code execution for a few seconds, it was used for a few reason here, such as:
       - allowing a more slower traffic between agents, preventing other agent to get the same peer prediction over time, making each interaction equals and each peer can receive different peer prediction output.
       
[=] You can download this setup here for a direct test: [P2PDirectTest.py](P2P_Setups/P2PDirectTest.py)

   - [=] for probability coordination, locally, get peers data from database or via socket.
```python
# step 4
# X_generated is an X samples generated from using dataset converted into Vectors.
dataset, X_generated = main_model.data_preparation(test_titles, label_map)
# sequence inputs for Transformer
sequence_inputs = main_model.sequence_encoding(dataset)
# NOTE:
# if you have X samples:
# sequence_inputs = main_model._features_to_sequence(X)
# ignore the dataset version.


# type='Hybrid' = Transformer probabilities will be present for ensemble weighting with MLp probabilities
# type='others' = Transformer probabilities will be ignored, and ensemble weighting will not be initiated as MLP probs will be used.
peer_probability_calibration = main_model.predict_proba(sequence_inputs, X, type='Hybrid', embedded=True) # peer-to-peer calibration is inside this function
```
[~] Note: the peer calibration coordination has a chance of triggering if both MLP and Transformer prediction doesn't agree on certain output. Consider using this setup below for using stand-alone peer-to-peer main function without being wrapped in other parent function, allowing flexible and auditable peer-to-peer sharing for probability coordination:
```python
from AbstractIntegratedModule import WeightedEnsemblePredictor
from AbstractIntegratedModule import Transformer

# Initialize the ensemble helper class:
# main_model.distribution is the AgentDistributedInference for Handling samples:
ensemble_method = WeightedEnsemblePredictor(main_model, main_model.distribution, memory_name) # consider using the same memory name used in your previous pipeline


# Suit based on what samples you have, n_features can be set to None for classification, or used in regression samples.
# use your own vocab_size, in this case we use the old initialized vocab_size:
transformer = Transformer(
   vocab_size=main_model.vocab_size, d_model=32, n_heads=4, num_classes=num_classes, learning_rate=0.1, n_features=None, task='classification') # you can audit how much parameter the transformer needs.
main_model.model2 = transformer # overwrite previous transformer initialization in IntegratedPipeline

# main_model.distribution is AgentDistributedInference() class
# consider using ssl for secure peer to peer coordination
# main_model.distribution.ssl_cert_file = <path_to_your_ssl_cert_file> 
# main_model.distribution.ssl_key_file = <path_to_your_ssl_key_file>

# Create Sequence input that Transformer can Process:
sequence_input = main_model._features_to_sequence(X)
transformer_probs, attn_weights = transformer.forward(sequence_input)

# use your X and y samples here samples here
# set embedded to always True for Transformer forward method.
probs = ensemble_method.predict_ensemble(sequence_input, X, y, method='dynamic', embedded=True)
# 3 options for ensemble weighting method:
# 1. dynamic: allow flexible, efficient weighting from both transformer and MLP,
# 2. meta: for a much more in-depth weighting from both model,
# 3. calibration: allow calibrating probability for both model outputs based on both best weights assembling.

# Get Models Agreement from previous ensemble prediction:
agreement = main_model.agreement

# start server to initiate socket for P2P listener
main_model.distribution.start_server()

# set connection timeout (Optional)
main_model.distribution.connection_timeout = 150 # 150 seconds before timeout
calibrated_probability = main_model._handle_distributed_connections(probs, attn_weights, sequence_input, agreement)

# if an Agent experience a failure on tasks, consider using this function to reduce peer trust for safer flexible coordination:

# main_model.distribution.report_failure(id(main_model), '<task_name>', reason='<unknown>') # you can add the task_name and reason
# main_model.distribution.print_network_status() # to show other peers info.
```
[~] Note: this calibrated_probability is later used to calculate confidence and chosen output based on given label_map.
   - Consider checking:
     - [multi_agent_client.py](P2P_Setups/multi_agent_client.py) for a In-depth start for client testing.
     - [multi_agent_server.py](P2P_Setups/multi_agent_server.py) for a In-depth start for server testing.
   - If you get undefined NoneType Behavior when using .accept(), consider see [Troubleshooting](#Troubleshooting) Issue 7 for a Quick fix.

10. SNN Prediction Usage (Separated from IntegratedPipeline):
    - Example usage:
    - NOTE: (Our SNN cant handle Regression samples):
    - ### [CLASSIFICATION TASK AND REGRESSION TASK] :
    - ```python
      from AbstractIntegratedModule import train_snn_network
      from AbstractIntegratedModule import snn_predict
      from AbstractIntegratedModule import initialize_snn_network

      # if your X samples (for regression and classification) has large values, consider re-scaling your X samples to lower range to [-1, 1] to prevent Exploding gradients.
      # X = main_model._safe_to_2d_float(X) if your X samples has len(shape) > 2

      # n_hidden and seed can be changed and flexible.
      # X, y, and label map is a must samples to pass.
      net, hidden_layer, utility = initialize_snn_network(X, y, label_map, n_hidden=64, seed=42)
    
      encoder, net, reports, warnings = train_snn_network(X=X, y=y, epochs=50,
            net=net, hidden_layer=hidden_layer, task='<your_task>', # pass your net, hidden_layer and task (can be classification and regression)
            batch_size=32, n_steps=20, # n number of steps for Input encoding for Noise robustness, higher steps, higher robustness from Noise.
            n_hidden=64, lr=2e-3,  # pass hidden layers (n_hidden) for the SNN Network (n_hidden must match the previous initialized n_hidden), and learning rate for the SNN network.
            pretrain_steps=200, unsupervised_stdp=False # pretrain steps can be set to 0 if unsupervised_std is set to False (This is for STDP Unsupervised leearning for the SNN before BPTT Training in order to shape better Weights)
            )
      # This function already includes diagnostic modules for the Whole SNN network to give you a Clearer picture of the SNN Condition before Prediction.
      # Very High Dead neurons is a sign of Bad development, consider to Increase hidden layers or learning rate in order for the Model to improve.

      # x must be a two dimensional sample, if X.ndim > 2, consider reshaping the X to be a 2D sample safely.
      results = snn_predict(X=X, y=y, task='your_task', label_map=label_map, encoder=encoder, net=net)
      # net is the SNNNetwork Class, and Encoder is PoissonEncoder, both are required for Prediction and must be passed in this function, consider saving the SNN net and the encoder using json for later use.
      # Small Note:
      # - This architecture is not guaranteed to work best for Classifying tabular datas, its good for time series event based activity classification.
      ```
      
12. Cross-Session availability:
   - To use Cross-session avialability to transfer or import memory, consider using this setup:
     - ```python
       main_model._cross_session_availability() # cross session capability function
       ```

12. As an option, You can add more feature's directly to what it should predict, behave using rules you have given, Create a visual dashboard, create a distributed mesh of this agent, and much more features you can try.

________
