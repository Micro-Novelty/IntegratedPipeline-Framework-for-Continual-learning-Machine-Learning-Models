# REGRESSION WORKFLOW WITH REGRESSION BASED SAMPLES/DATA


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

# activate explainability capability to explain uncertainty:
main_model.show_explainability_details = True
main_model.distribution.predict_manager = main_prediction # set PipelinePredictionManager to AgentDistributedInference for asynchronous prediction later (Very important for asynchronous prediction)
# main_model.use_transformer = True if you want to use transformer, this will notify all modules that used advanced_prediction_method will initiate prediction with both transformer and MLP.

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

