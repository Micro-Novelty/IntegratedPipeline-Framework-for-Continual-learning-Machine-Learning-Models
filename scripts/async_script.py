# Step 2
from AbstractIntegratedModule import PipelineAsyncManager
from AbstractIntegratedModule import SecurityConfig
from AbstractIntegratedModule import SecurityLevel

# NOTE:
# All the three Models (MLP, Transformer, LSTM MUST be Initialized and Trained before initiating Asynchronous Prediction, the Initialization workflow can be seen above.)
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
