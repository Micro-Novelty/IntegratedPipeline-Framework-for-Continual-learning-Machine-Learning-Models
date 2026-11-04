# NOTE:
# All three Models (MLP, Transformer and LSTM) must be Initialized and Trained first before initiating P2P.
# Consider using the previous pipeline_setup.py before Prediction for stateful continual learning behavior
from AbstractIntegratedModule import CohesiveAgentDeployment
from AbstractIntegratedModule import PipelinePredictionManager
from pipeline_setup import train_run
import asyncio
import traceback

# =========== P2P ====================
print("=== SECURE PEER-TO-PEER CLUSTER ===")

# CohesiveAgentDeployment is deeply tied and coupled with AgentDistributedInference,
# if you already set an SSL cert and key, CohesiveAgentDeployment will use the SSL directly from AgentDistributedInference
# allowing secure socket to be used directly by CohesiveAgentDeployment


# example task (change to regression if yours are regression task)
# function must be async to use the await syntax (Non-blocking)
async def p2p_tasks(X=None, y=None, titles=None, rules=None, task='classification', prediction_manager=None, secondary_model=None)
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
   X, y = '<your classification sample>' # NEED TO MODIFY

   # match to your own task (classification or regression based on your samples)
   # if you have X samples, you can use the previous prediction_manager loaded by train_run() for stateful continual learning behavior
   prediction_manager, main_model = train_run(db_path, X, y, task='classification', pred_config=None, 
              label_csv='ManualsTraining.txt', feature_cols=None, port=5000) # use port 5000
   # Loads a new secondary IntegratedPipeline instance (secondary prediction manager isnt needed) and train it
   # this new instance used a different port for P2P so it doesnt conflict with other instance
   _, secondary_model = train_run(db_path, X, y, task='classification', pred_config=None, 
              label_csv='ManualsTraining.txt', feature_cols=None, port=5001) # use port 5001

   # insert the IntegratedPipeline instance and PipelinePredictionManager for Prediction during P2P
   # insert secondary IntegratedPipeline model for p2p to make sure no same instance share the same port.
   p2p_tasks(X, y, titles=None, rules=None, task='classification', prediction_manager=prediction_manager, model=main_model, secondary_model=secondary_model)

   # We use the previous PJME_MW for example only (change with your actual sample) 
   # NEED TO MODIFY
   daily = ...
   feature_cols = ...
   X2, y2, config = secondary_manager.prepare_dataset(daily, feature_cols, target_col_name="PJME_MW",
                                    window_size=30, horizon=5)

   prediction_manager, main_model = train_run(db_path, X2, y2, task='regression', pred_config=config, # get your previous pred_config. 
              label_csv=None, feature_cols=len(feature_cols, port=5003) # use port 5003
   # Loads a new secondary IntegratedPipeline instance (secondary prediction manager isnt needed) and train it
   # this new instance used a different port for P2P so it doesnt conflict with other instance
   _, secondary_model = train_run(db_path, X2, y2, task='regression', pred_config=config, 
              label_csv=None, feature_cols=feature_cols, port=5004) # use port 5004

   p2p_tasks(X2, y2, titles=None, rules=None, task='regression', prediction_manager=prediction_manager, model=main_model, secondary_model=secondary_model)
