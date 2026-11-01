"""Async runners for secure agent clusters."""

import asyncio
import time
import traceback
from .cohesive_deployment import CohesiveAgentDeployment


# ============ EXAMPLE: SECURE PEER-TO-PEER CLUSTER ============
async def run_secure_agent_cluster(pipeline,test_titles, label_map, rules, X=None, y=None, agent_id=None, filename=None, title_name=None, label_name=None, manager=None, 
                                   daily=None, feature_cols=None, target_col_name=None):
    """
    Run multiple agents that securely communicate.
    Stops retrying once connected successfully.
    """
    print("\n" + "="*60)
    print("=== SECURE PEER-TO-PEER CLUSTER ===")
    print("="*60)
    
    # Set discovery secret (in production, use environment variable)
    secret_key = 'my-ultra-safe-secret-key-for-authentication'

    # Agent 1 - Primary (Port 5555)
    agent1 = CohesiveAgentDeployment(
        pipeline=pipeline,
        memory_name="agent_primary",
        filename=filename,
        target_title=title_name,
        label_name=label_name,
        security_level="PRODUCTION",
        enable_peers=True,
        trusted_networks=['127.0.0.1/32', '192.168.1.0/24'],
        peer_discovery_port=5555,
        shared_auth_token=secret_key,
        predict_manager=manager
    )
    
    # Agent 2 - Secondary (Port 5556)
    agent2 = CohesiveAgentDeployment(
        pipeline=pipeline,
        memory_name="agent_secondary",
        filename=filename,
        target_title=title_name,
        label_name=label_name,
        security_level="PRODUCTION",
        enable_peers=True,
        trusted_networks=['127.0.0.1/32', '192.168.1.0/24'],
        peer_discovery_port=5556,
        shared_auth_token=secret_key,
        predict_manager=manager
    )
    
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
        
        texts = {"test_titles": test_titles, "label_map": label_map, "rules": rules, "X":X, "y":y, "use_transformer": True, "agent_id": agent_id}
        
        if daily is not None:
            payload = {
                'context': daily,
                'feature_columns': feature_cols,
                'target_col_name': target_col_name,
                'window_size': 30,
                'horizon': 1
            }
        else:
            payload = None

        # Make prediction with peer ensemble
        # Peer Connection will be ensured successful during P2P 
        result = await agent1.multi_modal_peer_ensemble_prediction(
            texts=texts,
            api_key=api_key,
            method='advanced',
            disable_sync=True,
            regression_payload=payload
        )    

        result2 = await agent2.multi_modal_peer_ensemble_prediction(
            texts=texts,
            api_key=api_key,
            method='advanced',
            disable_sync=True,
            regression_payload=payload
        )      
        
        print(f"\n📊 Ensemble Result for Agent 1:")
        print(f"   Prediction: {result.get('prediction', 'N/A')}")
        print(f"   Confidence: {result.get('confidence', 0):.2%}")

        print(f"   Second Prediction: {result2.get('prediction', 'N/A')}")
        print(f"   Second Confidence: {result2.get('confidence', 0):.2%}")

        # Keep running briefly
        print("\n⏳ Cluster stable. Waiting 30 seconds before shutdown...")
        await asyncio.sleep(30)
        agent1._peer_agent.stop_server()
        agent2._peer_agent.stop_server()
        
    except Exception as e:
        print(f"\n❌ Error in cluster: {e}")
        traceback.print_exc()
        
    print("\n🛑 Shutting down cluster...")
    await agent1.shutdown()
    await agent2.shutdown()
    print("✅ Cluster shutdown complete")


async def example_async_with_result_queue(pipeline, test_titles, label_map, rules, X=None, y=None,agent_id=None, filename=None, title_name=None, label_name=None):
    # Example using the proper result queue
    
    """Example: build a CohesiveAgentDeployment and run predictions through the async result queue."""
    agent = CohesiveAgentDeployment(
        pipeline=pipeline,
        memory_name="test_agent",
        filename=filename,
        target_title=title_name,
        label_name=label_name,
        security_level="DEVELOPMENT",
        enable_peers=False
    )
    
    await agent.start()
    
    api_key = agent.get_api_key()
    payloads = {"test_titles": test_titles, "label_map": label_map, "rules": rules, "X":X, "y":y, "use_transformer": True, "agent_id": agent_id}
    
    # Single async prediction
    print('[==] Single sync prediction: (using single text: "Opening Thesis.docx")')
    sync_result = agent.predict_sync(
        texts=payloads,
        api_key=api_key,
        client_ip="127.0.0.1",
        method='advanced'
    )

    print(f"[=] Sync Result: {sync_result}")


    print("[==] Single async prediction: (using single text: Opening Thesis.docx)")
    result = await agent.predict_async(
        texts=payloads,
        api_key=api_key,
        client_ip="127.0.0.1",
    )
    print(f"[=] True Async Result: {result.get('prediction')} ({result.get('confidence', 0):.2%})")
    
    # Batch async predictions (parallel!)
    print("\n[=] Batch async predictions (parallel):")
    texts = [
        "Watching YouTube",
        "Programming in VS Code",
        "Checking Slack messages",
        "Reading documentation",
        "Taking a break"
    ]
    
    start_time = time.time()
    results = await agent.predict_batch_async(texts, timeout=60, api_key=api_key)
    elapsed = time.time() - start_time
    
    for result in results:
        print(f"[=] '{result['text']}' → {result['prediction']} ({result['confidence']:.1%})")
    
    print(f"\n[=] Completed {len(texts)} predictions in {elapsed:.2f}s")
    
    # Get queue stats
    stats = agent.get_queue_stats()
    print(f"[=] Queue stats: {stats}")
    
    await agent.shutdown()
