# Proofs and documentation

### MANN Intro:
[>] Memory augmented Neural network (MANN) is a neural network architecture coupled with an external, dynamic memory module, allowing it to store, retrieve, and update information similarly to a computer's RAM. Unlike traditional networks that store knowledge only in weight parameters, MANNs excel at fast learning, long-term dependency handling, and episodic recall, In IntegratedPipeline, Its memory is stored in a custom database inside your local machine, then later used for memory retrieval, transfered to the Model Dictionary where it can finnaly recall its memory when input condition matched with memory. 

### Abstract Weight Encoder (AWE) Intro
[>] AWE is a specialized custom weight shaping or encoding method Specifically designed for MLP Architecture that excels in Quantization-Aware-Training method in MLP dense Layers, using eigenvalue and spectral methods as base equations to calculate covariance inside a given input data, and calibrate the correct Weight from the given eigenvalue, AWE Works by processing input and then captures the necessary eigenvalue to shape a properly initialized Weight that aligns with input data complexity, With this approach, MLP training will be much more consistent especially during Quantization-Aware-Training.

- For a much In-Depth Explanation You can visit This repository to learn more about AWE and its performance results:
- Link: https://github.com/Micro-Novelty/Specialized-MLP-for-noise-robustness
> - Note:
>  - The explanation itself inside the link repository is Outdated, since AWE has been improved over time.

### LSTM And Transformer Intro:
A. LSTMs (Long Short-Term Memory) and Transformers are foundational deep learning architectures built to process sequential data (like text or time series). While both handle the flow of time and context, they do so using completely different mechanisms.
- LSTM (Long Short-Term Memory):
  - LSTMs are an advanced class of Recurrent Neural Networks (RNNs) introduced to solve the problem of traditional RNNs forgetting earlier data.
  - How it works: LSTMs process data sequentially—one word or time-step at a time. They regulate information using "gates" (forget, input, and output) that determine what information from the sequence to keep or discard.
>  - Use Cases: Ideal for tasks with strictly continuous chronological data like time-series forecasting (e.g., stock market or weather prediction) or speech recognition.
>  - Limitations: Because they process data in a strict chain, it is difficult to parallelize training, making them slow and prone to forgetting long contexts.


B. Transformer:
Transformers are the modern standard for AI, introduced in 2017 with the famous "Attention Is All You Need" paper.
- How it works: Instead of reading sequences linearly, Transformers read the entire sequence all at once. They utilize a self-attention mechanism, which calculates how much "attention" or weight every part of the sequence should give to every other part, understanding the global context instantly.
> - Use Cases: The backbone of Large Language Models (LLMs) like ChatGPT or BERT, making them perfect for machine translation, text generation, and summarization.
> - Limitations: Transformers require massive amounts of training data and computing power to work effectively.
> - Data Hungry when Analyzing Images.


_______________________________________________________________________________________________
## Detailed process of Alpha-computing

<img width="720" height="338" alt="WhatsApp Image 2026-05-04 at 17 43 35" src="https://github.com/user-attachments/assets/3d149dce-cf3b-44c9-80b0-fa68290a2019" />

<img width="720" height="388" alt="WhatsApp Image 2026-05-04 at 17 44 04" src="https://github.com/user-attachments/assets/b1efedf6-5aa1-431e-89da-5f422549b453" />

_________________________________________________________________________________________________
🧠 What Alpha-Based Computation Actually Does

At its core, alpha (α) is a control parameter that blends two different information paths inside a given transformer:
A_final = α · A_fixed + (1 − α) · A_learned.
[>] Where: 
- A_fixed → stable, non-trainable (or minimally changing) attention
- A_learned → dynamic, trainable attention (Q, K, V)
- α ∈ [0, 1] → controls how much each path contributes
  
[>] Forward Pass -> Controlling Information Flow:
> During the forward pass, alpha determines what representation the model uses.
> If α is high (e.g., 0.8–1.0): Model relies mostly on stable attention
> → outputs are consistent → less noise → safer early training, If α is low (e.g., 0.0–0.3), Model relies mostly on learned attention, → more expressive → but unstable early on.
> So in forward propagation, alpha is essentially, “How much do I trust learned attention vs safe attention?”

[>] Backward Pass -> Controlling Gradient Flow:
When gradients flow backward:
```
- dA_final → splits into two paths
    - Mathematically:
        - dA_fixed   = α · dA_final
        - dA_learned = (1 − α) · dA_final
```

[>] Simple Explanation:
1. If α (alpha) is high: → most gradient goes to fixed path → learned attention gets very little update → training is stable but slow
2. If α is low: → most gradient goes to learned attention → fast learning → but noisy / unstable
   
[>] Why This Matters for Training
1. Without alpha: attention starts random → gradients noisy → model stuck (~10% accuracy)
2. With alpha: early stage -> rely on stable structure → meaningful gradients

[>] Benefits:
 - Transformer can receive small samples without destroying Accuracy.

________________________________________________________________________________________________________________________
## Performance in linux ARM64 Docker Environment/Container
A. [OLD DATA] Computational performance results during Training with Transformer included in regular classification tasks (Not regression).
```txt
== TIME == | CPU % | RAM / RAM LIMIT ||

14:50:08.762  0.00%  25.52MiB / 3.71GiB 
14:50:10.339  5.56%  25.52MiB / 3.71GiB
14:50:12.428  0.17%  25.52MiB / 3.71GiB
14:50:15.835 103.60% 38.8MiB / 3.71GiB
14:50:17.844 438.11% 60.15MiB / 3.71GiB
14:50:19.896 419.36% 68.32MiB / 3.71GiB
14:50:22.022 434.15% 73.2MiB  / 3.71GiB
14:50:23.969 436.47% 78.18MiB / 3.71GiB
14:50:25.966 435.85% 80.22MiB / 3.71GiB
14:50:28.018 430.47% 92.12MiB / 3.71GiB
14:50:30.029 450.55% 99.5MiB / 3.71GiB
14:50:32.065 438.06% 105.4MiB / 3.71GiB
14:50:34.100 469.52% 115.2MiB / 3.71GiB
14:50:36.109 449.89% 116.8MiB / 3.71GiB
14:50:38.123 423.56% 119MiB   / 3.71GiB
```
- [>] Note: To read the full logs (For included Transformer performance): [performance_log_with_TF](performance_logs/performance_log_with_TF.txt)
- [>] Explanation: 
  - Average CPU Usage = 420% - 430% , Sustained roughlt 4.20 - 4.30 CPU Cores
    - Meaning = - Compute-Intensive workload
                - reduced guaranteed of experiencing deadlocks, thread starvation, and major stalls
                - numerical backend is parallelizing efficiently
  - Average RAM Usage = 180 - 195 Mib, Roughly 4.5 - 5%  usage of available RAM
    - Meaning = - Very efficient RAM Usage
                - Repeated 269 MiB suggests allocator settled, ndarray pools stabilized, workload reached steady state
                - gradual, safe and steady batch accumulation, reduced possibility of leak explosion.

B. [>] Advanced Prediction without Transformer, Only Specialized MLP + LSTM using AWE during Training.
```txt
== TIME == | CPU %  | RAM \ RAM LIMIT |
10:01:37.456 100.44% 269.8MiB / 3.71GiB # initial round 1 training started
10:01:38.686 101.93% 279.8MiB / 3.71GiB
10:01:40.694 101.94% 293.2MiB / 3.71GiB
10:01:42.703 99.90%  302.9MiB / 3.71GiB
10:01:44.712 99.34%  319.8MiB / 3.71GiB
10:01:46.721 100.15% 319.7MiB / 3.71GiB
10:01:48.727 101.91% 319.7MiB / 3.71GiB
10:01:50.734 100.86% 319.7MiB / 3.71GiB
10:01:52.742 136.43% 322.9MiB / 3.71GiB
10:01:54.751 101.91% 324.9MiB / 3.71GiB
10:01:56.758 100.29% 324.9MiB / 3.71GiB
10:01:58.769 102.42% 327.1MiB / 3.71GiB
10:02:00.775 0.00%   329.6MiB / 3.71GiB # First Training and prediction round 1 finished
10:02:02.788 19.55%  330.1MiB / 3.71GiB # round 2 training
10:02:04.798 134.23% 332.2MiB / 3.71GiB
10:02:06.803 108.80% 332.2MiB / 3.71GiB
10:02:08.811 107.85% 332.2MiB / 3.71GiB
10:02:10.857 104.43% 332.2MiB / 3.71GiB
10:02:12.854 101.14% 332.2MiB / 3.71GiB
10:02:14.858 0.00%  332.3MiB / 3.71GiB
10:02:16.873 98.84% 333.8MiB / 3.71GiB
10:02:18.869 0.00%  132.7MiB / 3.71GiB # container stopped and round 2 finished
```
- [>] Note: See full performance log in here: [performance_log_only_MLP](performance_logs/performance_log_only_MLP.txt)
- [>] Explanation:
     - Average CPU Usage = 85-90% Used, sustained roughly 1 CPU Core saturated on average
         - Meaning: - Very efficient computation
                 - Much lighter CPU Usage than Transformer.
                 - Stable and less parallelizing behavior (Efficient computation)    
                 - 4.7x lower CPU usage than transformer included
      - Average RAM usage = 314 MiB / 3.71 GiB, Roughly 8.5% usage of available RAM.
         - Meaning: - Slightly larger MiB used than Transformer.
                 - Model allocation growth is stable,
                 - training buffers stable
                 - temporary ndarray growth.

### Both performance Overview
<img width="1536" height="1024" alt="WhatsApp Image 2026-05-24 at 10 27 00" src="https://github.com/user-attachments/assets/9404277f-281f-4893-8367-e494833230ea" />
_______________________________________
- Note: 
   - This performance metric was calculated in older versions, With newer Transformers that has more, newer modules, The resulting cpu consumed may Increase 1.5 - 2x much more in Docker ARM64 Environment + QEMU, 
   MLP And LSTM stays the same and still fully optimized.

### Introduction for demo
- Video Documentation: [![Introduction and demo:](https://youtube.com)](https://youtu.be/RmWvwDHU_QY?si=Lvl8mt8c_BnFypS_)
- Quick demo start: [main.py](main.py)
  - purpose: let you demonstrate the advanced prediction method and asynchronous prediction directly.
- Quick test of P2P:
    - [multi_agent_client.py](P2P_Setups/multi_agent_client.py)
    - [multi_agent_server.py](P2P_Setups/multi_agent_server.py)
    - [P2PDirectTest.py](P2P_Setups/P2PDirectTest.py)
    > - Purpose: let you demonstrate simple P2P using AbstractIntegratedModule quickly, there may be edge case bugs in this P2P setup so feel free to share it in issues.


_____________________________________________________________________________________________________________
## Main Components
[>] 1. - Consider checking and run: [IntegratedPipeline_Flow.html](IntegratedPipeline_Flow.html) regarding each function of the whole components and deep-dive mechanism.
    2. - consider checking [ARCHITECTURE.md](ARCHITECTURE.md) for more explanation about the main components.
       
[>] With 23 total architectures working together as a standalone library that is efficient and robust, Main components include:
    
1. GeometricWeightShaping
Purpose: Analyzes the geometric structure of data (anisotropy, spectral properties, complexity) and generates optimal weight matrices based on that geometry. Essentially teaches the model how to "understand" the shape of data before processing it, Highly robust to noise, making it an excellent fit for messy environment.
---
2. Activation
Purpose: Provides standard neural network activation functions (ReLU, sigmoid, softmax) and their derivatives for backpropagation for MLP Class.
---
3. Loss
Purpose: Implements categorical crossentropy loss and its gradient for training classification models for both MLP and The Transformer
---
4. Transformer 
Purpose: A complete transformer implementation with multi-head attention, positional embeddings, feed-forward networks, layer normalization, and custom backpropagation. Includes both fixed (stable) and dynamic (adaptive) training modes necessary for Scarce data environment using algorithm such as Alpha based computing directly during forward pass.
---
5. Dense and QATDense:
Purpose: A regular dense layer and a Dense layer that specifically handles Quantization-Aware method for Training that is flexible.
---
6. SoftmaxOutput  
Purpose: A simple wrapper around softmax activation that stores the output and passes gradients through unchanged (since softmax + crossentropy gradient is handled elsewhere).
---
7. MLP 
Purpose: A multi-layer perceptron that can switch between standard training and "focused" training (using feed-forward layers only) based on data complexity. Includes prediction, scoring, and geometric measurement methods.
---
8. WeightedEnsemblePredictor 
Purpose: Combines Transformer and MLP predictions using dynamic weighting based on attention quality, model confidence, and agreement. Also manages memory storage, explainability, and peer agent communication.
---
9. CrossSessionAutomation
Purpose: Manages exporting, importing, and syncing model sessions across different devices or time periods. Allows saving entire model states to JSON and transferring them over network sockets.
---
10. ExplainabilityModule 
Purpose: Generates human-readable explanations for predictions, learns from user feedback, maintains decision history, and batch-trains on corrections. The transparency layer for the AI agent.
---
11. ModelStorage 
Purpose: SQLite-based persistence for models, attention weights, node memories, and agent data. Handles serialization/deserialization of numpy arrays and model dictionaries.
---
12. AgentDistributedInference 
Purpose: The distributed agent system - can act as a server or client, handles SSL/TLS security, rate limiting, authentication, peer-to-peer prediction requests, memory synchronization, ensemble voting, and trust management between agents.
---
14. QueryNode  
Purpose: Manages trust relationships and identity verification between nodes. Evaluates node agreement, establishes connections, performs safety checks, and maintains the network of trusted peers.
---
14. AutoBatcherAutomation  
Purpose: Automatically batches incoming prediction requests to optimize throughput. Collects requests up to a maximum batch size or time window, then processes them together.
---
15. IntegratedPipeline
Purpose: The main orchestration class that ties everything together - handles text encoding, model initialization, training, prediction, memory management, hybrid predictions, and distributed inference coordination.
---
16. PipelinePredictionManager
Purpose: High-level prediction interface that loads labels from CSV, performs regular/advanced/hybrid predictions, displays results, and calculates entropy for uncertainty estimation.
---
17. ThreadedMessageQueue
purpose: Handles message queue in thread for asynchronous message handling. (Queue based)
---
18. AsyncMessageQueue
purpose: Handles incoming message from Asychronous peer request (Queue based)
---
19. CohesiveAgentDeployment
purpose: Handles both synchronous prediction, and P2P Asynchronous prediction request, acts as a server and a listener simultaneously.
---
20. ConsecutivePeerAgent
purpose: handles ensemble weighting from received output from peer, managed server and acts as a listener simultaneously.
---
21. LSTMCell
purpose: handles LSTM Core equations In Numpy.
---
22. LSTMNetwork
purpose: handling LSTM Feed-forward layer, Training and Samples handling.
---
23. LSTMEngine
purpose: handles Proper LSTM Confidence and calibration layer.
---

