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



B. Transformer:
Transformers are the modern standard for AI, introduced in 2017 with the famous "Attention Is All You Need" paper.
- How it works: Instead of reading sequences linearly, Transformers read the entire sequence all at once. They utilize a self-attention mechanism, which calculates how much "attention" or weight every part of the sequence should give to every other part, understanding the global context instantly.
> - Use Cases: The backbone of Large Language Models (LLMs) like ChatGPT or BERT, making them perfect for machine translation, text generation, and summarization.
> - Limitations: Transformers require massive amounts of training data and computing power to work effectively.
> - Data Hungry when Analyzing Images.

