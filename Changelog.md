## Changelog
All notable changes to this project will be documented in this file.
### PyPi AbstractIntegratedModule New Version changelog:
- Updated: 03/11/2026
- Recent Changelogs [Will be reset each new +0.1.0 versions]:
     - v1.2.5 [MAJOR REWORKS]:
       - Reworks Advanced Prediction method to separate Training and prediction to provide User's control.
       - Reworks Regression Training for MLP, and LSTM.
       - Fixed numerous bugs that can appear in edge cases during Advanced prediction method.
       - New Quantization-Aware-Training method only for MLP to reduce Memory usage during Continuous Training.
       - Reworks P2P functions to handle regression Tasks.
       - Reworks Asynchronous function to predict chunked X samples instead of all X samples to prevent Memory overflow for edge devices.
       - Reworks Numerous important Functions inside IntegratedPipeline that handles number of classes, dataset creation from test titles.
       - Saving and Loading Weights as memory to SQLite database is now separated, meaning you can handle Continuous Training or stop Training when memory is present and not present.
     - v1.2.6:
        - Separate SNN initialization from train SNN function
        - added Regression based training and prediction for SNN
     - v1.2.7:
       - test_titles and rules is not a must required samples for Asynchronous prediction.
     - v1.2.8:
       - Fixed rigid logic in _handle_async method function where a missing test titles and rules stops the whole async process.
       - Fixed rigid logic where a crash can occur in advanced prediction if lstm_predictions results are None.
     - v1.2.9:
       - Change most [>] to [TRAIN] or [DATABASE] for better understanding of the mechanism for each print.
       - Added expected calibration error after MLP Validation scoring to help better understanding regarding model generalization.
     - v1.3.0:
       - Added new automatic Continual learning mechanism for memory persistence that user can now use their own custom Database to set and save the memory.
       - Fixed advanced prediction logic where the prediction crash due to shape error when calculating for ensemble needs.
     - v1.3.1:
       - Label map is now considered not used for regression samples and will be replaced by prediction config.
       - fixed multiple bugs in save and load memory for kNN Transformer.
       - removed useless functions for HNSW setups that are no longer in use.
       - added regression sample analysis for kNN Transformer.


# Binary-new Release Changelog (Only inside in this repository)
### [0.1.0] - 2026-05-05
[=] Added
Initial release of IntegratedPipeline
Memory-Augmented Neural Networks (MANN) support
Abstract Weight Encoder (AWE) implementation
Specialized MLP with noise robustness
Optimized Transformer with Alpha-based computing
Peer-to-Peer agent coordination
SQLite-based memory persistence
Multi-platform support (Windows, Linux, Raspberry Pi)

### [0.1.1] - 2026-05-08 
[=] Specific Bug fixed for P2P Robustness, in _handle_distributed_connections(), removing undefined variables that caused specific Failed connection.

### [0.1.2] - 2026-05-11
[=] Bug fixed:
- fix: conflicting roles in _handle_distributed-connections() in specific cases
- fix: fix check rate limit unbounded grow
- fix: added another authentication identification for P2P in handle client function
- fix connect_agent function contains undefined variables in last condition.

### [0.1.3] 2026-05-12
[=] Bug fixed:
- conflicting str and float during confidence handling in advanced prediction method

### [0.1.4] 2026-05-22
- More Robust advanced prediction than v 0.1.3
- Asynchronous Prediction capabilities, PipelineAsyncManager (Queue message handling, based architecture)
- New robust P2P Architecture
- CohesiveAgentDeployment (For asynchronous Prediction feature between peers, Hybrid based architecture that supports hybrid capabilities, synchronous prediction and asynchronous from ensemble prediction from peers)
- ConsecutivePeerAgent (Handles Asynchronous ensemble weighting from incoming peer, when CohesiveAgentDeployment is busy, or cant capture the prediction request message)

### [0.1.5] 2026-05-24 (Officially tested and validated in ARM64 Env.)
- More Robust asynchronous advanced prediction than v 0.1.4
- Updated P2P Architecture
- Ensuring singleton on IntegratedPipeline
- fixed fragility in feature parsing from json file
- Officially tested in ARM64 Environment using docker

### Note: rest of documented versions from [0.1.5] -> [0.2.4] got deleted because of unintended overwrite in this repository to delete all files.

### [0.2.4] 2026-06-14
- fixed undefined variables bug in calibrate probs
- fixed nan fragility in trust scoring in P2P.

### [0.2.5] 2026-06-14
- Added new dynamic gate for Transformer Fixed and dynamic switching condition.
- Added Capabilities for the Model to save the Transformer weights as binaries in local SQlite database.

### [0.2.6] 2026-06-15
- Transformer is now Optimized using Cython!
- Refined memory checks to reduce further corruption in IntegratedPipeline memory.
- Added safety guards for cosine similarity to prevent float cascading through the flow.

### [0.2.7] 2026-06-16
- New cache management system for Transformers Cache
- Optimized Transformer Training time with less Computational overhead.
- Optimized Transformer attention quality computing with more newer Conditional gate for quality computing.

### [0.2.8] 2026-06-17
- Added safety guards for Cosine similarity to prevent string arrays to cause further errors.
- Refined csv load labels function to accept path dir, so users can finnaly find their Training labels without much restrictions and errors.
- Fixed bug in advanced prediction block that caused lstm weights to be set None.

### [0.2.9] 2026-06-18
- Ensure proper shape handling before shape alignment in Transformer Training.
- Fixed Silent bugs that can cause single scalar to bypass shape conditions gate in training step for Transformer.
- Ensure Robustness in LSTM Training step and prevent division by zero.

### [0.3.0] 2026-06-19
- New architecture: AccuracyAnswerCache
   - Purpose: Storing correct answered problem for better reliability in Non-dynamic environment, triggered after the model answered correctly and initiate a new prediction, the model will choose the previous correct answer when samples match.
- Fixed edge case where shape mismatch bug happens during MLP Training.

### [0.3.1] 2026-06-20
- Fixed bug where mlp always return index 0 during prediction.
- Optimized processing batch in chunk function.
- fixed bug where num_classes doesnt match given labels.

### [0.3.2] 2026-06-21
- Added Rust specific compiled binary for handling complex weight matrices.
          - added specific optimized weight saving handling with rust optimization.
          - Adding Optimization and refinements, especially bug fixes in:
             - IntegratedPipeline lstm samples creating
             - IntegratedPipeline shape adaptation
             - IntegratedPipeline probability calibration
             - IntegratedPipeline auto generate labels text
             - IntegratedPipeline MLP samples generation.
  
### [=] Features
Continuous learning without catastrophic forgetting
Local AI orchestrator with SQLite database
Hybrid MLP + Transformer ensemble predictions
Distributed agent system with SSL/TLS security
Explainability module for predictions
Cross-session model automation
Asynchronous local prediction (Worker based Queue) and P2P Asynchronous feature (P2P using socket)

### [=] Supported Platforms
Windows x86_64
Linux x86_64
Raspberry Pi (ARM64)

### [=] Known Limitations
Raspberry Pi: installation takes 30+ minutes
Large dataset optimization (1M+ samples) pending
