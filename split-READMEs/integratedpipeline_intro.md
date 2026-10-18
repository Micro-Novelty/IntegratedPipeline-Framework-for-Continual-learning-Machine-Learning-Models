# [=] AbstractIntegratedModule - Specialized Deep Learning Framework for Continual-learning Models.

### [INTRO] :
- Machine Learning Framework that Creates a Model that can Continuously learn, backed by augmented memory from SQLite Database to store Weights, in which each Agent memory is Auditable, Transparent, and you can Create different Memory for the Model to learn new Things, without overwriting the previous saved Trained weights from different Tasks, User can firstly Train the Model, and load the required Weights for that specific memory name for later use.
- The Framework's Models are handled by Numpy library, allowing The Models to be used efficiently from High-end embedded systems to regular consumed Based PCs or Laptops without GPU (GPU Acceleration will be considered in the Future).
----------------------------------

<img width="800" height="400" alt="Energy_Consumption_Training py - Desktop - Visual Studio Code 18_10_2026 12_57_26" src="https://github.com/user-attachments/assets/a9ea0b41-3761-4302-bccb-a3228dd77fd3" />
<img width="400" height="200" alt="WhatsApp Image 2026-10-18 at 12 47 50" src="https://github.com/user-attachments/assets/57d0a356-3f42-4b84-9b0f-2da1ecd06ec1" />
<img width="400" height="200" alt="WhatsApp Image 2026-10-18 at 13 07 38" src="https://github.com/user-attachments/assets/99d0d728-09a3-44e7-adcf-8443b21da8fe" />


-----------------------------------


### [~] Introductory Explanation:
- AbstractIntegratedModule or IntegratedPipeline in short, is a standalone Specialized Machine-Learning Framework for Non-LLM memory Augmented Local Agentic orchestrator - Specifically designed to provide Local Continuous Learning capability for any Autonomous Model as single Instance or Coordinatively that runs efficiently from consumer based machine to High-end embedded systems, where the Model Can directly and continously learn using SQlite database as External memory, with minimal and efficient compute, built-in automatic augmented memory handler, Secure Peer-To-Peer (Multi-Agent) Coordination with security layers as an option, And Explainability capability based on proof from in it's internal metrics, reducing Black-Box condition necessary for reliability. 
- AbstractIntegratedModule Contains specialized MLP using Its Own specialized geometric Weight encoding (AWE) that is used for Quantization-Aware-Training (QAT) for lower memory Usage and faster convergence during Training, Specialized efficient Transformer and LSTM (Long-short term memory) architecture for Scarce Data training/prediction with Alpha-based computation, specifically designed for low-amount samples environment or Messy environments.
  - IntegratedPipeline Best Use-Case:
    - Tabular Data Prediction: Finding patterns in structured spreadsheets or rows of numerical and categorical features.
    - Classification Tasks: Separating data into groups, such as spam detection, customer churn, or disease diagnosis.
    - Regression Tasks: Predicting continuous numerical values like house prices, sensors output, sales numbers, or weather metrics.

___________________________________________________________________________________________________________________
### [+] Framework's Purpose
  - This framework is intended for Engineers who is early in ML Engineering in order to provide a Clearer picture of How edge AI Should operate, Low memory management, and Also how it can be Made continuously learn using Pragmatic solution by using SQLite to save Weights to prevent continuous catasthropic forgetting, IntegratedPipeline source code is open-sourced and can be found in our github repository, Meaning you can modify the Source code however You liked it to be, and help Us Grow the first Community!

### Important Note:
- All of the files provided here can be found in the main Github repository, if You are seeing this on PyPi and wanted to try out the provided .py, .sh or Dockerfile scripts below, consider visiting the official github repository:
- link: https://github.com/Micro-Novelty/IntegratedPipeline-Specialized-Non-LLM-AI-Agent-Framework

____________________________________________________________________________________________________________________
### Library Short Description
- Development Stage on PyPi: 1.2.8 Official Release.
- Author and Maintainer: Micro-Novelty and EpsitronNet-bot.
- library Source-Code is Open-sourced with MIT License.
- Purpose: Specifically Designed for providing Non-LLM AI Agent Framework for edge Devices, Optimized for ARM64 architecture.
- Library installation: 
  ```bash
  pip install AbstractIntegratedModule
  python -m install AbstractIntegratedModule
  ```

### Github Link (for Visiting and cloning)
- https://github.com/Micro-Novelty/IntegratedPipeline-Specialized-Non-LLM-AI-Agent-Framework

- The library also includes precompiled binaries for:
 - aarch64 manylinux (accepts version 2.17+) architecture, accepts python version 3.10, 3.11 only.
 - aarch64 musllinux (accepts version 1.2+) architecture, accepts python version 3.10, 3.11 only.
 - Windows 64 bit architecture (python 3.10, 3.11, 3.12, 3.13 only)
 - single tarball file for Users who want to test the library up to Python version 3.12 or 3.13+.
 ____________________________________________________________________________________________________
- Library installation if you dont have aarch64 setup, you can download the correct wheel for your setup in this repository or by using pip:
  - ✨ use pip for downloading the correct wheels for your setup:
     - ```bash
       pip install abstractintegratedmodule --extra-index-url https://Micro-Novelty.github.io/abstract-modules/whl/ --break-system-packages
       # ensures proper installation by bypassing pip strict external download setup. 
       ```
       - Note: Using the above installation --extra-index method provides Python Wheels for:
       - x86_64 and aarch64 with manylinux (v. 2.17+) and musllinux (v. 1.2+) architecture.
       - macOS with aarch64 and x86_64 architecture. (v. 10.9+)
       - All of this Wheels provided in the Repository Only Provides Wheels for python with version 3.10 3.11 and 3.12.
       - This setup will automatically download the correct wheel Based on your python/pip version and OS / hardware architecture setup.

- For specific module in Rust for handling and loading Weights and Parsing JSON values with reduced memory lookup overhead.
 - Consider using this Optimization library by using:
    - ```bash
      pip install abstract-weights-core
      ```
    - Note: This Optimization would help AbstractIntegratedModule with faster JSON parsing and much more flexible database handling handled efficiently using Rust programming language.
   
- Proven Capabilities [Frameworks entire Journey]:
   - The library has been thoroughly first successfully tested in Multiple Environments from Windows to ARM64 QEMU Based Environment. The library is now Robust for Small usages for Toy projects.
   - Proven Works on ARM64 Environment using Github runners, Training and Prediction works efficient on Docker ARM64 environment with QEMU, good parallelizing behavior is guaranteed.
   - P2P Works efficiently in ARM64 Docker + QEMU, No conflicting socket and all prediction works efficiently.
   - AWE setup Proven Efficient on Hard-uncontrolled dataset such as Activity Recognition from the given Database.
   - LSTM is Optimized efficiently for scarce data with AWE method.
   - Robust Advanced prediction capabilities proven effective on ARM64 Using MLP + LSTM Architectures.
   - Transformer Optimized using Cython, reduced Memory overhead and Reduce CPU Usage, With Reduced Training Time is guaranteed.
   - Reworked Prediction and Training method for User convenience in controlling the Model's capabilities and reducing Blackbox-ness.
-----
  - Recent Changelogs [Will be reset each new +0.0.5 versions]:
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

    
     - Note: if you want to see the Changelog history of the library Older versions consider visiting this link:
       - PyPi history: https://pypi.org/project/AbstractIntegratedModule/#history
       
________________________________________________________________________________________________________________________
