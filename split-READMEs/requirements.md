## [=] Requirements
[~] To run and execute IntegratedPipeline, Requirement's include Or see [Requirements](installation_guides/usage_needs.txt):
- Python 3.10 -> Python 3.14.
- pip install for quick usage:
- ```bash
  pip install AbstractIntegratedModule # or
  python -m pip install AbstractIntegratedModule
  ```
  
- NOTE: for Mac-OS or linux x86-64 Users, consider downloading the correct python wheel for your setup in here for a much flexible installation:
   - ```bash
     pip install abstractintegratedmodule --extra-index-url https://Micro-Novelty.github.io/abstract-modules/whl/ --break-system-packages
     ```
     Note: using this installation method grants you a much safer, predictable behavior, and more secure AbstractIntegratedModule library rather than a pure binary file.
     
- X and y samples or titles and rules samples for Training and prediction (see Library in depth usage instructions below).

- [OPTIONAL] For using AbstractIntegratedModule Binaries for users who doesnt have pip Installed:
  - NOTE:
  - The binaries inside the releases section might be Outdated in terms of version, so we advised to use the pip installation method.
  - (Optional) Download AbstractIntegratedModule Compiled Binary file extension manually in the repository for Users who can't use pip install (Choose one minimal, specified for your needs):
     - AbstractOptimizedModules for Optimizing Transformer using Cython (Optional)
     - AbstractIntegratedModule (Main library)
     - Compiled Binaries can be downloaded from release section in the Main repo or You can use this link:
       - Release section: https://github.com/Micro-Novelty/IntegratedPipeline-Specialized-Non-LLM-AI-Agent-Framework/releases
       - Supported OS: Mac-Os, linux x86-64 and Windows.   
   - (Optional) Install AbstractIntegratedModule Compiled binaries from release section:
   - (Optional) Libraries needed if pip is not installed (For using Compiled binary, not python wheels):
     - Pandas
     - aiohttp
     - scikit-learn
     - numpy
     - psutil
     - cryptography
     
    
[=] for labels assignation:
- CSV file that contains training labels and titles used for training and prediction label map, example will be provided below, Go to Step's for in-depth Usage below.
