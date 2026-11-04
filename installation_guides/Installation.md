## Requirements and Installations
[~] To run and execute IntegratedPipeline, Requirement's include Or see [Requirements](installation_guides/usage_needs.txt):
- Python 3.10 -> Python 3.14.
- pip install for quick usage:
- ```bash
  pip install AbstractIntegratedModule # or
  python -m pip install AbstractIntegratedModule
  ```
  
- NOTE: for Mac-OS or Linux x86-64 Users, consider downloading the correct python wheel for your setup using this command:
   - ```bash
     pip install abstractintegratedmodule --extra-index-url https://Micro-Novelty.github.io/abstract-modules/whl/ --break-system-packages
     ```
     Note: using this installation method grants you an up-to-date AbstractIntegratedModule library specifically for Mac-OS and Linux x86-64 users.
     
- X and y samples or titles and rules samples for Training and prediction (see Library in depth usage instructions below).

> - [OPTIONAL] For using AbstractIntegratedModule Binaries for users who doesnt have pip Installed:
>  - (Optional) Download AbstractIntegratedModule Compiled Binary file extension manually in the repository for Users who can't use pip install (Choose one minimal, specified for your needs):
>    - AbstractOptimizedModules for Optimizing Transformer using Cython (Optional)
>    - AbstractIntegratedModule (Main library)
>     - Compiled Binaries can be downloaded from release section in the Main repo or You can use this link:
>       - Release section: https://github.com/Micro-Novelty/IntegratedPipeline-Specialized-Non-LLM-AI-Agent-Framework/releases
>       - Supported OS: Mac-Os, linux x86-64 and Windows, bundled in zip folders.
>   - (Optional) Install AbstractIntegratedModule Compiled binaries from release section:
>   - (Optional) Libraries needed if pip is not installed (For using Compiled binary, not python wheels):
>     - Pandas
>     - aiohttp
>     - scikit-learn
>     - numpy
>     - psutil
>     - cryptography
     
    
[>] for labels assignation used for Training and prediction on Classification task only:
- CSV file that contains training labels and titles used for training and prediction label map, example will be provided below, Go to Step's for in-depth Usage below.
- [Go to Step's to use IntegratedPipeline](#steps-for-library-in-depth-usage)
- Note:
  - Regression sample will have its own custom prediction Config.
_______________________________________________________________________________________________________________________
### Requirements for Docker container
- Dockerfile (For Container assembler) 
- Python scripts (Such as main.py for Dockerfile usage).
- entrypoint.sh (for smart entry point for Dockerfile container usage).
  
## System-Specific Notes
1. Windows:
   - Requires Visual C++ Build Tools for compatibility
   - Use PowerShell or CMD (not WSL bash for best results)
     
2. Linux:
   - Ensure gcc and build-essential are installed
   - Different distributions may require different package managers
     
3. ARM64 - Raspberry Pi
   - Installation may take 30+ minutes due to ARM architecture
   - Monitor system resources during installation
   - Consider using faster storage (USB SSD) for better performance

## Docker Container Application
0. See [Docker_installation_Section](docker-setups/Docker_installation_Section.md) for an in-depth explanation, or [Quick_Docker_start](docker-setups/Quick_Docker_start.sh) for a quick start.
   - Note Consider checking:
     - [Dockerfile](docker-setups/Dockerfile) contains all the instructions need to assemble a Docker container.
     - [start.sh](docker-setups/start.sh) for Quick single agent start in Docker container.
     - [entrypoint.sh](docker-setups/entrypoint.sh) for a Smart entrypoint used in Dockerfile (Must be downloaded too along with Dockerfile and put in the same folder with Dockerfile after its downloaded)
     - [start-multi-agent-cluster.sh](P2P_Setups/start-multi-agent-cluster.sh) for Multi-agent cluster start in Docker container, What it does:
       - Starts a multi-agent Docker cluster — runs docker-compose up -d in detached mode (background), scaling the agent-client service to 5 simultaneous instances, alongside whatever server is defined in the docker-compose.yml.
       - Tails the logs — runs docker-compose logs -f which streams live logs from all containers (server + all 5 clients) to your terminal until you hit Ctrl+C.
     - [main.py](main.py) for executing a python script in the Docker container that used main.py, like in this code:
        - ```bash
          docker run -it -v $(pwd):/app/data integrated-pipeline:latest python /app/data/main.py
          ```
     - To use a [.dockerignore](docker-setups/.dockerignore) file, place it in your build context directory (the same location as your Dockerfile) to specify which files and folders should be excluded when building your image.
     - [.env](.env) is used for environment setup in [entrypoint.sh](docker-setups/entrypoint.sh), this file must be in the same folder with Dockerfile and entrypoint.sh after its downloaded.

1. Build Image:
   - Clone repository:
   - ```bash
     git clone https://github.com/Micro-Novelty/IntegratedPipeline-Continous-Learning-AI-Agent-library-framework.git
     cd IntegratedPipeline-Continous-Learning-AI-Agent-library-framework
     ```
   - Download:
   - [Dockerfile](docker-setups/Dockerfile),
   - [entrypoint.sh](docker-setups/entrypoint.sh)
   - [.env](docker-setups/.env)
   - In the code or release section.
     
     - If the downloaded Dockerfile or .dockerignore or .env has .txt extension, remove the extension:
     - ```bash
       # remove .txt extension
       mv Dockerfile.txt Dockerfile
       mv env.txt .env
       mv dockerignore.txt .dockerignore
       ```
   - Navigate to the folder: Use the cd command to enter the directory containing the Dockerfile and entrypoint.sh file.
   - ```
     cd /path/to/your/folder
     ```   
   - build image:
   - ```bash
     sudo docker build -t integrated-agent.
     ```
2. Download library dependencies for binaries usage:
   > - [Optional]: Required libraries (when using binary, not python wheel provided in the library):
   > - Numpy
   > - Scikit-learn
   > - pandas
   > - aiohttp
   > - psutil
     
4. Run IntegratedPipeline in a Container:
    - Install AbstractIntegratedModule via PIP or Compiled binaries:
    - ```bash
      pip install AbstractIntegratedModule # or
      python -m pip install AbstractIntegratedModule
      ```
      
    - ```bash
      docker run -it --name ai-agent integrated-agent:latest python
      ```
   - In python shell:
       - ```
         from AbstractIntegratedModule import IntegratedPipeline, PipelinePredictionManager
         model = IntegratedPipeline('agent_memory')
         print("✓ IntegratedPipeline initialized successfully!")
         ```
         
5. Run script:
   ```bash
   # Mount your local directory and run a script
   docker run -it -v $(pwd)/data:/app/data integrated-agent:latest python main.py # main.py could be replaced
   ```
     
6. For Single Agent and Multi-Agent P2P:
   [=] Single agent:
   ```bash
   # Build image
   docker build -t integrated-agent:latest .

   # Run single agent
   docker run -it -v $(pwd)/data:/app/data integrated-agent:latest python
   ```
   
   [=] Multi agent P2P (Consider docker-compose) :
   - Note: Use the provided [docker-compose.yml](docker-setups/docker-compose.yml) for Quick multi-agent.
   - Navigate to the folder: Use the cd command to enter the directory containing the docker-compose.yml file.
   - ```
     cd /path/to/your/folder
     ```
   - run this command to run docker-compose.yml:
   - ```bash
     # Start multiple agents
     docker-compose up -d

     # View logs
     docker-compose logs -f

     # Stop all agents
     docker-compose down
     ```
   
