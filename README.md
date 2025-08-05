# Setup (CPU)
Code using Python version 3.11.5
- Do git clone stuff as usual
- Highly suggested to create a virtual environment for project, do so by:
> python3.11 -m venv influence_env
- Now start your virtual environment (*You have to do this step everytime you want to run project*)
> source influence_env/bin/activate
- Install dependencies via requirements.txt (For CPU)
> pip install -r requirements.txt

# Setup (GPU, using conda)
Code using Python version 3.11.5
- Do git clone stuff as usual
- Highly suggested to create a virtual environment for project, do so by:
> conda create -n influence_env python=3.11 -y
- Now start your virtual environment (*You have to do this step everytime you want to run project*)
> conda activate influence_env
- Install PyTorch with CUDA
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
- [Optional] Verify that CUDA is working
> python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
- Install dependencies via requirements+cuda.txt (For GPU)
> pip install -r requirements+cuda.txt

# Run
- Go to project directory on your console (Terminal for mac)
> cd {Drag and Drop project folder onto Terminal}
- Run project virtual environment if you chose to set one up
> source influence_env/bin/activate
- Start notebook as usual
> python -m notebook
