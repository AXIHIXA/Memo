# Memo

## new ubuntu

    sudo apt install vim
    sudo apt install tmux
    
## .tmux.conf

    touch ~/.tmux.conf
    set -g mouse on
    set -g status-interval 60
    set -g display-time 3000
    set -g history-limit 65535

## anaconda virtual environment

    conda create -n py3 python=3.7 numpy scipy sympy matplotlib cython ipykernel
    source activate py3
    pip install opencv-python
    python -m ipykernel install --name py3 --user
    
## jupyter lab

### generate config

    jupyter notebook --generate-config
    jupyter notebook password
    
### edit `jupyter_notebook_config.py`    
    
    c.NotebookApp.ip = '0.0.0.0'         # 204
    c.NotebookApp.open_browser = False   # 267
    c.NotebookApp.password = 'sha1:...'  # 276
    c.NotebookApp.port = 9000            # 287
    
### view installed kernels

    jupyter kernelspec list
    
