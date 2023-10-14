# What is this
kaggle docker can use GPU  

worked on docker version 24.x.x higher

how to build and run
```bash
$ cd my_kaggle_docker
$ docker compose up -d
```

# acess jupyter notebook
acess here http://127.0.0.1:8888  
default jupyter notebook password is "kaggle"  
if u want to change password, look run.sh  

# Attach to a running container
In Visual Code, using attach to a running container.
https://code.visualstudio.com/docs/devcontainers/attach-container#_attach-to-a-docker-container

# setting default python
default python switch
```bash
$ alias python='/opt/conda/bin/python'
```

if u want to conda activate
```bash
$ source /opt/conda/bin/activate
```
