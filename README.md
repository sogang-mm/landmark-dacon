# Dacon: Landmark classification

#### 
1. `git clone https://github.com/sogang-mm/landmark-dacon.git`
2. edit `docker-compose.yml`
    ```
    services:
      main:
        container_name: landmark
        ...
        ports:
          - "{host ssh}:22"
          - "{host tensorboard}:6006"
        ipc: host
        stdin_open: true
    ```

3. `docker-compose up -d`

4. Train  `main.py`
    ```bash
    #/workspace
    python main.py --ckpt_dir {ckpt directory} --epoch 100 --batch_size 64 --lr 0.001 --weight_decay 1e-5 -step 10 --gamma 0.8 
    ```
    Tensorboard
    ```bash
    # host에서 
    docker exec landmark tensorboard --logdir={ckpt directory} --bind_all
    ```
    
5. submit 
`{ckpt directory}/submission.csv`



 