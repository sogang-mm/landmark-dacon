# Dacon: Landmark classification

#### 
1. `git clone `
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

```bash
#/workspace
$python main.py 
```