# START DOCKER CONTAINER
docker run --gpus all \
        -dit \
        -v ~/stash:/stash \
        --name pratipada\
        -p 3310:8888 \
        -p 3311:6006 \
	bhootmali/morl_eno:pratipada \
	screen -S jlab jupyter lab --no-browser --ip=0.0.0.0 --port 8888 --allow-root --LabApp.token=''
