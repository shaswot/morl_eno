# morl_eno
### Energy Neutral Operation using Multi-Objective Reinforcement Learning

Build docker image using
> docker build -f ./docker/Dockerfile -t <some_image_name:tag> .

Spin up a docker container
> Make appropriate changes to 
>> --name <whimsical_container_name>, 
>> -p <host_machine_jupyter_lab_port>:8888,
>> -p <host_machine_tensorboardb_port>:6006,
>> <some_image_name:tag>


and then run the script
> chmod +x start_docker_container.sh
> ./start_docker_container.sh

Your docker container should now contain a replica of this repository in /workspace
The /stash folder should be able to access the ~/stash folder in the host machine

