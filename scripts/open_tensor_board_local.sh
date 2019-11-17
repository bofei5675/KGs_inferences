# run below command on terminal of the machine that actually runs the training

tensorboard --logdir=/scratch/bz1030/capstone/rl_experiment/run/model_2019-10-17_18-41-44

# if you are on your local machine, pay attention to the terminal output, use the localhost to access the web
# if you are on prince to train model, you need to open a tunnel ON YOUR MACHINE by the following command.

ssh -L 6006:prince.hpc.nyu.edu:6006 bz1030@prince.hpc.nyu.edu

# Then use port 6006 (where you want to access by your machine)to access the port 6006
# (where the tensorboard is listening ) of prince
