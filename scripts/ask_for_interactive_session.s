# request a gpu
srun -t1:30:00 --mem=16000 --gres=gpu:1 --pty /bin/bash

# request 4 cpu
srun -c4 -t2:00:00 --mem=16000 --pty /bin/bash