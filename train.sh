module load gnu7
module load nvidia/cuda/10.1

srun --account=MST107266 --gres=gpu:2 --cpus-per-task=8 python3 -u tools/train_net.py \
                        --dataset fss_cell \
                        --lr_decay_epochs 20 30 40\
                        --use_tfboard --bs 64 --nw 8 \
						--g 1 --seen 1 --k 5\
                        --epochs 50

srun --account=MST107266 --gres=gpu:4 --cpus-per-task=16 python3 -u tools/train_net.py \
                        --dataset fss_cell \
                        --lr_decay_epochs 20 30 40\
                        --use_tfboard --bs 64 --nw 16 \
						--g 2 --seen 1 --k 5\
                        --epochs 50

srun --account=MST107266 --gres=gpu:4 --cpus-per-task=16 python3 -u tools/train_net.py \
                        --dataset fss_cell \
                        --lr_decay_epochs 20 30 40\
                        --use_tfboard --bs 64 --nw 16 \
						--g 3 --seen 1 --k 5\
                        --epochs 50

srun --account=MST107266 --gres=gpu:4 --cpus-per-task=16 python3 -u tools/train_net.py \
                        --dataset fss_cell \
                        --lr_decay_epochs 20 30 40\
                        --use_tfboard --bs 64 --nw 16 \
						--g 4 --seen 1 --k 5\
                        --epochs 50

srun --account=MST107266 --gres=gpu:4 --cpus-per-task=16 python3 -u tools/train_net.py \
                        --dataset fss_cell \
                        --lr_decay_epochs 20 30 40\
                        --use_tfboard --bs 64 --nw 16 \
						--g 5 --seen 1 --k 5\
                        --epochs 50

srun --account=MST107266 --gres=gpu:4 --cpus-per-task=16 python3 -u tools/train_net.py \
                        --dataset fss_cell \
                        --lr_decay_epochs 20 30 40\
                        --use_tfboard --bs 64 --nw 16 \
						--g 1 --seen 1 --k 1\
                        --epochs 50

srun --account=MST107266 --gres=gpu:4 --cpus-per-task=16 python3 -u tools/train_net.py \
                        --dataset fss_cell \
                        --lr_decay_epochs 20 30 40\
                        --use_tfboard --bs 64 --nw 16 \
						--g 2 --seen 1 --k 1\
                        --epochs 50

srun --account=MST107266 --gres=gpu:4 --cpus-per-task=16 python3 -u tools/train_net.py \
                        --dataset fss_cell \
                        --lr_decay_epochs 20 30 40\
                        --use_tfboard --bs 64 --nw 16 \
						--g 3 --seen 1 --k 1\
                        --epochs 50

srun --account=MST107266 --gres=gpu:4 --cpus-per-task=16 python3 -u tools/train_net.py \
                        --dataset fss_cell \
                        --lr_decay_epochs 20 30 40\
                        --use_tfboard --bs 64 --nw 32 \
						--g 4 --seen 1 --k 1\
                        --epochs 50

srun --account=MST107266 --gres=gpu:4 --cpus-per-task=16 python3 -u tools/train_net.py \
                        --dataset fss_cell \
                        --lr_decay_epochs 20 30 40\
                        --use_tfboard --bs 64 --nw 16 \
						--g 5 --seen 1 --k 1\
                        --epochs 50
