module load gnu7
module load nvidia/cuda/10.1

srun --account=MST107266 --gres=gpu:1 --cpus-per-task=4 python3 -u tools/test_net_few_shot.py \
                        --dataset fss_cell \
                        --load_ckpt Outputs/e2e_mask_rcnn_R-50-FPN_1x_1_5shot/Feb29-20-15-52_gn1110.twcc.ai/ckpt/model_5shot_epoch49_step23.pth \
						--g 1 --seen 2 --k 5  --a 5 --vis
"""
srun --account=MST107266 --gres=gpu:1 --cpus-per-task=4 python3 -u tools/test_net_few_shot.py \
                        --dataset fss_cell \
                        --load_ckpt Outputs/e2e_mask_rcnn_R-50-C4_1x_2/Feb20-02-34-56_gn1101.twcc.ai_step/ckpt/model_step9999.pth \
						--g 2 --seen 2 --a 5 \
                        #--vis

srun --account=MST107266 --gres=gpu:1 --cpus-per-task=4 python3 -u tools/test_net_few_shot.py \
                        --dataset fss_cell \
                        --load_ckpt Outputs/e2e_mask_rcnn_R-50-C4_1x_3/Feb20-04-53-36_gn1118.twcc.ai_step/ckpt/model_step9999.pth \
						--g 3 --seen 2 --a 5 \
                        #--vis

srun --account=MST107266 --gres=gpu:1 --cpus-per-task=4 python3 -u tools/test_net_few_shot.py \
                        --dataset fss_cell \
                        --load_ckpt Outputs/e2e_mask_rcnn_R-50-C4_1x_4/Feb20-07-24-28_gn1101.twcc.ai_step/ckpt/model_step9999.pth \
						--g 4 --seen 2 --a 5 \
                        #--vis
srun --account=MST107266 --gres=gpu:1 --cpus-per-task=4 python3 -u tools/test_net_few_shot.py \
                        --dataset fss_cell \
                        --load_ckpt Outputs/e2e_mask_rcnn_R-50-C4_1x_5/Feb20-09-59-16_gn1101.twcc.ai_step/ckpt/model_step9999.pth \
						--g 5 --seen 2 --a 5 \
                        #--vis
"""