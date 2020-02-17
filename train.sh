CUDA_VISIBLE_DEVICES=0 python tools/train_net_step.py \
				   --dataset fss_cell --cfg configs/few_shot/e2e_mask_rcnn_R-50-C4_1x.yaml \
				   --use_tfboard \
                   --bs 1 --nw 8 \
				   --g 1 --seen 1
