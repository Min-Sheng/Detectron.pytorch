CUDA_VISIBLE_DEVICES=0 python tools/test_net_few_shot.py \
                        --dataset fss_cell --cfg configs/few_shot/e2e_mask_rcnn_R-50-C4_1x.yaml \
                        --load_ckpt Outputs3/e2e_mask_rcnn_R-50-C4_1x/Feb12-04-50-20_32b15552e77a_step/ckpt/model_step17999.pth \
                        #--vis
