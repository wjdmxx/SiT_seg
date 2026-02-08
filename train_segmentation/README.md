# 训练
python train.py --image_dir /path/to/trainval-image \
                --mask_dir /path/to/trainval-mask \
                --exp_name tn3k_unet
# OOD评测
python eval.py --ckpt_dir outputs/tn3k_unet \
               --image_dir /path/to/ood-image \
               --mask_dir /path/to/ood-mask

# 评测并输出20张可视化图像
python eval.py --ckpt_path /mnt/bit/liyuanxi/projects/DUSA_cell_segmentation/train_segmentation/outputs/tn3k_unet_mask_smooth/best_model.pth \
               --image_dir /mnt/bit/liyuanxi/datasets/DUSA_cell_seg/Thyroid/tn3k/test-image/ \
               --mask_dir /mnt/bit/liyuanxi/datasets/DUSA_cell_seg/Thyroid/tn3k/test-mask/ \
               --save_vis --num_vis 20


python eval.py --ckpt_path /mnt/bit/liyuanxi/projects/DUSA_cell_segmentation/train_segmentation/outputs/tn3k_unet_mask_smooth/best_model.pth \
               --image_dir /mnt/bit/liyuanxi/datasets/DUSA_cell_seg/Thyroid/DDTI/2_preprocessed_data/stage1/p_image/ \
               --mask_dir /mnt/bit/liyuanxi/datasets/DUSA_cell_seg/Thyroid/DDTI/2_preprocessed_data/stage1/p_mask/ \
               --save_vis --num_vis 20