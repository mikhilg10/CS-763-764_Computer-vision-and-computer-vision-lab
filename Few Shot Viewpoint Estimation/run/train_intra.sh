gpu=0
save_dir="save_models/IntraDataset"

# base class training
CUDA_VISIBLE_DEVICES=$gpu python -W ignore training.py --setting IntraDataset \
--root_dir_train /data/ObjectNet3D --annot_train ObjectNet3D.txt \
--save_dir ${save_dir} \
--n_epoch 150 --novel --keypoint
