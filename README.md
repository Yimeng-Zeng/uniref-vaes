# train-vae
Train VAE on sequence data (molecules, amino acid sequences, etc)


CUDA_VISIBLE_DEVICES=9 python3 train.py
CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.0005 --d_model 256  