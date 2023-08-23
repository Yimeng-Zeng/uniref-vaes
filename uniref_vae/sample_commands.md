python train_flash.py --lr 0.0001 --batch_size 256 --k 1 --vae_type flash_bf16 --d_model 128 --dropout 0.05 --kl_factor 0.0001

python train_flash.py --lr 0.0001 --batch_size 256 --k 1 --vae_type flash_bf16 --d_model 128 --dropout 0.05 --kl_factor 0.001

python train_flash.py --lr 0.0002 --batch_size 256 --k 1 --vae_type flash_bf16 --d_model 128 --dropout 0.05 --kl_factor 0.0001

python train_flash.py --lr 0.0002 --batch_size 256 --k 1 --vae_type flash_bf16 --d_model 128 --dropout 0.05 --kl_factor 0.001


python train_flash.py --lr 0.0001 --batch_size 256 --k 1 --vae_type flash_bf16 --d_model 64 --dropout 0.05 --kl_factor 0.0001

python train_flash.py --lr 0.0001 --batch_size 256 --k 1 --vae_type flash_bf16 --d_model 32 --dropout 0.05 --kl_factor 0.0001




# shorter vae training

CUDA_VISIBLE_DEVICES=0 python train_flash.py --lr 0.0002 --batch_size 512 --k 1 --vae_type flash_bf16 --d_model 128 \
--dropout 0.05 --kl_factor 0.0001 --encoder_dim_feedforward 256 --decoder_dim_feedforward 256 --encoder_num_layers 5 \
--decoder_num_layers 5 

CUDA_VISIBLE_DEVICES=1 python train_flash.py --lr 0.0002 --batch_size 512 --k 1 --vae_type flash_bf16 --d_model 128 \
--dropout 0.05 --kl_factor 0.0001 --encoder_dim_feedforward 256 --decoder_dim_feedforward 256 --encoder_num_layers 6 \
--decoder_num_layers 6 