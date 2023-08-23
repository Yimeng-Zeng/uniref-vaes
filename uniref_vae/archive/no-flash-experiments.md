## This contains commands for running experiments on low precision with no flash attention using bf16/fp16, normally bf16 does not need gradient scaling while fp16 does:


1. bf16 without gradscaling

    ```python train_no_flash_torch113.py --lr 0.0001 --d_model 512 --batch_size 64 --k 1 --precision bf16 --gradscale False```

2. bf16 with gradscaling

    ```python train_no_flash_torch113.py --lr 0.0001 --d_model 512 --batch_size 64 --k 1 --precision bf16 --gradscale True```

3. fp16 without gradscaling

    ```python train_no_flash_torch113.py --lr 0.0001 --d_model 512 --batch_size 64 --k 1 --precision fp16 --gradscale False```

4. fp16 with gradscaling

    ```python train_no_flash_torch113.py --lr 0.0001 --d_model 512 --batch_size 64 --k 1 --precision fp16 --gradscale True```

5. fp32 baseline

    ```python train_no_flash_torch113.py --lr 0.0001 --d_model 512 --batch_size 64 --k 1 --precision fp32```