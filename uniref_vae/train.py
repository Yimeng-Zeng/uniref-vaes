import torch
from data import DataModuleKmers
import time 
import wandb 
import os 
os.environ["WANDB_SILENT"] = "true" 


def start_wandb(args_dict):
    import wandb 
    tracker = wandb.init(entity="yimengz", project='UNIREF-VAE', config=args_dict) 
    print('running', wandb.run.name) 
    return tracker 


def train(args_dict):
    print("training") 
    tracker = start_wandb(args_dict) 
    model_save_path = 'saved_models/' + wandb.run.name + '/' + wandb.run.name + '_model_state'  
    if not os.path.exists('saved_models/' + wandb.run.name + '/'):
        os.makedirs('saved_models/' + wandb.run.name + '/')
    datamodule = DataModuleKmers(args_dict["batch_size"], k=args_dict["k"], version=args_dict['data_version'] ) 

    if args_dict['debug']:
        print("Reducing to num points to debug")
        datamodule.train.data = datamodule.train.data[0: args_dict['num_debug']]
        print("now len data: ", len(datamodule.train.data))
        print('first point:', datamodule.train.data[0]) 
    
    tracker.log({'N train':len(datamodule.train.data)}) 

    if args_dict['vae_type'] == 'base':
        from transformer_vae_unbounded import InfoTransformerVAE
        print("using base vae")
    elif args_dict['vae_type'] == 'encoder_only':
        from encoder_only_vae import InfoTransformerVAE
        print("using encoder only vae")
    elif args_dict['vae_type'] == 'embed_only':
        from embed_only_vae import InfoTransformerVAE
        print("using embed only vae")

    model = InfoTransformerVAE(dataset=datamodule.train, d_model=args_dict['d_model'], kl_factor=args_dict['kl_factor'])
    print("model created with kl factor: ", args_dict['kl_factor'])

    if args_dict['load_ckpt']: 
        state_dict = torch.load(args_dict['load_ckpt']) # load state dict 
        model.load_state_dict(state_dict, strict=True) 
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    model = model.cuda()    
    optimizer = torch.optim.Adam([ {'params': model.parameters()} ], lr=args_dict['lr']) 
    lowest_loss = torch.inf 
    for epoch in range(args_dict['max_epochs']):
        start_time = time.time() 

        print("Starting training epoch: ", epoch)

        model = model.train()  
        sum_train_loss = 0.0 
        num_iters = 0
        for data in train_loader:
            optimizer.zero_grad() 
            input = data.cuda() 
            out_dict = model(input) 
            train_dict = {'train_' + k:out_dict[k] for k in out_dict.keys() }
            tracker.log(train_dict) 
            loss = out_dict['loss'] 
            sum_train_loss += loss.item()  
            num_iters += 1
            loss.backward()
            optimizer.step() 
        avg_train_loss = sum_train_loss/num_iters
        tracker.log({'time for train epoch':time.time() - start_time,
                    'avg_train_loss_per_epoch':avg_train_loss,
                    'epochs completed':epoch+1 }) 
        
        print("Training epoch: ", epoch, " completed.")
        print("Time for epoch: ", time.time() - start_time)

        if epoch % args_dict['compute_val_freq'] == 0: 
            start_time = time.time() 
            model = model.eval()  
            sum_val_loss = 0.0 
            num_val_iters = 0
            for data in val_loader:
                input = data.cuda() 
                out_dict = model(input)
                sum_val_loss += out_dict['loss'].item() 
                num_val_iters += 1 
                val_dict = {'val_' + k:out_dict[k] for k in out_dict.keys() }
                tracker.log(val_dict) 
            tracker.log({'time for val epoch':time.time() - start_time})
            avg_val_loss = sum_val_loss/num_val_iters 
            tracker.log({'avg_val_loss':avg_val_loss, 'epochs completed':epoch+1}) 

            if avg_val_loss < lowest_loss: 
                lowest_loss = avg_val_loss 
                tracker.log({'lowest avg val loss': lowest_loss, 
                                    'saved model at end epoch': epoch+1 }) 
                torch.save(model.state_dict(), model_save_path + '_epoch_' + str(epoch+1) + '.pkl')


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--debug', type=bool, default=False)  
    parser.add_argument('--lr', type=float, default=0.0001)  
    parser.add_argument('--compute_val_freq', type=int, default=3 )  
    parser.add_argument('--load_ckpt', default="" )  
    parser.add_argument('--max_epochs', type=int, default=100_000 )  
    parser.add_argument('--num_debug', type=int, default=100 )  
    parser.add_argument('--batch_size', type=int, default=128 )  
    parser.add_argument('--k', type=int, default=1 )  
    parser.add_argument('--kl_factor', type=float, default=0.0001 )
    parser.add_argument('--vae_type', type=str, default='base')
    parser.add_argument('--data_version', type=int, default=1 ) 
    parser.add_argument('--d_model', type=int, default=128 )
    args = parser.parse_args() 

    args_dict = {} 
    args_dict['d_model'] = args.d_model
    args_dict['batch_size'] = args.batch_size 
    args_dict['k'] = args.k 
    args_dict['lr'] = args.lr  
    args_dict['debug'] = args.debug  
    args_dict['compute_val_freq'] = args.compute_val_freq  
    args_dict['load_ckpt'] = args.load_ckpt
    args_dict['max_epochs'] = args.max_epochs 
    args_dict['num_debug'] = args.num_debug 
    args_dict['data_version'] = args.data_version 
    args_dict['kl_factor'] = args.kl_factor
    args_dict['vae_type'] = args.vae_type
    train(args_dict) 

    # CUDA_VISIBLE_DEVICES=4 python3 train.py --debug True --num_debug 1000 --lr 0.001 --compute_val_freq 50 
    # 0,1,2 

    # cd protein-BO/utils/oas_heavy_ighg_vae/
    # conda activate lolbo_mols
    # CUDA_VISIBLE_DEVICES=9 python3 train.py --lr 0.001 --compute_val_freq 50 --k 3 --d_model 128 
    # CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.0005 --d_model 256  

    # CUDA_VISIBLE_DEVICES=4 python train.py --lr 0.0001 --d_model 8 --batch_size 512 --k 1
