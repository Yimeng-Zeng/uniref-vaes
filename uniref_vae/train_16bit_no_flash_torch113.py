import torch
from data import DataModuleKmers
import time 
import wandb 
import os 
from tqdm import tqdm
os.environ["WANDB_SILENT"] = "true" 

torch.set_float32_matmul_precision("medium")


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

    if args_dict['vae_type'] == '16bit':
        from flash_attention_test import InfoTransformerVAE
        print("using vae with 16bit precision")

    model = InfoTransformerVAE(vocab_size=len(datamodule.train.vocab), d_model=args_dict['d_model'], \
                               kl_factor=args_dict['kl_factor'], encoder_dropout=args_dict['dropout'], decoder_dropout=args_dict['dropout'])
    print("model created with kl factor: ", args_dict['kl_factor'])

    if args_dict['load_ckpt']: 
        state_dict = torch.load(args_dict['load_ckpt']) # load state dict 
        model.load_state_dict(state_dict, strict=True) 
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    model = model.cuda()     # no need for casting with .half() when using torch.amp

    optimizer = torch.optim.Adam([ {'params': model.parameters()} ], lr=args_dict['lr']) 
    lowest_loss = torch.inf 

    # init gradient scaler if specified
    if args_dict['gradscale']:
        scaler = torch.cuda.amp.GradScaler()
        print("using gradient scaling")

    if args_dict['precision'] == 'bf16':
        print("using bf16 precision")
    elif args_dict['precision'] == 'fp16':
        print("using fp16 precision")
    elif args_dict['precision'] == 'fp32':
        print("using fp32 precision")

    for epoch in range(args_dict['max_epochs']):
        start_time = time.time() 

        print("Starting training epoch: ", epoch)

        model = model.train()  
        sum_train_loss = 0.0 
        num_iters = 0
        for data in tqdm(train_loader):
            optimizer.zero_grad() 
            input = data.cuda() 
            # cast to bf16 or fp16 depending on specification
            if args_dict['precision'] == 'bf16':
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_dict = model(input) 
            elif args_dict['precision'] == 'fp16':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out_dict = model(input)
            elif args_dict['precision'] == 'fp32':
                out_dict = model(input)

            train_dict = {'train_' + k:out_dict[k] for k in out_dict.keys() }
            tracker.log(train_dict) 
            loss = out_dict['loss'] 
            sum_train_loss += loss.item()  
            num_iters += 1

            if args_dict['gradscale']:

                # Scales the loss, and calls backward()
                # to create scaled gradients
                scaler.scale(loss).backward()

                # Unscales gradients and calls
                # or skips optimizer.step()
                scaler.step(optimizer)

                # Updates the scale for next iteration
                scaler.update()
            else:
                loss.backward() 
                optimizer.step()

        avg_train_loss = sum_train_loss/num_iters
        tracker.log({'time for train epoch':time.time() - start_time,
                    'avg_train_loss_per_epoch':avg_train_loss,
                    'epochs completed':epoch+1 }) 
        
        print("Finished training epoch: ", epoch)
        print("Time for epoch: ", time.time() - start_time)

        if epoch % args_dict['compute_val_freq'] == 0: 
            start_time = time.time() 
            model = model.eval()  
            sum_val_loss = 0.0 
            num_val_iters = 0
            for data in val_loader:
                input = data.cuda() 
                # cast to bf16 or fp16 depending on specification
                if args_dict['precision'] == 'bf16':
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out_dict = model(input) 
                elif args_dict['precision'] == 'fp16':
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        out_dict = model(input)
                elif args_dict['precision'] == 'fp32':
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
    parser.add_argument('--k', type=int, default=3 )  
    parser.add_argument('--kl_factor', type=float, default=0.0001 )
    parser.add_argument('--vae_type', type=str, default='16bit' ) 
    parser.add_argument('--data_version', type=int, default=1 ) 
    parser.add_argument('--d_model', type=int, default=128 )
    # torch113 should not have flash attention integrated, and also compile is not supported
    parser.add_argument('--dropout', type=float, default=0.05 )
    # add fp16 vs bf16 option
    parser.add_argument('--precision', type=str, default="bf16" )
    # add gradscale option
    parser.add_argument('--gradscale', type=str, default='False', help='Whether to use gradient scaling')
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
    # torch113 should not have flash attention integrated, and also compile is not supported
    args_dict['dropout'] = args.dropout
    # add fp16 vs bf16 option
    args_dict['precision'] = args.precision
    # add gradscale option
    args_dict['gradscale'] = args.gradscale.lower() == 'true'
    train(args_dict) 
