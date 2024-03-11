import os
import time
import argparse
import json
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from torch.utils.data import DistributedSampler, DataLoader
from dataset import MelDataset, get_dataset_filelist
from model import ArticulatoryInverter, ModelConfig
from module import upsample_ssl_emb, pearson_correlation
from scheduler import CosineAnnealingWarmUpRestarts
from WavLM import WavLM, WavLMConfig
from utils import scan_checkpoint, load_checkpoint, save_checkpoint, plot_ema_compared
from discriminator import MDPD
from loss_functions import RMSELoss, feature_loss, generator_loss, discriminator_loss
import torch.nn as nn
import torch.nn.functional as F
import pkbar
from statsmodels.nonparametric.smoothers_lowess import lowess


torch.backends.cudnn.benchmark = True

def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)
    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    cfg = ModelConfig()
    model = ArticulatoryInverter(cfg).to(device)
    mdpd = MDPD().to(device)

    scaler = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()

    wavlm_ckpt = torch.load('/path/to/WavLMckpt/WavLM-Large.pt')
    wavlm_cfg = WavLMConfig(wavlm_ckpt['cfg'])
    wavlm = WavLM(wavlm_cfg).to(device)
    wavlm.load_state_dict(wavlm_ckpt['model'])
    wavlm.eval()
    
    if rank == 0:
        print(model)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):  
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')

    steps = 0
    if cp_g is None :
        state_dict_g = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        model.load_state_dict(state_dict_g['model'])
        if 'mdpd' in state_dict_g:
            mdpd.load_state_dict(state_dict_g['mdpd'])
        steps = state_dict_g['steps'] + 1
        last_epoch = state_dict_g['epoch']
        
    if h.num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[rank]).to(device)
        mdpd = DistributedDataParallel(mdpd, device_ids=[rank]).to(device)


    optim_g = torch.optim.Adam(model.parameters(), lr=h.learning_rate, betas=(h.adam_b1, h.adam_b2))
    optim_d = torch.optim.AdamW(mdpd.parameters(),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_g is not None:
        optim_g.load_state_dict(state_dict_g['optim_g'])
        if 'mdpd' in state_dict_g:
            optim_d.load_state_dict(state_dict_g['optim_d']) 

    scheduler_g  = CosineAnnealingWarmUpRestarts(optim_g, 
                                                T_0=1, 
                                                T_mult=2, 
                                                eta_max=3e-4, 
                                                T_up=0, 
                                                gamma=0.99)
    scheduler_d  = CosineAnnealingWarmUpRestarts(optim_d, 
                                                T_0=1, 
                                                T_mult=2, 
                                                eta_max=3e-4, 
                                                T_up=0, 
                                                gamma=0.99)

    traindata, valdata = get_dataset_filelist('spkid') # F01, F02, ...
    
    trainset = MelDataset(traindata, h.n_fft, h.num_mels,
                            h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                            shuffle=True if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                            train=True)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                                sampler=train_sampler,
                                batch_size=h.batch_size,
                                pin_memory=True,
                                drop_last=True)
    val_batch_size = 1
    if rank == 0:
        validset = MelDataset(valdata, h.n_fft, h.num_mels,
                                h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                                fmax_loss=h.fmax_for_loss, device=device, train=False)

        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                        sampler=None,
                                        batch_size=val_batch_size,
                                        pin_memory=True,
                                        drop_last=True)
        
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    model.train()
    mdpd.train()
    mse = nn.MSELoss()
    rmse = RMSELoss()
    #################################### Training ####################################
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))
        accum_iter = 2
        for ii, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()

            cleanaudio, ema, index, filename = batch
            ema = torch.autograd.Variable(ema.to(device, non_blocking=True))
            cleanaudio = torch.autograd.Variable(cleanaudio.to(device, non_blocking=True))

            with torch.no_grad():
                if wavlm_cfg.normalize:
                    cleanaudio_norm = torch.nn.functional.layer_norm(cleanaudio , cleanaudio.shape)
                w_s, layer_results = wavlm.extract_features(F.pad(cleanaudio_norm, (0,160)), output_layer=24, ret_layer_results=True)[0]

            ema_hat = model(upsample_ssl_emb(w_s, ratio=2), mask=True)
            if ema_hat.size(2) != ema.size(2):
                minlen = min(ema_hat.size(2), ema.size(2))
                ema_hat = ema_hat[:,:,:minlen]
                ema = ema[:,:,:minlen]
                
            mse_loss = mse(ema, ema_hat)

            if steps >= 10400:
                loss_disc_all = 0
                y_df_hat_r, y_df_hat_g, _, _ = mdpd(ema, ema_hat.detach(), mask=True)
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
                loss_disc_all = loss_disc_all + loss_disc_f

                if ii == 0:
                    optim_d.zero_grad()
                loss_d = loss_disc_all / accum_iter
                scaler_d.scale(loss_d).backward()
                if ((ii+1) % accum_iter ==0) or (ii+1 == len(train_loader)):
                    scaler_d.step(optim_d)
                    scaler_d.update()
                    optim_d.zero_grad()

                loss_gen_all = 0
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mdpd(ema, ema_hat, mask=True)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_all = loss_gen_f + loss_fm_f
                loss_gen_all = loss_gen_all + mse_loss
                
            else:
                loss_gen_all = mse_loss
            
            if ii == 0:
                optim_g.zero_grad()

            loss_g = loss_gen_all / accum_iter
            scaler.scale(loss_g).backward()
            if ((ii+1) % accum_iter ==0) or (ii+1 == len(train_loader)):
                scaler.step(optim_g)
                scaler.update()
                optim_g.zero_grad()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    if steps >= 10400:
                        print('Steps : {:d}, Gen Loss Total : {:4.3f}, MSE-loss: {:4.5f}, Gen-loss: {:4.5f}, Disc-loss: {:4.5f}, s/b : {:4.3f}'\
                                                            .format(steps, loss_gen_all, mse_loss, loss_gen_all-mse_loss, loss_disc_all, time.time() - start_b))
                    else:
                        print('Steps : {:d}, Gen Loss Total : {:4.3f}, MSE-loss: {:4.5f}, s/b : {:4.3f}'\
                                                            .format(steps, loss_gen_all, mse_loss, time.time() - start_b))
                # Checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    if steps >= 10400:
                        save_checkpoint(checkpoint_path,
                                    {'model': (model.module if h.num_gpus > 1 else model).state_dict(),
                                    'mdpd': (mdpd.module if h.num_gpus > 1 else mdpd).state_dict(),
                                    'optim_g': optim_g.state_dict(),
                                    'optim_d': optim_d.state_dict(),
                                    'steps': steps,
                                    'epoch': epoch})
                    else:
                        save_checkpoint(checkpoint_path,
                                    {'model': (model.module if h.num_gpus > 1 else model).state_dict(),
                                    'optim_g': optim_g.state_dict(),
                                    'steps': steps,
                                    'epoch': epoch})
                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    if steps >= 10400:
                        sw.add_scalar("training/tot_loss", loss_gen_all + loss_disc_all, steps)
                        sw.add_scalar("training/mse_loss", mse_loss, steps)
                        sw.add_scalar("training/gen_loss", loss_gen_all-mse_loss, steps)
                        sw.add_scalar("training/disc_loss", loss_disc_all, steps)
                        sw.add_scalar("training/gen_lr", optim_g.param_groups[0]['lr'], steps)
                        sw.add_scalar("training/disc_lr", optim_d.param_groups[0]['lr'], steps)
                    else:
                        sw.add_scalar("training/tot_loss", loss_gen_all, steps)
                        sw.add_scalar("training/mse_loss", mse_loss, steps)
                        sw.add_scalar("training/gen_lr", optim_g.param_groups[0]['lr'], steps)
                        sw.add_scalar("training/disc_lr", optim_d.param_groups[0]['lr'], steps)


                #################################### Validation ####################################
                if steps % a.validation_interval == 0:  # and steps != 0:
                    model.eval()
                    torch.cuda.empty_cache()
                    mse_error = 0
                    rmse_error = 0
                    pcc_error = 0
                    with torch.no_grad():
                        pcount = 0
                        pbar = pkbar.Pbar('validation', len(validation_loader))
                        for jj, batch in enumerate(validation_loader):
                            pbar.update(pcount)
                            cleanaudio, ema, index, filename = batch
                            ema = ema.to(device)
                            cleanaudio = cleanaudio.to(device)
                                
                            if wavlm_cfg.normalize:
                                cleanaudio_norm = torch.nn.functional.layer_norm(cleanaudio , cleanaudio.shape)
                            w_s, layer_results = wavlm.extract_features(F.pad(cleanaudio_norm, (0,160)), output_layer=24, ret_layer_results=True)[0]

                            ema_hat = model(upsample_ssl_emb(w_s, ratio=2))

                            if ema_hat.size(2) != ema.size(2):
                                minlen = min(ema_hat.size(2), ema.size(2))
                                ema_hat = ema_hat[:,:,:minlen]
                                ema = ema[:,:,:minlen]
                                
                            mse_error += mse(ema, ema_hat)
                            
                            ema = ema.squeeze(0)
                            ema_hat = ema_hat.squeeze(0)
                            
                            ema_hat = ema_hat.cpu().numpy()
                            chn, tim_len = ema_hat.shape
                            time_x = np.linspace(0, tim_len-1, tim_len)
                            for ch in range(chn):
                                if ch == 0:
                                    lw_sm_ema = lowess(ema_hat[ch], time_x, frac=0.05)[:,1]
                                else:
                                    lw_sm_ema = np.row_stack((lw_sm_ema, lowess(ema_hat[ch], time_x, frac=0.05)[:,1]))
                            lw_sm_ema = torch.from_numpy(lw_sm_ema).to(device)
                            rmse_error += rmse(ema, lw_sm_ema)
                            pcc_error += pearson_correlation(ema, lw_sm_ema)
                            
                            if jj == 0 or jj == 60 or jj == 120 or jj == 180 or jj == 240:
                                sw.add_figure('generatedimu/imu_hat_{}'.format(filename),
                                                plot_ema_compared(lw_sm_ema.cpu().numpy(), ema.cpu().numpy()), steps)
                                sw.add_audio('gt/{}'.format(filename), cleanaudio[0], steps, h.sampling_rate)
                            pcount += 1
                            
                        val_mse_err = mse_error / (jj+1)
                        val_rmse_err = rmse_error / (jj+1)
                        val_ema_pcc_tot = pcc_error / (jj+1) 
                        sw.add_scalar("validation/mse_error", val_mse_err, steps)
                        sw.add_scalar("validation/rmse_error", val_rmse_err, steps)
                        sw.add_scalar("validation/ema_pcc_tot", torch.mean(val_ema_pcc_tot), steps)
                        
                        sw.add_scalar("validation/ema_pcc_TR_x", val_ema_pcc_tot[0], steps)
                        sw.add_scalar("validation/ema_pcc_TR_y", val_ema_pcc_tot[1], steps)
                        sw.add_scalar("validation/ema_pcc_TR_z", val_ema_pcc_tot[2], steps)
                        sw.add_scalar("validation/ema_pcc_TB_x", val_ema_pcc_tot[3], steps)
                        sw.add_scalar("validation/ema_pcc_TB_y", val_ema_pcc_tot[4], steps)
                        sw.add_scalar("validation/ema_pcc_TB_z", val_ema_pcc_tot[5], steps)
                        sw.add_scalar("validation/ema_pcc_TT_x", val_ema_pcc_tot[6], steps)
                        sw.add_scalar("validation/ema_pcc_TT_y", val_ema_pcc_tot[7], steps)
                        sw.add_scalar("validation/ema_pcc_TT_z", val_ema_pcc_tot[8], steps)
                        sw.add_scalar("validation/ema_pcc_UL_x", val_ema_pcc_tot[9], steps)
                        sw.add_scalar("validation/ema_pcc_UL_y", val_ema_pcc_tot[10], steps)
                        sw.add_scalar("validation/ema_pcc_UL_z", val_ema_pcc_tot[11], steps)
                        sw.add_scalar("validation/ema_pcc_LL_x", val_ema_pcc_tot[12], steps)
                        sw.add_scalar("validation/ema_pcc_LL_y", val_ema_pcc_tot[13], steps)
                        sw.add_scalar("validation/ema_pcc_LL_z", val_ema_pcc_tot[14], steps)
                        sw.add_scalar("validation/ema_pcc_LI_x", val_ema_pcc_tot[15], steps)
                        sw.add_scalar("validation/ema_pcc_LI_y", val_ema_pcc_tot[16], steps)
                        sw.add_scalar("validation/ema_pcc_LI_z", val_ema_pcc_tot[17], steps)

                    model.train()
            steps += 1
        
        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', default='/path/to/save/checkpoints/')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--training_epochs', default=35, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=800, type=int)
    parser.add_argument('--summary_interval', default=50, type=int)
    parser.add_argument('--validation_interval', default=800, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass
    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()

