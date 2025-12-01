from utils import calc_psnr_and_ssim
from model import Vgg19
#import torchmetrics
#from torchmetrics import PeakSignalNoiseRatio ,StructuralSimilarityIndexMeasure
import os
import numpy as np
from imageio import imread, imsave
from PIL import Image
from sewar import vifp
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils

#PSNR = PeakSignalNoiseRatio()
#SSIM = StructuralSimilarityIndexMeasure()

class Trainer():
    def __init__(self, args, logger, dataloader, model, loss_all):
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.loss_all = loss_all
        self.device = torch.device('cpu') if args.cpu else torch.device('cuda')
        self.vgg19thr = Vgg19.Vgg19thr(requires_grad=True).to(self.device)
        if ((not self.args.cpu) and (self.args.num_gpu > 1)):            
            self.vgg19thr = nn.DataParallel(self.vgg19thr, list(range(self.args.num_gpu)))

        self.params = [
            {"params": filter(lambda p: p.requires_grad, self.model.MainNet.parameters()),
            "lr": args.lr_rate
            },
            {"params": filter(lambda p: p.requires_grad, self.model.LTVE.parameters()), 
            "lr": args.lr_rate_lte
            }
        ]
        self.optimizer = optim.Adam(self.params, betas=(args.beta1, args.beta2), eps=args.eps)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.decay, gamma=self.args.gamma)
        self.max_psnr = 0.
        self.max_psnr_epoch = 0
        self.max_ssim = 0.
        self.max_ssim_epoch = 0

    def load(self, model_path=None):
        if (model_path):
            self.logger.info('load_model_path: ' + model_path)
            #model_state_dict_save = {k.replace('module.',''):v for k,v in torch.load(model_path).items()}
            model_state_dict_save = {k:v for k,v in torch.load(model_path, map_location=self.device).items()}
            model_state_dict = self.model.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.model.load_state_dict(model_state_dict)

    def prepare(self, sample_batched):
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched

    def train(self, current_epoch=0, is_init=False):
        self.model.train()
        if (not is_init):
            self.scheduler.step()
        self.logger.info('Current epoch learning rate: %e' %(self.optimizer.param_groups[0]['lr']))

        for i_batch, sample_batched in enumerate(self.dataloader['train']):
            self.optimizer.zero_grad()

            sample_batched = self.prepare(sample_batched)
            lr_vis = sample_batched['LR_vis']
            lr_sr_vis = sample_batched['LR_sr_vis']
            hr_vis = sample_batched['HR_vis']
            ref_vis = sample_batched['Ref_vis']
            ref_sr_vis = sample_batched['Ref_sr_vis']
            
            lr_thr = sample_batched['LR_thr']
            lr_sr_thr = sample_batched['LR_sr_thr']
            hr_thr = sample_batched['HR_thr']
            ref_thr = sample_batched['Ref_thr']
            ref_sr_thr = sample_batched['Ref_sr_thr']
            lr_sr2_thr = lr_sr_thr[:,0,:,:].reshape(hr_thr.shape[0],1,hr_thr.shape[2],hr_thr.shape[2])
            lrsr = torch.cat((lr_sr_vis, lr_sr2_thr), 1)
            #print(lrsr.shape)
            
            sr, S, T_lv3, T_lv2, T_lv1 = self.model(lr_vis=lr_vis, lrsr_vis=lr_sr_vis, ref_vis=ref_vis, refsr_vis=ref_sr_vis, lr_thr=lr_thr, lrsr_thr=lr_sr_thr, ref_thr=ref_thr,refsr_thr=ref_sr_thr)
            hr = torch.cat((hr_vis, hr_thr[:,0,:,:].reshape(hr_thr.shape[0],1,hr_thr.shape[2],hr_thr.shape[2])), dim=1)
            ### calc loss
            #print('hr', hr.shape)
            #print('sr', sr.shape)
            is_print = ((i_batch + 1) % self.args.print_every == 0) ### flag of print

            rec_loss = self.args.rec_w * self.loss_all['rec_loss'](sr, hr)
            loss = rec_loss
            if (is_print):
                self.logger.info( ('init ' if is_init else '') + 'epoch: ' + str(current_epoch) + 
                    '\t batch: ' + str(i_batch+1) )
                self.logger.info( 'rec_loss: %.10f' %(rec_loss.item()) )

            if (not is_init):
                if ('stv_loss' in self.loss_all):
                    stv_loss = self.args.stv_w * self.loss_all['stv_loss'](sr, hr)
                    loss += stv_loss
                    if (is_print):
                        self.logger.info( 'stv_loss: %.10f' %(stv_loss.item()) )
                if ('tpl_loss' in self.loss_all):
                    tpl_loss = self.args.tpl_w * self.loss_all['tpl_loss'](sr, lrsr)
                    loss += tpl_loss
                    if (is_print):
                        self.logger.info( 'tpl_loss: %.10f' %(tpl_loss.item()) )
                if ('adv_loss' in self.loss_all):
                    adv_loss = self.args.adv_w * self.loss_all['adv_loss'](sr, hr)
                    if (is_print):
                        self.logger.info( 'adv_loss: %.10f' %(adv_loss.item()) )
                  
            loss.backward()
            self.optimizer.step()

        if ((not is_init) and current_epoch % self.args.save_every == 0):
            self.logger.info('saving the model...')
            tmp = self.model.state_dict()
            model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp if 
                (('SearchNet' not in key) and ('_copy' not in key))}
            model_name = self.args.save_dir.strip('/')+'/model/model_'+str(current_epoch).zfill(5)+'.pt'
            torch.save(model_state_dict, model_name)

    
    def evaluate(self, current_epoch=0):
        self.logger.info(f'Epoch {current_epoch} evaluation process...')
        self.out = nn.Conv2d(4, 3, kernel_size=1, stride=1)
    
        if self.args.dataset == 'DRONES':
           self.model.eval()
           with torch.no_grad():
                psnr, ssim, vifp_total, cnt = 0., 0., 0., 0
                for i_batch, sample_batched in enumerate(self.dataloader['test']['1']):
                    cnt += 1
                    sample_batched = self.prepare(sample_batched)
                    lr_vis = sample_batched['LR_vis']
                    lr_sr_vis = sample_batched['LR_sr_vis']
                    hr_vis = sample_batched['HR_vis']
                    ref_vis = sample_batched['Ref_vis']
                    ref_sr_vis = sample_batched['Ref_sr_vis']

                    lr_thr = sample_batched['LR_thr']
                    lr_sr_thr = sample_batched['LR_sr_thr']
                    hr_thr = sample_batched['HR_thr']
                    ref_thr = sample_batched['Ref_thr']
                    ref_sr_thr = sample_batched['Ref_sr_thr']
                
                    hr = torch.cat((hr_vis, hr_thr[:, 0, :, :].reshape(hr_thr.shape[0], 1, hr_thr.shape[2], hr_thr.shape[2])), dim=1)
                    sr, _, _, _, _ = self.model(lr_vis=lr_vis, lr_thr=lr_thr, lrsr_vis=lr_sr_vis, lrsr_thr=lr_sr_thr, ref_vis=ref_vis, ref_thr=ref_thr, refsr_vis=ref_sr_vis, refsr_thr=ref_sr_thr)
                    
                    if self.args.eval_save_results:
                       sr_save = (sr + 1.) * 127.5
                       sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                       imsave(os.path.join(self.args.save_dir, 'save_results', f'{i_batch:05d}.jpg'), sr_save[:, :, 0])
                
                # Calculate PSNR and SSIM
                    _psnr, _ssim = calc_psnr_and_ssim(sr.detach(), hr.detach())
                    _vifp = vifp(hr.detach().cpu().numpy(), sr.detach().cpu().numpy())
                    psnr += _psnr
                    ssim += _ssim
                    vifp_total += _vifp

                if cnt != 0:
                   psnr_ave = psnr / cnt
                   ssim_ave = ssim / cnt
                   vif_ave = vifp_total / cnt


                   self.logger.info(f'Ref PSNR (now): {psnr_ave:.3f} \t SSIM (now): {ssim_ave:.4f}')

                   if psnr_ave > self.max_psnr:
                      self.max_psnr = psnr_ave
                      self.max_psnr_epoch = current_epoch
                   if ssim_ave > self.max_ssim:
                      self.max_ssim = ssim_ave
                      self.max_ssim_epoch = current_epoch
                   if vif_ave > self.max_vif:
                      self.max_vif = vif_ave
                      self.max_vif_epoch = current_epoch


                   self.logger.info(f'Ref PSNR (max): {self.max_psnr:.3f} ({self.max_psnr_epoch}) \t SSIM (max): {self.max_ssim:.4f} ({self.max_ssim_epoch})')
                   self.logger.info(f'Ref VIF (max): {self.max_vif:.4f} ({self.max_vif_epoch}) \t VSI (max): {self.max_vsi:.4f} ({self.max_vsi_epoch})')

        self.logger.info('Evaluation over.')
    def test(self):
        self.logger.info('Test process...')
        self.logger.info('lrvis path:     %s' %(self.args.lr_vis_path))
        self.logger.info('refvis path:    %s' %(self.args.ref_vis_path))
        self.logger.info('lrthr path:     %s' %(self.args.lr_thr_path))
        self.logger.info('refthr path:    %s' %(self.args.ref_thr_path))

        ### LR and LR_sr
        LR_vis = imread(self.args.lr_vis_path)
        LR_thr = imread(self.args.lr_thr_path)
        LR_thr = np.repeat(LR_thr[..., np.newaxis], 3, -1)
        h1, w1 = LR_vis.shape[:2]
        LR_sr_vis = np.array(Image.fromarray(LR_vis).resize((w1*4, h1*4), Image.NEAREST))
        LR_sr_thr = np.array(Image.fromarray(LR_thr).resize((w1*4, h1*4), Image.NEAREST))
        
        ### Ref and Ref_sr
        Ref_vis = imread(self.args.ref_vis_path)
        Ref_thr = imread(self.args.ref_thr_path)
        Ref_thr = np.repeat(Ref_thr[..., np.newaxis], 3, -1)
        h2, w2 = Ref_vis.shape[:2]
        h2, w2 = h2//4*4, w2//4*4
        Ref_vis = Ref_vis[:h2, :w2, :]
        Ref_thr = Ref_thr[:h2, :w2, :]
        Ref_sr_vis = np.array(Image.fromarray(Ref_vis).resize((w2//4, h2//4), Image.NEAREST))
        Ref_sr_vis = np.array(Image.fromarray(Ref_sr_vis).resize((w2, h2), Image.NEAREST))
        Ref_sr_thr = np.array(Image.fromarray(Ref_thr).resize((w2//4, h2//4), Image.NEAREST))
        Ref_sr_thr = np.array(Image.fromarray(Ref_sr_thr).resize((w2, h2), Image.NEAREST))

        ### change type
        LR_vis = LR_vis.astype(np.float32)
        LR_thr = LR_thr.astype(np.float32)
        LR_sr_vis = LR_sr_vis.astype(np.float32)
        LR_sr_thr = LR_sr_thr.astype(np.float32)
        Ref_vis = Ref_vis.astype(np.float32)
        Ref_sr_vis = Ref_sr_vis.astype(np.float32)
        Ref_thr = Ref_thr.astype(np.float32)
        Ref_sr_thr = Ref_sr_thr.astype(np.float32)

        ### rgb range to [-1, 1]
        LR_vis = LR_vis / 127.5 - 1.
        LR_thr = LR_thr / 127.5 - 1.
        LR_sr_vis = LR_sr_vis / 127.5 - 1.
        LR_sr_thr = LR_sr_thr / 127.5 - 1.
        Ref_vis = Ref_vis / 127.5 - 1.
        Ref_sr_vis = Ref_sr_vis / 127.5 - 1.
        Ref_thr = Ref_thr / 127.5 - 1.
        Ref_sr_thr = Ref_sr_thr / 127.5 - 1.

        ### to tensor
        LR_vis_t = torch.from_numpy(LR_vis.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        LR_thr_t = torch.from_numpy(LR_thr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        LR_sr_vis_t = torch.from_numpy(LR_sr_vis.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        LR_sr_thr_t = torch.from_numpy(LR_sr_thr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        Ref_vis_t = torch.from_numpy(Ref_vis.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        Ref_thr_t = torch.from_numpy(Ref_thr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        Ref_sr_vis_t = torch.from_numpy(Ref_sr_vis.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        Ref_sr_thr_t = torch.from_numpy(Ref_sr_thr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            #sr, _, _, _, _ = self.model(lr=LR_t, lrsr=LR_sr_t, ref=Ref_t, refsr=Ref_sr_t)
            sr,_, _, _, _ = self.model(lr_vis=LR_vis_t, lr_thr=LR_thr_t, lrsr_vis=LR_sr_vis_t, lrsr_thr=LR_sr_thr_t, ref_vis=Ref_vis_t, ref_thr=Ref_thr_t, refsr_vis=Ref_sr_vis_t, refsr_thr=Ref_sr_thr_t)
            #sr = torch.cat((sr_vis, sr_thr),dim=1)
            sr_save = (sr+1.) * 127.5
            h = w = 360
            print('sr', sr.shape)
            sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
            sr_save_vis =np.concatenate(((sr_save[:,:,0]).reshape(h, w, 1),(sr_save[:,:,1]).reshape(h, w, 1),(sr_save[:,:,2]).reshape(h, w, 1)),axis=2)
            print('sr_vis', sr_save_vis.shape)
            save_path = os.path.join(self.args.save_dir, 'save_results', os.path.basename(self.args.lr_vis_path)+'.jpg')
            imsave(save_path, sr_save_vis)
            self.logger.info('output path: %s' %(save_path))

        self.logger.info('Test over.')
