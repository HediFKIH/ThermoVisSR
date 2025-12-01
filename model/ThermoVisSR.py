from model import MainNet, LTE, SearchTransfer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class ThermoVisSR(nn.Module):
    def __init__(self, args):
        super(ThermoVisSR, self).__init__()
        self.args = args
        self.num_res_blocks = list( map(int, args.num_res_blocks.split('+')) )
        self.MainNet = MainNet.MainNet(num_res_blocks=self.num_res_blocks,n_feats=args.n_feats, 
            res_scale=args.res_scale)
        self.LTVE      = LTE.LTVE(requires_grad=True)
        self.LTVE_copy = LTE.LTVE(requires_grad=False) ### used in transferal perceptual loss
        self.LTTE      = LTE.LTTE(requires_grad=True)
        self.LTTE_copy = LTE.LTTE(requires_grad=False)
        self.SearchTransfer = SearchTransfer.SearchTransfer()

    def forward(self, lr_vis=None, lrsr_vis=None, ref_vis=None, refsr_vis=None, sr_vis=None,lr_thr=None, lrsr_thr=None, ref_thr=None, refsr_thr=None, sr_thr=None):
        if (type(sr_vis) != type(None) and type(sr_thr) != type(None) ):
            ### used in transferal perceptual loss
            self.LTVE_copy.load_state_dict(self.LTE.state_dict())
            sr_vis_lv1, sr_vis_lv2, sr_vis_lv3 = self.LTE_copy((sr_vis + 1.) / 2.)
            ### used in transferal perceptual loss
            self.LTTE_copy.load_state_dict(self.LTE1.state_dict())
            sr_thr_lv1, sr_thr_lv2, sr_thr_lv3 = self.LTE1_copy((sr_thr + 1.) / 2.)
            sr_lv1 = torch.cat((sr_vis_lv1,sr_thr_lv1), dim=1)
            sr_lv2 = torch.cat((sr_vis_lv2,sr_thr_lv2), dim=1)
            sr_lv3 = torch.cat((sr_vis_lv3,sr_thr_lv3), dim=1)
            return sr_lv1, sr_lv2, sr_lv3
        ref_vis_lv1, ref_vis_lv2, ref_vis_lv3  = self.LTVE((ref_vis.detach() + 1.) / 2.)
        ref_thr_lv1, ref_thr_lv2, ref_thr_lv3  = self.LTTE((ref_thr.detach() + 1.) / 2.)
        lrsr_vis_lv1, lrsr_vis_lv2, lrsr_vis_lv3  = self.LTVE((lrsr_vis.detach() + 1.) / 2.)
        refsr_vis_lv1, refsr_vis_lv2, refsr_vis_lv3 = self.LTVE((refsr_vis.detach() + 1.) / 2.)
        lrsr_thr_lv1, lrsr_thr_lv2, lrsr_thr_lv3  = self.LTTE((lrsr_thr.detach() + 1.) / 2.)
        refsr_thr_lv1, refsr_thr_lv2, refsr_thr_lv3 = self.LTTE((refsr_thr.detach() + 1.) / 2.)
        ref_vis_lv1, ref_vis_lv2, ref_vis_lv3 = self.LTVE((ref_vis.detach() + 1.) / 2.)
        ref_thr_lv1, ref_thr_lv2, ref_thr_lv3 = self.LTTE((ref_thr.detach() + 1.) / 2.)
        lrsr_lv1 = torch.cat((lrsr_vis_lv1,lrsr_thr_lv1), dim=1)
        lrsr_lv2 = torch.cat((lrsr_vis_lv2,lrsr_thr_lv2), dim=1)
        lrsr_lv3 = torch.cat((lrsr_vis_lv3,lrsr_thr_lv3), dim=1)
        refsr_lv3 = torch.cat((refsr_vis_lv3,refsr_thr_lv3), dim=1)
        refsr_lv2 = torch.cat((refsr_vis_lv2, refsr_thr_lv2), dim=1)
        refsr_lv1 = torch.cat((refsr_vis_lv1, refsr_thr_lv1), dim=1)
        ref_lv1 = torch.cat((ref_vis_lv1, ref_thr_lv1), dim=1)
        ref_lv2 = torch.cat((ref_vis_lv2, ref_thr_lv2), dim=1)
        ref_lv3 =torch.cat((ref_vis_lv3,ref_thr_lv3), dim=1)
        S3,S2,S1, R_lv3_star_arg,T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv1, lrsr_lv2, lrsr_lv3,refsr_lv1, refsr_lv2, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)
        #sr_vis,sr_thr = self.MainNet(lr_vis, lr_thr, S, T_lv3, T_lv2, T_lv1)
        sr = self.MainNet(lr_vis, lr_thr, lrsr_vis, lrsr_thr,  S3, S2, S1,  R_lv3_star_arg, T_lv3, T_lv2, T_lv1)
        return sr,  S3, T_lv3, T_lv2, T_lv1
        #return sr_vis,sr_thr, S, T_lv3, T_lv2, T_lv1
