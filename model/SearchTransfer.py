import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SearchTransfer(nn.Module):
    def __init__(self):
        super(SearchTransfer, self).__init__()

    def bis(self, input, dim, index):
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, lrsr_lv1, lrsr_lv2,lrsr_lv3, refsr_lv1, refsr_lv2, refsr_lv3, ref_lv1, ref_lv2, ref_lv3):
        ### search
        lrsr_lv3_unfold  = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)
        lrsr_lv2_unfold = F.unfold(lrsr_lv2, kernel_size=(6,6), padding =2, stride=2)
        lrsr_lv1_unfold = F.unfold(lrsr_lv1, kernel_size=(12,12), padding=4, stride=4)
        refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv2_unfold = F.unfold(refsr_lv2, kernel_size=(6,6), padding =2, stride=2)
        refsr_lv1_unfold = F.unfold(refsr_lv1, kernel_size=(12,12), padding=4, stride=4)       
        #refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)

        refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2) # [N, Hr*Wr, C*k*k]
        refsr_lv2_unfold = F.normalize(refsr_lv2_unfold, dim=2)
        refsr_lv1_unfold = F.normalize(refsr_lv1_unfold, dim=2)
        lrsr_lv3_unfold  = F.normalize(lrsr_lv3_unfold, dim=1) # [N, C*k*k, H*W]
        lrsr_lv2_unfold  = F.normalize(lrsr_lv2_unfold, dim=1)
        lrsr_lv1_unfold  = F.normalize(lrsr_lv1_unfold, dim=1)
        #R_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold) #[N, Hr*Wr, H*W]
        mean_x3 = torch.mean(refsr_lv3_unfold)
        mean_x2 = torch.mean(refsr_lv2_unfold)
        mean_x1 = torch.mean(refsr_lv1_unfold)
        
        xm3 = refsr_lv3_unfold.sub(mean_x3.expand_as(refsr_lv3_unfold))
        ym3 = lrsr_lv3_unfold.sub(mean_x3.expand_as(lrsr_lv3_unfold))
        xm3 = xm3.permute(0,2,1)
        num3 = torch.bmm(xm3, ym3)
        den3 = (torch.sqrt(torch.sum(xm3 ** 2)) * torch.sqrt(torch.sum(ym3 ** 2)))
        R_lv3 = num3 / den3
        R_lv3_merged, R_lv3_merged_arg = torch.max(R_lv3, dim=1) #[N, H*W]

        xm2 = refsr_lv2_unfold.sub(mean_x2.expand_as(refsr_lv2_unfold))
        ym2 = lrsr_lv2_unfold.sub(mean_x2.expand_as(lrsr_lv2_unfold))
        xm2 = xm2.permute(0,2,1)
        num2 = torch.bmm(xm2, ym2)
        den2 = (torch.sqrt(torch.sum(xm2 ** 2)) * torch.sqrt(torch.sum(ym3 ** 2)))
        R_lv2 = num2 / den2
        R_lv2_merged, R_lv2_merged_arg = torch.max(R_lv2, dim=1) #[N, H*W]
        
        xm1 = refsr_lv1_unfold.sub(mean_x1.expand_as(refsr_lv1_unfold))
        ym1 = lrsr_lv1_unfold.sub(mean_x1.expand_as(lrsr_lv1_unfold))
        xm1 = xm1.permute(0,2,1)
        num1 = torch.bmm(xm1, ym1)
        den1 = (torch.sqrt(torch.sum(xm1 ** 2)) * torch.sqrt(torch.sum(ym1 ** 2)))
        R_lv1 = num1 / den1
        R_lv1_merged, R_lv1_merged_arg = torch.max(R_lv1, dim=1) #[N, H*W]
        ### transfer
        ref_lv3_unfold = F.unfold((ref_lv3), kernel_size=(3, 3), padding=1)
        ref_lv2_unfold = F.unfold((ref_lv2), kernel_size=(6, 6), padding=2, stride=2)
        ref_lv1_unfold = F.unfold((ref_lv1), kernel_size=(12, 12), padding=4, stride=4)

        '''ref_lv3_unfold = F.unfold((ref_lv3-refsr_lv3), kernel_size=(3, 3), padding=1)
        ref_lv2_unfold = F.unfold((ref_lv2-refsr_lv2), kernel_size=(6, 6), padding=2, stride=2)
        ref_lv1_unfold = F.unfold((ref_lv1-refsr_lv1), kernel_size=(12, 12), padding=4, stride=4)'''

        T_lv3_unfold = self.bis(ref_lv3_unfold, 2, R_lv3_merged_arg)
        T_lv2_unfold = self.bis(ref_lv2_unfold, 2, R_lv3_merged_arg)
        T_lv1_unfold = self.bis(ref_lv1_unfold, 2, R_lv3_merged_arg)

        T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3,3), padding=1) / (3.*3.)
        T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv3.size(2)*2, lrsr_lv3.size(3)*2), kernel_size=(6,6), padding=2, stride=2) / (3.*3.)
        T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv3.size(2)*4, lrsr_lv3.size(3)*4), kernel_size=(12,12), padding=4, stride=4) / (3.*3.)
        
        S1 = R_lv1_merged.view(R_lv1_merged.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3)) 
        S2 = R_lv2_merged.view(R_lv2_merged.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3))              
        S3 = R_lv3_merged.view(R_lv3_merged.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3))
        R_lv3_merged_arg = R_lv3_merged_arg.view(R_lv3_merged_arg.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3))
        return S3, S2, S1, R_lv3_merged_arg, T_lv3, T_lv2, T_lv1
