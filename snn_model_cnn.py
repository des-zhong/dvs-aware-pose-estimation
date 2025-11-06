import torch
import torch.nn as nn
import torch.nn.functional as F
from uuid import uuid1
# from wrap_load_save import rand


class lynchip_state_wrapper():
    def __init__(self, mem, on_apu):
        self.mem = mem
        self.id = uuid1()
        self.on_apu = on_apu

    @staticmethod
    def lynload(tensor, uuid):
        from torch import ops
        from custom_op_in_lyn.custom_op_load_save import load
        ops.load_library('./custom_op_in_pytorch/build/libcustom_ops.so')
        from wrap_load_save import load
        return load(tensor, f'{uuid}')

    @staticmethod
    def lynsave(tensor, uuid):
        from torch import ops
        from custom_op_in_lyn.custom_op_load_save import save
        ops.load_library('./custom_op_in_pytorch/build/libcustom_ops.so')
        from wrap_load_save import save
        save(tensor, f'{uuid}')

    def get(self):
        if not self.on_apu:
            return self.mem
        else:
            return self.lynload(self.mem.clone(), self.id)

    def set(self, mem):
        if not self.on_apu:
            self.mem = mem
        else:
            self.lynsave(mem.clone(), self.id)

class Spike_Act_with_Surrogate_Gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, vth):
        ctx.save_for_backward(input)
        ctx.vth = vth
        return input.gt(vth).float()

    @staticmethod
    def backward(ctx, grad_output):
        lens = 0.5
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - ctx.vth) < lens
        return grad_input * temp.float() / (lens * 2), None

class Spiking_Network_CNN(nn.Module):
    def __init__(self, img_size, num_channels, vth, decay, activation, on_apu, init_batch=1):
        super().__init__()
        self.Spike_Act = activation
        self.on_apu = on_apu
        
        self.num_channels = num_channels
        self.img_size = img_size

        self.vth = nn.Parameter(vth * torch.ones(1), requires_grad=False)
        self.decay = nn.Parameter(decay * torch.ones(1), requires_grad=False)

        # self.conv1 = nn.Conv2d(1, self.num_channels, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, padding=1)
        # self.fc = nn.Linear(self.num_channels * (img_size // 4) ** 2, self.num_classes)

        # SCNN
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=0, bias=False)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=0, bias=False)

        # SFC
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)
        
        self.init_mem(batch_size=init_batch)

    # def init_mem(self):
    #     self.mem1 = lynchip_state_wrapper(torch.zeros(1, self.num_channels, self.img_size, self.img_size), self.on_apu)
    #     self.mem2 = lynchip_state_wrapper(torch.zeros(1, self.num_channels, self.img_size // 2, self.img_size // 2), self.on_apu)
    #     self.mem_fc = lynchip_state_wrapper(torch.zeros(1, self.num_classes), self.on_apu)

    def init_mem(self, batch_size=1):
        # adjust init memeroy unit dynamicly base on batch_size
        self.mem = [
            lynchip_state_wrapper(torch.zeros(batch_size, 16, 99, 99), self.on_apu),  # After conv1 + pool
            lynchip_state_wrapper(torch.zeros(batch_size, 16, 47, 47), self.on_apu),  # After conv2 + pool
            lynchip_state_wrapper(torch.zeros(batch_size, 32, 21, 21), self.on_apu),  # After conv3 + pool
            lynchip_state_wrapper(torch.zeros(batch_size, 32, 8, 8), self.on_apu),    # After conv4 + pool
            lynchip_state_wrapper(torch.zeros(batch_size, 256), self.on_apu),         # Before fc2
            lynchip_state_wrapper(torch.zeros(batch_size, 128), self.on_apu),         # Before fc3
            lynchip_state_wrapper(torch.zeros(batch_size, 3), self.on_apu)            # Final output
        ]
            
    # def mem_update(self, operator, inputs, mem, vth, decay): # IF OLD
    #     mem_ = mem.get().to(inputs.device)
    #     last_spike = self.Spike_Act(mem_, vth).float()
    #     state = operator(inputs)
    #     mem_ = mem_ * (1 - last_spike) * decay + state
    #     spike_out = self.Spike_Act(mem_, vth).float()
    #     mem.set(mem_)
    #     # 考虑mem是否被reset，并且将最终Mem传出
    #     return spike_out
    

    # def mem_update(self, op, x, idx, return_mem=False):
    #     mem_wrapper = self.mem[idx]
    #     mem = mem_wrapper.get().to(x.device)
        
    #     spk = self.Spike_Act(mem, self.vth).float()
    #     new_mem = mem * (1 - spk) * self.decay + op(x)
    #     mem_wrapper.set(new_mem)
    #     if return_mem:
    #         return new_mem
    #     return spk
    
    # def mem_update(self, op, x, idx, return_mem=False):
    #     spike_scale = 4.5
    #     mem_wrapper = self.mem[idx]
    #     mem = mem_wrapper.get().to(x.device)
        
    #     spk = self.Spike_Act(mem, self.vth).float()
    #     # print('spk:\n','max:',torch.max(spk).item(), '\n min:',torch.min(spk).item(),'\n')
    #     # print('op:\n','max:',torch.max(op(x)).item(), '\n min:',torch.min(op(x)).item(),'\n')
    #     tmp1 = mem * self.decay*spike_scale
    #     tmp2 = mem * spk * self.decay * spike_scale
    #     tmp3 = op(x) * spike_scale
    #     tmp4 = torch.round(tmp1-tmp2+tmp3)
    #     new_mem = tmp4/spike_scale

    #     mem_wrapper.set(new_mem)
    #     if return_mem:
    #         return new_mem
    #     return spk

    
    def mem_update(self, op, x, idx, return_mem=False):
        mem_wrapper = self.mem[idx]
        mem = mem_wrapper.get().to(x.device)
        # print(mem)
        spk = self.Spike_Act(mem, self.vth).float()
        if return_mem:
            new_mem = mem * (1 - spk) * self.decay + op(x)
        else:
            new_mem = mem * (1 - spk) * self.decay*8 + op(x)
        mem_wrapper.set(new_mem)
        if return_mem:
            return new_mem
        return spk


    # def forward(self, x):
    #     if self.on_apu:
    #         rr = rand(x[0,0,0,0], x.size(), 0x4000, mode=0)
    #         x = (rr < x).float()
    #     else:
    #         x = torch.bernoulli(x)
        
    #     x = self.mem_update(self.conv1, x, self.mem1, self.vth, self.decay)
    #     x = nn.functional.max_pool2d(x, 2)
    #     x = self.mem_update(self.conv2, x, self.mem2, self.vth, self.decay)
    #     x = nn.functional.max_pool2d(x, 2)
    #     x = x.view(x.size(0), -1)
    #     x = self.mem_update(self.fc, x, self.mem_fc, self.vth, self.decay)
    #     return x
    
    # def forward(self, x):
    #     if self.on_apu:
    #         rr = rand(x[0,0,0,0], x.size(), 0x4000, mode=0)
    #         x = (rr < x).float()
    #     else:
    #         x = x.unsqueeze(0) if x.dim() == 3 else x
    #         assert x.min() >= 0 and x.max() <= 1
    #         x = torch.bernoulli(x)
            
    #     x = self.mem_update(self.conv1, x, 0)
    #     x = F.max_pool2d(x, 2)
    #     x = self.mem_update(self.conv2, x, 1)
    #     x = F.max_pool2d(x, 2)
    #     x = self.mem_update(self.conv3, x, 2)
    #     x = F.max_pool2d(x, 2)
    #     x = self.mem_update(self.conv4, x, 3)
    #     x = F.max_pool2d(x, 2)
        
    #     x = x.view(x.size(0), -1)
    #     x = self.mem_update(self.fc1, x, 4)
    #     x = self.mem_update(self.fc2, x, 5)
    #     x = self.mem_update(self.fc3, x, 6, return_mem=True)
    #     return x
    
    def forward(self, x):
        # if self.on_apu:
        #     rr = rand(x[0,0,0,0], x.size(), 0x4000, mode=0)
        #     x = (rr < x).float()
        # else:
        #     assert x.min() >= 0 and x.max() <= 1
        #     x = torch.bernoulli(x)
        
        x = self.mem_update(self.conv1, x, 0)
        x = F.max_pool2d(x, 2)
        x = self.mem_update(self.conv2, x, 1)
        x = F.max_pool2d(x, 2)
        x = self.mem_update(self.conv3, x, 2)
        x = F.max_pool2d(x, 2)
        x = self.mem_update(self.conv4, x, 3)
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1)
        x = self.mem_update(self.fc1, x, 4)
        x = self.mem_update(self.fc2, x, 5)
        x = self.mem_update(self.fc3, x, 6, return_mem=True)
        return x