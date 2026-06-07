

import torch
import torch.nn as nn
from copy import deepcopy


class EMA:
    
    def __init__(
        self, 
        model: nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 100,
        update_every: int = 10
    ):
        self.model = model
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        
        self.shadow = deepcopy(model)
        
        for param in self.shadow.parameters():
            param.requires_grad = False
        
        self.num_updates = 0
        
        self.backup = {}
    
    def update(self, model: nn.Module = None):
        
        self.num_updates += 1
        
        if self.num_updates < self.update_after_step:
            return
        
        if self.num_updates % self.update_every != 0:
            return
        
        if model is None:
            model = self.model
        
        with torch.no_grad():
            model_params = dict(model.named_parameters())
            shadow_params = dict(self.shadow.named_parameters())
            
            for name in model_params.keys():
                if name in shadow_params:
                    # shadow = decay * shadow + (1 - decay) * model
                    shadow_params[name].data.mul_(self.decay).add_(
                        model_params[name].data, alpha=1 - self.decay
                    )
            
            model_buffers = dict(model.named_buffers())
            shadow_buffers = dict(self.shadow.named_buffers())
            
            for name in model_buffers.keys():
                if name in shadow_buffers:
                    shadow_buffers[name].data.copy_(model_buffers[name].data)
    
    @torch.no_grad()
    def apply_shadow(self):
        
        self.backup = {}
        for name, param in self.model.named_parameters():
            self.backup[name] = param.data.clone()
        
        shadow_params = dict(self.shadow.named_parameters())
        for name, param in self.model.named_parameters():
            if name in shadow_params:
                param.data.copy_(shadow_params[name].data)
    
    @torch.no_grad()
    def restore(self):
        
        if not self.backup:
            return
        
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        
        self.backup = {}
    
    def state_dict(self):
        
        return {
            'shadow': self.shadow.state_dict(),
            'num_updates': self.num_updates,
            'decay': self.decay
        }
    
    def load_state_dict(self, state_dict):
        
        self.shadow.load_state_dict(state_dict['shadow'])
        self.num_updates = state_dict.get('num_updates', 0)
        self.decay = state_dict.get('decay', self.decay)
    
    def to(self, device):
        
        self.shadow = self.shadow.to(device)
        return self


if __name__ == '__main__':
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)
        
        def forward(self, x):
            return self.fc(x)
    
    model = SimpleModel()
    ema = EMA(model, decay=0.999, update_after_step=10, update_every=5)
    
    print("=" * 60)
    print("EMA test")
    print("=" * 60)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for step in range(1, 101):
        x = torch.randn(2, 10)
        y = model(x)
        loss = y.mean()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        ema.update(model)
        
        if step % 20 == 0:
            orig_weight = model.fc.weight.data[0, 0].item()
            shadow_weight = ema.shadow.fc.weight.data[0, 0].item()
            print(f"Step {step}: Original={orig_weight:.6f}, EMA={shadow_weight:.6f}, Diff={abs(orig_weight - shadow_weight):.6f}")
    
    print("\n" + "=" * 60)
    print("Testing apply_shadow() and restore()")
    print("=" * 60)
    
    orig_weight_before = model.fc.weight.data[0, 0].item()
    print(f"Original weight before apply: {orig_weight_before:.6f}")
    
    ema.apply_shadow()
    ema_weight = model.fc.weight.data[0, 0].item()
    print(f"EMA weight after apply: {ema_weight:.6f}")
    
    ema.restore()
    orig_weight_after = model.fc.weight.data[0, 0].item()
    print(f"Original weight after restore: {orig_weight_after:.6f}")
    
    assert abs(orig_weight_before - orig_weight_after) < 1e-6, "Restore failed"
    print("\nEMA test passed")
    
    print("\n" + "=" * 60)
    print("Testing state_dict() and load_state_dict()")
    print("=" * 60)
    
    ema_state = ema.state_dict()
    print(f"Saved EMA state: num_updates={ema_state['num_updates']}")
    
    ema_new = EMA(SimpleModel(), decay=0.999)
    ema_new.load_state_dict(ema_state)
    print(f"Loaded EMA state: num_updates={ema_new.num_updates}")
    
    diff = (ema.shadow.fc.weight.data - ema_new.shadow.fc.weight.data).abs().max().item()
    print(f"Weight difference: {diff:.10f}")
    assert diff < 1e-6, "Load failed"
    
    print("\nAll tests passed")
