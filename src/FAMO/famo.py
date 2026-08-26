# Vendored from https://github.com/Cranial-XIX/FAMO (MIT License, Copyright (c) 2023 Bo Liu)
# B. Liu, Y. Feng, P. Stone, and Q. Liu, "FAMO: Fast Adaptive Multitask Optimization,"
# NeurIPS 2023. Unmodified from upstream; see THIRD_PARTY_NOTICES.md for the full license text.

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union


class FAMO:
    """
    Fast Adaptive Multitask Optimization.
    """
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        gamma: float = 0.01,   # the regularization coefficient
        w_lr: float = 0.025,   # the learning rate of the task logits
        max_norm: float = 1.0, # the maximum gradient norm
    ):
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        self.n_tasks = n_tasks
        self.device = device
    
    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses):
        if not hasattr(self, "_min_inited") or not self._min_inited:
            with torch.no_grad():
                k = losses.shape[0]
                self.min_losses[:k] = losses.detach()
                self._min_inited = True

        self.prev_loss = losses
        # Ensure min_losses is on the same device as losses
        if self.min_losses.device != losses.device:
            self.min_losses = self.min_losses.to(losses.device)
        # Ensure w is on the same device as losses and update optimizer
        if self.w.device != losses.device:
            # Create new tensor on target device
            new_w = self.w.detach().to(losses.device)
            new_w.requires_grad_(True) 
            # Create new optimizer for the new tensor
            old_state = None
            if len(self.w_opt.state) > 0:
                old_state = self.w_opt.state[self.w]
            self.w = new_w
            self.w_opt = torch.optim.Adam([self.w], lr=self.w_opt.param_groups[0]['lr'], weight_decay=self.w_opt.param_groups[0]['weight_decay'])
            # Transfer optimizer state if it exists
            if old_state is not None:
                self.w_opt.state[self.w] = {
                    key: value.to(losses.device) if torch.is_tensor(value) else value
                    for key, value in old_state.items()
                }
        

        k = losses.shape[0]
        z_all = F.softmax(self.w, -1)
        z = z_all[:k]                                # (k,)
        D = losses - self.min_losses[:k] + 1e-8      # (k,)
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss, z

    def update(self, curr_loss):
        # Ensure curr_loss is on the same device as prev_loss
        if curr_loss.device != self.prev_loss.device:
            curr_loss = curr_loss.to(self.prev_loss.device)
        # Ensure min_losses is on the same device
        if self.min_losses.device != self.prev_loss.device:
            self.min_losses = self.min_losses.to(self.prev_loss.device)
        # Ensure w is on the same device and update optimizer if needed
        if self.w.device != self.prev_loss.device:
            new_w = self.w.detach().to(self.prev_loss.device)   # <<< detach
            new_w.requires_grad_(True)   
            old_state = None
            if len(self.w_opt.state) > 0:
                old_state = self.w_opt.state[self.w]
            self.w = new_w
            self.w_opt = torch.optim.Adam([self.w], lr=self.w_opt.param_groups[0]['lr'], weight_decay=self.w_opt.param_groups[0]['weight_decay'])
            # Transfer optimizer state if it exists
            if old_state is not None:
                self.w_opt.state[self.w] = {
                    key: value.to(self.prev_loss.device) if torch.is_tensor(value) else value
                    for key, value in old_state.items()
                }
            
        k = self.prev_loss.shape[0]
        delta = (self.prev_loss[:k] - self.min_losses[:k] + 1e-8).log() - \
                (curr_loss[:k]      - self.min_losses[:k] + 1e-8).log()

        z_all = F.softmax(self.w, -1)  # shape: (n_tasks,)
        if k < self.w.numel():
            grad_out = torch.cat([delta, delta.new_zeros(self.w.numel() - k)]) if k < self.w.numel() else delta
        with torch.enable_grad():
            d = torch.autograd.grad(z_all,  # gradient only for first k logits
                                    self.w,
                                    grad_outputs=grad_out, retain_graph=False, create_graph=False)[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
    ) -> Union[torch.Tensor, None]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        Returns
        -------
        Loss, extra outputs
        """
        loss = self.get_weighted_loss(losses=losses)
        self.manual_backward(loss)
        if self.max_norm > 0 and shared_parameters is not None:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return loss
    
    def compute_meta_loss(self, train_losses, val_losses):
        if isinstance(train_losses, list):
            train_losses = torch.tensor(train_losses, device=self.device)
        if isinstance(val_losses, list):
            val_losses = torch.tensor(val_losses, device=self.device)
        k = min(train_losses.shape[0], val_losses.shape[0], self.w.numel())
        if self.w.device != self.device:
            self.w = self.w.detach().to(self.device).requires_grad_(True)

        z = F.softmax(self.w, -1)[:k]
        return (z * val_losses[:k]).sum()


if __name__ == "__main__":

    n   = 1000 # number of datapoints
    dim = 20   # dimension of data
    K   = 100  # number of tasks
    X = torch.randn(n, dim)
    Y = torch.randn(n, K)

    model = torch.nn.Linear(dim, K)
    weight_opt = FAMO(n_tasks=K, device="cpu")
    opt = torch.optim.Adam(model.parameters())

    for it in range(100):
        loss = (Y - model(X)).pow(2).mean(0) # (K,)
        opt.zero_grad()
        weight_opt.backward(loss)
        opt.step()
        # update the task weighting
        with torch.no_grad():
            new_losses = {
                'rgb': rgb_total,
                'depth': total_geometry_loss,
                'semantics': loss_semantics,
                'segment': loss_segment_clustering,
                'instance': loss_instance_clustering
            }
            new_loss_tensor = torch.stack(list(new_losses.values()))
            weight_opt.update(new_loss_tensor)
        print(f"[info] iter {it:3d} | avg loss {loss.mean().item():.4f}")
