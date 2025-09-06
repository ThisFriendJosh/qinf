from __future__ import annotations
import torch, torch.nn.functional as F

class QLearner:
    def __init__(self, qnet, target_qnet, optim, replay, gamma=0.99, n_step=1, double_q=True):
        self.q, self.tgt, self.opt = qnet, target_qnet, optim
        self.replay, self.gamma, self.n_step, self.double_q = replay, gamma, n_step, double_q
        self.step_i = 0
    def step(self, batch):
        import numpy as np
        s, a, r, s2, d = zip(*batch)
        s = torch.tensor(np.stack(s)).float()
        a = torch.tensor(a).long().unsqueeze(1)
        r = torch.tensor(r).float().unsqueeze(1)
        s2 = torch.tensor(np.stack(s2)).float()
        d = torch.tensor(d).float().unsqueeze(1)
        qsa = self.q(s).gather(1, a)
        with torch.no_grad():
            if self.double_q:
                a2 = torch.argmax(self.q(s2), dim=1, keepdim=True)
                q_next = self.tgt(s2).gather(1, a2)
            else:
                q_next, _ = torch.max(self.tgt(s2), dim=1, keepdim=True)
            target = r + (1.0 - d) * (self.gamma ** self.n_step) * q_next
        loss = F.smooth_l1_loss(qsa, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.step_i += 1
        return {"loss": float(loss.item())}
