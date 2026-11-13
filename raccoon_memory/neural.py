"""Two learned components: recurrent recall and a next-action classifier."""
import torch
from torch import nn

from .world import N_ACTIONS, N_FEATURES, N_TOKENS


class EpisodicMemory(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.recurrent = nn.GRU(N_FEATURES+1, hidden_size, batch_first=True)
        self.decoder = nn.Linear(hidden_size, N_FEATURES)

    def forward(self, sequence, hidden=None):
        output, hidden = self.recurrent(sequence, hidden)
        return self.decoder(output).reshape(*output.shape[:-1], N_TOKENS, 5), hidden


class ActionPolicy(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.width = width
        self.network = nn.Sequential(
            nn.Linear(N_FEATURES*4, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
        )
        self.action_head = nn.Linear(width, N_ACTIONS)
        self.value_head = nn.Sequential(nn.Linear(width, 1), nn.Softplus())

    def predict(self, current, goal):
        encoded = self.network(torch.cat((current, goal, current-goal, current*goal), dim=-1))
        return self.action_head(encoded), self.value_head(encoded).squeeze(-1)

    def forward(self, current, goal, allowed=None):
        raw, _ = self.predict(current, goal)
        return raw if allowed is None else raw.masked_fill(~allowed, -1e9)


class RelationalActionPolicy(nn.Module):
    """One shared scorer for every source/destination pair.

    Equality, height and support relations are observations, not action labels.
    No object identity or slot number has a dedicated learned embedding.
    """
    def __init__(self, width=128):
        super().__init__()
        self.width = width
        self.scorer = nn.Sequential(nn.Linear(24, width), nn.ReLU(),
                                    nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
        self.terminal = nn.Sequential(nn.Linear(8, width), nn.ReLU(), nn.Linear(width, 2))
        self.value_head = nn.Sequential(nn.Linear(8, width), nn.ReLU(), nn.Linear(width, 1), nn.Softplus())
        self.register_buffer("sources", torch.tensor([s for s in range(4) for d in range(4) if s != d]))
        self.register_buffer("destinations", torch.tensor([d for s in range(4) for d in range(4) if s != d]))

    def relations(self, current, goal):
        c = current.reshape(-1, 4, 4, 5)
        g = goal.reshape(-1, 4, 4, 5)
        occupied, required = 1-c[...,0], 1-g[...,0]
        height, goal_height = occupied.sum(-1), required.sum(-1)
        matches = (c*g).sum(-1)*occupied
        prefix = matches.cumprod(-1).sum(-1)
        top = c.gather(2, (height.long()-1).clamp(min=0)[:,:,None,None].expand(-1,-1,1,5)).squeeze(2)[...,1:]
        top = top*(height>0).unsqueeze(-1)
        gobjects = g[...,1:]
        goal_slot = torch.einsum("bsi,bdli->bsd", top, gobjects)
        levels = torch.arange(4, device=c.device, dtype=c.dtype)
        goal_level = torch.einsum("bsi,bdli,l->bs", top, gobjects, levels)
        # Required object immediately below each moved object, or zeros at base.
        below = torch.cat((torch.zeros_like(gobjects[:,:,:1]), gobjects[:,:,:-1]), dim=2)
        needed_support = torch.einsum("bsi,bdli,bdlj->bsj", top, gobjects, below)
        support_match = torch.einsum("bsi,bdi->bsd", needed_support, top)
        at_base = (goal_level==0).to(c.dtype)
        global_features = torch.stack((prefix.sum(-1)/4, matches.sum((1,2))/4,
            (height==0).sum(-1)/4, height.max(-1).values/4,
            goal_height.max(-1).values/4, (height==goal_height).sum(-1)/4,
            (current*goal).sum(-1)/16, (current==goal).all(-1).to(c.dtype)), -1)
        s, d = self.sources, self.destinations
        batch = torch.arange(len(c), device=c.device)[:,None]
        src_goal = goal_slot[:,s,s]
        dst_goal = goal_slot[:,s,d]
        f = torch.stack((height[:,s]/4, height[:,d]/4, goal_height[:,s]/4, goal_height[:,d]/4,
            prefix[:,s]/4, prefix[:,d]/4, src_goal, dst_goal, goal_level[:,s]/4,
            ((height[:,s]==prefix[:,s]) & (height[:,s]>0)).to(c.dtype),
            (height[:,d]==prefix[:,d]).to(c.dtype),
            (height[:,d]==goal_level[:,s]).to(c.dtype),
            support_match[batch,s,d], at_base[:,s],
            (height[:,s]-prefix[:,s])/4, (height[:,d]-prefix[:,d])/4,
            *(global_features[:,i,None].expand(-1,12) for i in range(8))), -1)
        return f, global_features

    def predict(self, current, goal):
        relations, global_features = self.relations(current, goal)
        logits = torch.cat((self.scorer(relations).squeeze(-1), self.terminal(global_features)), -1)
        return logits, self.value_head(global_features).squeeze(-1)

    def forward(self, current, goal, allowed=None):
        raw, _ = self.predict(current, goal)
        return raw if allowed is None else raw.masked_fill(~allowed, -1e9)


def load_model(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    classes = {"action_policy": ActionPolicy, "relational_policy": RelationalActionPolicy,
               "episodic_memory": EpisodicMemory}
    if payload.get("schema") != "numbered_slots_v1" or payload.get("kind") not in classes:
        raise ValueError("Unsupported model artifact")
    model = classes[payload["kind"]](**payload["config"]).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())
