"""Output-layer policy-gradient primitives for the fixed FK200 experiment.

This module does not run a simulator, collect a physical outcome, or start
training on import.  ``SamplingPolicy`` fits the existing guarded executor:
its score tensor merely makes the executor select the sampled action.  The
original distribution, unprotected sample, and confirmed applied control are
recorded separately.  Training replays the entire public history instead of
backpropagating through the executor's detached recurrent state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from raccoon_control.kinematics import JOINT_LIMITS
from . import motion_policy, scene_boundary_recurrent_pair as paired
from .scene_causal_frame_policy import CONTEXT_COUNT, context


ROOT = Path(__file__).resolve().parents[1]
FK200_PATH = ROOT / "artifacts/architecture_research/20261103_gru_nominal_fk_candidate_seed20261101_v1/transfer_reset.pt"
FK200_SHA256 = "a184b7a7f387eac1fff976f39f796e706b55c056ff2c6227e86e52d8401bea72"
TRAJECTORY_FORMAT = "fk200-output-rl-trajectory.v1"
SIGMA_DEG = tuple(min(1.0, .005 * (hi - lo)) for lo, hi in JOINT_LIMITS)


def _finite_vector(values, length, name):
    if (not isinstance(values, (tuple, list)) or len(values) != length
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return tuple(float(x) for x in values)


def _model_contract(model):
    if (not isinstance(model, paired.BoundaryRecurrentPolicy) or model.mode != "transfer_reset"
            or model.horizon != 1 or model.output_space != "joint"
            or not isinstance(model.output, nn.Linear) or model.output.weight.shape != (8, 64)
            or model.output.bias.shape != (8,) or next(model.parameters()).device.type != "cpu"):
        raise ValueError("the CPU FK200 transfer-reset GRU and its Linear(64,8) are required")


def configure_output_only(model):
    """Freeze the whole model except its 520 output weight and bias values."""
    _model_contract(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.output.parameters():
        parameter.requires_grad_(True)
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != 520:
        raise ValueError("unexpected trainable parameter count")
    return model.eval()


def state_digest(model):
    """Fingerprint all parameters and buffers, independently of GRU runtime state."""
    _model_contract(model)
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite policy state")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_fk200_checkpoint(path=FK200_PATH):
    """Load only the fingerprinted FK200 checkpoint; no optimizer is restored."""
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != FK200_SHA256:
        raise ValueError("FK200 checkpoint fingerprint differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]):
        model = configure_output_only(paired.load_checkpoint(payload))
    model._fk200_initial_digest = state_digest(model)
    return model


@dataclass(frozen=True)
class PublicHistory:
    """Unpadded 51-dimensional public features and previous applied controls.

    ``features`` must be the existing polar ``context`` of public robot/task
    observations, not simulator object positions, contacts, or success labels.
    The first row and each subsequent transfer have ``transfer_start=True``.
    """
    features: torch.Tensor
    previous: torch.Tensor
    transfer_start: torch.Tensor

    def __post_init__(self):
        f, p, start = self.features, self.previous, self.transfer_start
        if (not isinstance(f, torch.Tensor) or f.ndim != 2 or f.shape[1] != CONTEXT_COUNT
                or not len(f) or f.dtype != torch.float32 or f.device.type != "cpu"
                or not isinstance(p, torch.Tensor) or p.shape != (len(f), 4)
                or p.dtype != torch.float32 or p.device.type != "cpu"
                or not isinstance(start, torch.Tensor) or start.shape != (len(f),)
                or start.dtype != torch.bool or start.device.type != "cpu" or not bool(start[0])
                or not bool(torch.isfinite(f).all() and torch.isfinite(p).all())):
            raise ValueError("a finite, nonempty CPU public history with transfer boundaries is required")


def history_from_public_rows(rows, previous, transfer_start):
    """Build model inputs from the executor's 82-value public observation rows."""
    with torch.inference_mode(False):
        rows = torch.as_tensor(rows, dtype=torch.float32, device="cpu").clone()
        previous = torch.as_tensor(previous, dtype=torch.float32, device="cpu").clone()
        starts = torch.as_tensor(transfer_start, dtype=torch.bool, device="cpu").clone()
        if rows.ndim != 2 or rows.shape[1] != 82 or previous.shape != (len(rows), 4):
            raise ValueError("public rows and applied targets must have shapes [T,82] and [T,4]")
        features = context(rows, previous, "polar").to(torch.float32)
        return PublicHistory(features, previous, starts)


def histories_from_padded(features, previous, valid, transfer_start):
    """Convert existing direct-demonstration public batches to unpadded histories."""
    if (not isinstance(features, torch.Tensor) or features.ndim != 3
            or features.shape[-1] != CONTEXT_COUNT or features.dtype != torch.float32
            or not isinstance(previous, torch.Tensor) or previous.shape != (*features.shape[:2], 4)
            or previous.dtype != torch.float32 or not isinstance(valid, torch.Tensor)
            or valid.shape != features.shape[:2] or valid.dtype != torch.bool
            or not isinstance(transfer_start, torch.Tensor) or transfer_start.shape != valid.shape
            or transfer_start.dtype != torch.bool or features.device.type != "cpu"
            or previous.device.type != "cpu" or valid.device.type != "cpu" or transfer_start.device.type != "cpu"
            or not len(features) or not bool(valid[:, 0].all())
            or bool(((~valid[:, :-1]) & valid[:, 1:]).any())
            or bool((transfer_start & ~valid).any())):
        raise ValueError("nonempty right-padded public demonstration histories are required")
    return tuple(PublicHistory(features[i, :int(mask.sum())].detach().clone(),
                               previous[i, :int(mask.sum())].detach().clone(),
                               transfer_start[i, :int(mask.sum())].detach().clone())
                 for i, mask in enumerate(valid))


@dataclass(frozen=True)
class Decision:
    public_row: tuple[float, ...]
    previous_applied_deg: tuple[float, ...]
    transfer_start: bool
    valid_history_count: int
    action: int
    sampled_move_deg: tuple[float, ...] | None
    collection_log_prob: float
    mode_logits: tuple[float, ...]
    move_mean_deg: tuple[float, ...]
    applied_target_deg: tuple[float, ...] | None
    command_status: str


@dataclass(frozen=True)
class Trajectory:
    policy_sha256: str
    decisions: tuple[Decision, ...]
    reward: int | None

    def to_dict(self):
        _validate_trajectory(self)
        return {"format": TRAJECTORY_FORMAT, "policy_sha256": self.policy_sha256,
                "decisions": [asdict(item) for item in self.decisions], "reward": self.reward}

    @classmethod
    def from_dict(cls, payload):
        if (not isinstance(payload, dict) or set(payload) != {"format", "policy_sha256", "decisions", "reward"}
                or payload["format"] != TRAJECTORY_FORMAT or not isinstance(payload["decisions"], list)):
            raise ValueError("invalid trajectory serialization")
        items = []
        fields = set(Decision.__dataclass_fields__)
        for raw in payload["decisions"]:
            if not isinstance(raw, dict) or set(raw) != fields:
                raise ValueError("invalid serialized decision")
            item = dict(raw)
            for name in ("public_row", "previous_applied_deg", "sampled_move_deg", "mode_logits",
                         "move_mean_deg", "applied_target_deg"):
                if item[name] is not None:
                    if not isinstance(item[name], (tuple, list)):
                        raise ValueError("invalid serialized decision vector")
                    item[name] = tuple(item[name])
            items.append(Decision(**item))
        result = cls(payload["policy_sha256"], tuple(items), payload["reward"])
        _validate_trajectory(result)
        return result


def _validate_trajectory(trajectory):
    if (not isinstance(trajectory, Trajectory) or not isinstance(trajectory.policy_sha256, str)
            or len(trajectory.policy_sha256) != 64
            or any(c not in "0123456789abcdef" for c in trajectory.policy_sha256)
            or not isinstance(trajectory.decisions, tuple)
            or (trajectory.reward is not None and (type(trajectory.reward) is not int or trajectory.reward not in (0, 1)))):
        raise ValueError("a complete on-policy trajectory and binary posthoc reward are required")
    if not trajectory.decisions and trajectory.reward is not None:
        raise ValueError("a scored trajectory must contain at least one decision")
    count, expected_previous, ended = 0, None, True
    for index, item in enumerate(trajectory.decisions):
        if not isinstance(item, Decision) or type(item.transfer_start) is not bool:
            raise ValueError("invalid recorded decision")
        _finite_vector(item.public_row, 82, "public observation")
        previous = _finite_vector(item.previous_applied_deg, 4, "previous applied target")
        if any(not lo - 3e-5 <= q <= hi + 3e-5 for q, (lo, hi) in zip(previous, JOINT_LIMITS)):
            raise ValueError("previous applied target is outside the joint limits")
        _finite_vector(item.mode_logits, 4, "mode logits")
        _finite_vector(item.move_mean_deg, 4, "MOVE mean")
        if expected_previous is not None and any(abs(a-b) > 3e-5 for a, b in zip(previous, expected_previous)):
            raise ValueError("previous input differs from confirmed applied control")
        if item.transfer_start:
            if not ended:
                raise ValueError("a new transfer must follow DONE")
            count, ended = 1, False
        else:
            if ended or count == 0:
                raise ValueError("missing transfer start")
            count = min(count + 1, 8)
        if (type(item.valid_history_count) is not int or item.valid_history_count != count
                or type(item.action) is not int or item.action not in range(4)
                or type(item.collection_log_prob) not in (int, float) or not math.isfinite(item.collection_log_prob)):
            raise ValueError("invalid history, action, or collected log probability")
        if item.action == motion_policy.MOVE:
            _finite_vector(item.sampled_move_deg, 4, "unprotected MOVE sample")
        elif item.sampled_move_deg is not None:
            raise ValueError("only MOVE has a Gaussian sample")
        if item.action == motion_policy.DONE:
            if item.command_status != "done" or item.applied_target_deg is not None:
                raise ValueError("DONE cannot apply a command")
            expected_previous, ended = previous, True
        elif item.command_status == "applied":
            expected_previous = _finite_vector(item.applied_target_deg, 4, "applied target")
            if any(not lo <= q <= hi for q, (lo, hi) in zip(expected_previous, JOINT_LIMITS)):
                raise ValueError("confirmed applied target is outside the joint limits")
            if item.action != motion_policy.MOVE and any(abs(a-b) > 3e-5 for a, b in zip(previous, expected_previous)):
                raise ValueError("non-MOVE cannot change applied joints")
        elif item.command_status in ("no_command", "pending"):
            if item.applied_target_deg is not None:
                raise ValueError("an unapplied command cannot report an applied target")
            if item.command_status == "pending" and trajectory.reward is not None:
                raise ValueError("a scored trajectory cannot contain an unconfirmed command")
            if index != len(trajectory.decisions) - 1:
                raise ValueError("an unapplied command must end the rollout")
            expected_previous = previous
        else:
            raise ValueError("unknown command status")
    return trajectory


def trajectory_history(trajectory):
    _validate_trajectory(trajectory)
    return history_from_public_rows([x.public_row for x in trajectory.decisions],
                                    [x.previous_applied_deg for x in trajectory.decisions],
                                    [x.transfer_start for x in trajectory.decisions])


def _predict(model, history):
    _model_contract(model)
    if not isinstance(history, PublicHistory):
        raise ValueError("a public history is required")
    # PublicHistory is frozen, but its tensors are mutable; recheck at use.
    PublicHistory(history.features, history.previous, history.transfer_start)
    valid = torch.ones((1, len(history.features)), dtype=torch.bool)
    logits, angles = model.forward_padded(history.features[None], history.previous[None],
                                          valid, history.transfer_start[None])
    return logits[0], angles[0] * 180.0


class SamplingPolicy(nn.Module):
    """One registration of on-policy samples for the unchanged guarded executor."""
    history_length, feature_count = 8, 82

    def __init__(self, model, *, generator):
        super().__init__()
        _model_contract(model)
        if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
            raise ValueError("an explicit CPU exploration generator is required")
        self.model, self.generator = model.eval(), generator
        self.policy_sha256 = state_digest(model)
        self._registered = self._active = self._first = self._ended = self._finished = self._aborted = False
        self._decisions = []
        self._expected_target = None
        self._history_count = 0

    def _check_weights(self):
        if state_digest(self.model) != self.policy_sha256:
            raise ValueError("policy weights changed during rollout collection")

    def on_scene_registered(self, _scene, _nominal_slot_xy_mm):
        if self._registered or self._finished:
            raise ValueError("exactly one registration is permitted per rollout")
        self._check_weights()
        self.model.begin_registration()
        self._registered = True

    def on_transfer_started(self, _before, _task, applied_targets):
        if (not self._registered or self._finished or self._aborted or (self._active and not self._ended)
                or not isinstance(applied_targets, dict)):
            raise ValueError("a registered rollout and completed previous transfer are required")
        target = _finite_vector(applied_targets.get("joints_deg"), 4, "confirmed applied target")
        if any(not lo <= q <= hi for q, (lo, hi) in zip(target, JOINT_LIMITS)):
            raise ValueError("confirmed applied target is outside the joint limits")
        if self._expected_target is not None and any(abs(a-b) > 3e-5 for a, b in zip(target, self._expected_target)):
            raise ValueError("transfer began with an unexpected applied target")
        self._expected_target = target
        self._active, self._first, self._ended, self._history_count = True, True, False, 0

    @torch.no_grad()
    def forward_with_control_state(self, windows, masks, previous):
        self._check_weights()
        if (not self._active or self._ended or self._finished or self._aborted
                or (self._decisions and self._decisions[-1].command_status == "pending")):
            raise ValueError("a transfer and confirmed previous command are required")
        if (not isinstance(windows, torch.Tensor) or windows.shape != (1, 8, 82)
                or windows.dtype != torch.float32 or windows.device.type != "cpu"
                or not bool(torch.isfinite(windows).all()) or not isinstance(masks, torch.Tensor)
                or masks.shape != (1, 8) or masks.dtype != torch.bool or masks.device.type != "cpu"
                or not isinstance(previous, torch.Tensor) or previous.shape != (1, 4)
                or previous.dtype != torch.float32 or previous.device.type != "cpu"
                or not bool(torch.isfinite(previous).all())):
            raise ValueError("one CPU public history and applied target are required")
        self._history_count = 1 if self._first else min(8, self._history_count + 1)
        if masks[0].tolist() != [False] * (8-self._history_count) + [True] * self._history_count:
            raise ValueError("transfer history mask does not match its boundary")
        prior = tuple(previous[0].tolist())
        if any(abs(a-b) > 3e-5 for a, b in zip(prior, self._expected_target)):
            raise ValueError("the next input must use the confirmed applied target")
        row = tuple(windows[0, -1].tolist())
        history = history_from_public_rows([x.public_row for x in self._decisions] + [row],
                    [x.previous_applied_deg for x in self._decisions] + [prior],
                    [x.transfer_start for x in self._decisions] + [self._first])
        logits, means = _predict(self.model, history)
        logits, mean = logits[-1], means[-1]  # decode before the executor's argmax gate
        action = int(torch.multinomial(logits.softmax(-1), 1, generator=self.generator).item())
        logp = torch.log_softmax(logits, -1)[action]
        sample = None
        target = previous[0] / 180.0
        if action == motion_policy.MOVE:
            sigma = mean.new_tensor(SIGMA_DEG)
            draw = mean + sigma * torch.randn((4,), generator=self.generator, dtype=mean.dtype)
            logp = logp + torch.distributions.Normal(mean, sigma).log_prob(draw).sum()
            sample = tuple(draw.tolist())
            target = draw / 180.0
        status = "done" if action == motion_policy.DONE else "pending"
        self._decisions.append(Decision(row, prior, self._first, self._history_count, action, sample,
                               float(logp), tuple(logits.tolist()), tuple(mean.tolist()), None, status))
        self._first = False
        self._ended = action == motion_policy.DONE
        # The common executor selects argmax; these are selection scores, not
        # the model logits.  The latter remain in the recorded decision.
        selection = torch.full_like(logits, -1.0)
        selection[action] = 1.0
        return selection[None], target[None]

    def on_control_applied(self, action, transmitted_target_deg, applied_target_deg, _opening):
        self._check_weights()
        if self._finished or not self._decisions or self._decisions[-1].command_status != "pending":
            raise ValueError("no sampled command is awaiting confirmation")
        item = self._decisions[-1]
        sent = _finite_vector(transmitted_target_deg, 4, "transmitted target")
        actual = _finite_vector(applied_target_deg, 4, "applied target")
        if (action != motion_policy.ACTIONS[item.action]
                or any(abs(a-b) > 3e-5 for a, b in zip(sent, actual))
                or any(not lo <= q <= hi for q, (lo, hi) in zip(actual, JOINT_LIMITS))
                or (item.action != motion_policy.MOVE
                    and any(abs(a-b) > 3e-5 for a, b in zip(actual, item.previous_applied_deg)))):
            raise ValueError("applied control does not match the sampled command")
        self._decisions[-1] = replace(item, applied_target_deg=actual, command_status="applied")
        self._expected_target = actual

    def mark_no_command(self):
        """Record an executor-confirmed decision for which no command was sent."""
        self._check_weights()
        if self._finished or not self._decisions or self._decisions[-1].command_status != "pending":
            raise ValueError("no unconfirmed command exists")
        self._decisions[-1] = replace(self._decisions[-1], command_status="no_command")
        self._aborted = True

    def finish(self, *, reward):
        """Attach only the official final 0/1 score; None denotes an excluded error."""
        if self._finished or (not self._registered and reward is not None):
            raise ValueError("rollout is not active")
        self._check_weights()
        result = Trajectory(self.policy_sha256, tuple(self._decisions), reward)
        _validate_trajectory(result)
        self._finished = True
        return result


@dataclass(frozen=True)
class Replay:
    logits: torch.Tensor
    move_mean_deg: torch.Tensor
    decision_log_prob: torch.Tensor


def replay_trajectory(model, trajectory):
    """Recompute every decision differentiably from actual public history."""
    history = trajectory_history(trajectory)
    logits, means = _predict(model, history)
    actions = torch.tensor([x.action for x in trajectory.decisions], dtype=torch.long)
    logp = torch.log_softmax(logits, -1).gather(1, actions[:, None])[:, 0]
    indices = torch.nonzero(actions == motion_policy.MOVE, as_tuple=True)[0]
    if len(indices):
        draws = torch.tensor([trajectory.decisions[int(i)].sampled_move_deg for i in indices], dtype=means.dtype)
        gaussian = torch.distributions.Normal(means[indices], means.new_tensor(SIGMA_DEG)).log_prob(draws).sum(-1)
        logp = logp.index_add(0, indices, gaussian)
    return Replay(logits, means, logp)


def policy_kl(reference, candidate, histories):
    """Decision-weighted mean KL(reference || candidate) on public histories."""
    histories = tuple(histories)
    if not histories:
        raise ValueError("KL requires a nonempty public history set")
    total, count = 0.0, 0
    for history in histories:
        with torch.no_grad():
            ref_logits, ref_mean = _predict(reference, history)
        logits, mean = _predict(candidate, history)
        ref_log = ref_logits.log_softmax(-1)
        cand_log = logits.log_softmax(-1)
        probability = ref_log.exp()
        mode_kl = (probability * (ref_log - cand_log)).sum(-1)
        gaussian_kl = .5 * (((ref_mean - mean) / mean.new_tensor(SIGMA_DEG)) ** 2).sum(-1)
        total = total + (mode_kl + probability[:, motion_policy.MOVE] * gaussian_kl).sum()
        count += len(history.features)
    return total / count


def _histories_digest(histories):
    """Bind later rounds to exactly the same direct-demonstration histories."""
    digest = hashlib.sha256()
    for history in histories:
        PublicHistory(history.features, history.previous, history.transfer_start)
        for tensor in (history.features, history.previous, history.transfer_start):
            digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


class OutputLayerTrainer:
    """Fixed-FK200-reference score-function updates with two separate KL gates.

    Each update receives whole four-rollout groups and the fixed direct-demo
    public histories.  Group construction, posthoc official scoring, the
    64-rollout pilot gate, and physical execution belong to the caller.
    """

    def __init__(self, model):
        configure_output_only(model)
        if getattr(model, "_fk200_initial_digest", None) != state_digest(model):
            raise ValueError("the unchanged FK200 checkpoint must initialize the trainer")
        self.model = model
        self.reference = copy.deepcopy(model).eval()
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)
        self._reference_digest = state_digest(self.reference)
        self._demonstration_digest = None
        self.optimizer = torch.optim.Adam(model.output.parameters(), lr=.0003, weight_decay=0.0)
        self.rounds = self.consecutive_unaccepted = 0

    @property
    def stopped(self):
        return self.rounds >= 15 or self.consecutive_unaccepted >= 2

    def _unaccepted(self, reason, **extra):
        self.consecutive_unaccepted += 1
        return {"accepted": False, "reason": reason, "round": self.rounds,
                "consecutive_unaccepted": self.consecutive_unaccepted, **extra}

    def update(self, groups, demonstration_histories):
        if self.stopped:
            raise ValueError("the update limit or two consecutive unaccepted rounds was reached")
        expected_params = list(self.model.output.parameters())
        if (state_digest(self.reference) != self._reference_digest
                or not all(p.requires_grad == name.startswith("output.")
                           for name, p in self.model.named_parameters())
                or type(self.optimizer) is not torch.optim.Adam or len(self.optimizer.param_groups) != 1
                or self.optimizer.param_groups[0]["lr"] != .0003
                or self.optimizer.param_groups[0]["weight_decay"] != 0.0
                or self.optimizer.param_groups[0]["betas"] != (.9, .999)
                or self.optimizer.param_groups[0]["eps"] != 1e-8
                or self.optimizer.param_groups[0]["amsgrad"]
                or self.optimizer.param_groups[0]["maximize"]
                or len(self.optimizer.param_groups[0]["params"]) != len(expected_params)
                or any(a is not b for a, b in zip(self.optimizer.param_groups[0]["params"], expected_params))):
            raise ValueError("fixed reference, output-only trainability, or Adam settings changed")
        groups = tuple(tuple(group) for group in groups)
        demos = tuple(demonstration_histories)
        if not groups or not demos or any(len(group) != 4 for group in groups):
            raise ValueError("four-rollout groups and fixed demonstration histories are required")
        if any(not isinstance(h, PublicHistory) for h in demos):
            raise ValueError("direct demonstrations must contain public histories")
        demo_digest = _histories_digest(demos)
        if self._demonstration_digest is not None and self._demonstration_digest != demo_digest:
            raise ValueError("fixed direct-demonstration histories changed")
        current_digest = state_digest(self.model)
        valid_groups = []
        for group in groups:
            for item in group:
                _validate_trajectory(item)
                if item.policy_sha256 != current_digest:
                    raise ValueError("only current on-policy samples may be updated")
            if all(item.reward is not None for item in group):
                valid_groups.append(group)
        if not valid_groups:
            self._demonstration_digest = demo_digest
            self.rounds += 1
            return self._unaccepted("no_valid_groups")
        if all(len({item.reward for item in group}) == 1 for group in valid_groups):
            self._demonstration_digest = demo_digest
            self.rounds += 1
            return self._unaccepted("zero_advantage")
        losses, histories = [], []
        for group in valid_groups:
            rewards = [item.reward for item in group]
            for item in group:
                replay = replay_trajectory(self.model, item)
                recorded = replay.decision_log_prob.new_tensor([x.collection_log_prob for x in item.decisions])
                if not torch.allclose(replay.decision_log_prob.detach(), recorded, rtol=1e-5, atol=2e-4):
                    raise ValueError("current policy replay differs from collected probabilities")
                advantage = item.reward - (sum(rewards) - item.reward) / 3.0
                losses.append(-advantage * replay.decision_log_prob.sum())
                histories.append(trajectory_history(item))
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite score-function loss")
        previous_demo_digest = self._demonstration_digest
        self._demonstration_digest = demo_digest
        self.rounds += 1
        parameters = list(self.model.output.parameters())
        before = [p.detach().clone() for p in parameters]
        optimizer_before = copy.deepcopy(self.optimizer.state_dict())
        try:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise ValueError("nonfinite or missing output-layer gradient")
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
            self.optimizer.step()
            delta = [p.detach().clone() - old for p, old in zip(parameters, before)]
            measured = None
            for halvings in range(4):
                if halvings:
                    with torch.no_grad():
                        for p, old, change in zip(parameters, before, delta):
                            p.copy_(old + change * (0.5 ** halvings))
                with torch.no_grad():
                    demo_kl = float(policy_kl(self.reference, self.model, demos))
                    rollout_kl = float(policy_kl(self.reference, self.model, histories))
                measured = (demo_kl, rollout_kl)
                if all(math.isfinite(k) and k <= .1 for k in measured):
                    self.consecutive_unaccepted = 0
                    return {"accepted": True, "reason": "accepted", "round": self.rounds,
                            "consecutive_unaccepted": 0, "loss": float(loss.detach()),
                            "demo_kl": demo_kl, "rollout_kl": rollout_kl, "halvings": halvings,
                            "valid_groups": len(valid_groups)}
            with torch.no_grad():
                for p, old in zip(parameters, before):
                    p.copy_(old)
            self.optimizer.load_state_dict(optimizer_before)
            return self._unaccepted("kl_exceeded", demo_kl=measured[0], rollout_kl=measured[1], halvings=3)
        except BaseException:
            with torch.no_grad():
                for p, old in zip(parameters, before):
                    p.copy_(old)
            self.optimizer.load_state_dict(optimizer_before)
            self._demonstration_digest = previous_demo_digest
            self.rounds -= 1
            raise
