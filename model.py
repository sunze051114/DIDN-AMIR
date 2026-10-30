"""DIDN backbone with AMIR instruction, spatial and channel routing.

AMIR routing is adapted from Yaziwel's Apache-2.0 implementation.
See instruction/AMIR_NOTICE.md for provenance and changes. Public state names
follow Model_AMIR.py; exp/sum/log expert fusion uses stable logsumexp.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal


class Mlp(nn.Module):
    def __init__(self, in_feat, h_feat, out_feat):
        super().__init__()
        self.fc1 = nn.Linear(in_feat, h_feat)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h_feat, out_feat)
        self.drop = nn.Dropout(0.)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class MoE(nn.Module):
    """Per-pixel noisy top-2 routing among four MLP experts, as in AMIR."""
    def __init__(self, input_size, output_size, mlp_ratio=2.66,
                 num_experts=4, noisy_gating=True, use_experts=2):
        super().__init__()
        if not 1 <= use_experts <= num_experts:
            raise ValueError("use_experts must be between 1 and num_experts")
        self.k, self.num_experts = use_experts, num_experts
        self.noisy_gating = noisy_gating
        self.experts = nn.ModuleList([
            Mlp(input_size, int(input_size * mlp_ratio), output_size)
            for _ in range(num_experts)])
        self.w_gate = nn.Parameter(torch.randn(2 * input_size, num_experts))
        self.w_noise = nn.Parameter(torch.zeros(2 * input_size, num_experts))
        self.register_buffer("mean", torch.tensor([0.]))
        self.register_buffer("std", torch.tensor([1.]))

    @staticmethod
    def cv_squared(x):
        x = x.float()
        return x.var() / (x.mean().square() + 1e-10) if x.numel() > 1 else x.sum() * 0

    def noisy_top_k_gating(self, x):
        """Return softmax gates (for routing/load), top-k indices and top-k log-probabilities."""
        # Keep probability/CDF calculations in FP32 even during optional AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            clean = x @ self.w_gate.float()
            if self.noisy_gating and self.training:
                std = F.softplus(x @ self.w_noise.float()) + 1e-2
                logits = clean + torch.randn_like(clean) * std
            else:
                logits = clean
            values, indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
            top_gates = values[:, :self.k].softmax(dim=1)
            # log_softmax keeps tiny selected weights exact; log(softmax(...)) would
            # clamp or underflow them and distort the fused expert output.
            top_log_gates = values[:, :self.k].log_softmax(dim=1)
            gates = torch.zeros_like(logits).scatter(1, indices[:, :self.k], top_gates)
            if self.noisy_gating and self.training and self.k < self.num_experts:
                threshold_in = values[:, self.k:self.k + 1]
                threshold_out = values[:, self.k - 1:self.k]
                normal = Normal(self.mean.float(), self.std.float())
                probability = torch.where(
                    logits > threshold_in,
                    normal.cdf((clean - threshold_in) / std),
                    normal.cdf((clean - threshold_out) / std))
                load = probability.sum(0)
            else:
                load = (gates > 0).sum(0)
        return gates, load, indices[:, :self.k], top_log_gates

    def forward(self, x, prompt):
        batch, channels, height, width = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(-1, channels)
        instruction = prompt[:, None, None, :].expand(batch, height, width, channels).reshape(-1, channels)
        gates, load, indices, log_weights = self.noisy_top_k_gating(torch.cat((tokens, instruction), 1))
        balance = self.cv_squared(gates.sum(0)) + self.cv_squared(load)
        # Dispatch only selected tokens. Keep top-k slots to fuse in log space;
        # this is log(sum(exp(log_gate + expert))) so tiny gates stay exact.
        slots = tokens.new_zeros((tokens.shape[0] * self.k, channels), dtype=torch.float32)
        for expert_id, expert in enumerate(self.experts):
            token_id, slot_id = torch.where(indices == expert_id)
            if token_id.numel():
                value = expert(tokens[token_id]).float()
                slots = slots.index_copy(0, token_id * self.k + slot_id, value)
        merged = torch.logsumexp(slots.reshape(-1, self.k, channels) + log_weights[..., None], dim=1)
        merged = merged.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return merged.to(x.dtype), balance


class RIN(nn.Module):
    def __init__(self, in_dim, atom_num=16, atom_dim=256):
        super().__init__()
        self.CondNet = nn.Sequential(
            nn.Conv2d(in_dim, 64, 3, 3), nn.LeakyReLU(.1, True),
            nn.Conv2d(64, 64, 3, 3), nn.LeakyReLU(.1, True),
            nn.Conv2d(64, 64, 1), nn.LeakyReLU(.1, True),
            nn.Conv2d(64, 64, 1), nn.LeakyReLU(.1, True), nn.Conv2d(64, 32, 1))
        self.lastOut = nn.Linear(32, atom_num)
        self.act = nn.GELU()
        self.dictionary = nn.Parameter(torch.randn(atom_num, atom_dim))

    def forward(self, x):
        weights = self.lastOut(F.adaptive_avg_pool2d(self.CondNet(x), 1).flatten(1)).softmax(-1)
        return self.act(weights @ self.dictionary)


class Channel_Routing(nn.Module):
    def __init__(self, atom_dim, dim):
        super().__init__()
        self.fc = nn.Linear(atom_dim, dim)
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x, prompt):
        factors = self.fc(prompt).sigmoid()[:, :, None, None]
        return x + (x * self.gamma + self.beta) * factors


class Spatial_Routing(nn.Module):
    def __init__(self, atom_dim, dim, ffn_expansion_factor=2.66):
        super().__init__()
        self.fc = nn.Linear(atom_dim, dim)
        self.moe = MoE(dim, dim, ffn_expansion_factor)

    def forward(self, x, prompt):
        routed, balance = self.moe(x, self.fc(prompt))
        return x + routed, balance


class _Residual_Block(nn.Module):
    """
    DIDN core residual block with U-Net style encoder-decoder.
    Encoder: 2 downsampling stages, channels: N → 2N → 4N
    Decoder: 2 upsampling stages, channels: 4N → 2N → N
    Skip connections preserve encoder details for upsampling fusion.
    Residual connections at each stage stabilize gradient flow.
    """
    def __init__(self, N=256):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(N, N, 3, 1, 1, bias=False)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(N, N, 3, 1, 1, bias=False)
        self.relu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(N, 2 * N, 3, 2, 1, bias=False)
        self.relu6 = nn.PReLU()

        self.conv7 = nn.Conv2d(2 * N, 2 * N, 3, 1, 1, bias=False)
        self.relu8 = nn.PReLU()

        self.conv9 = nn.Conv2d(2 * N, 4 * N, 3, 2, 1, bias=False)
        self.relu10 = nn.PReLU()

        self.conv11 = nn.Conv2d(4 * N, 4 * N, 3, 1, 1, bias=False)
        self.relu12 = nn.PReLU()

        self.conv13 = nn.Conv2d(4 * N, 8 * N, 1, 1, 0, bias=False)
        self.up14 = nn.PixelShuffle(2)

        self.conv15 = nn.Conv2d(4 * N, 2 * N, 1, 1, 0, bias=False)
        self.conv16 = nn.Conv2d(2 * N, 2 * N, 3, 1, 1, bias=False)
        self.relu17 = nn.PReLU()

        self.conv18 = nn.Conv2d(2 * N, 4 * N, 1, 1, 0, bias=False)
        self.up19 = nn.PixelShuffle(2)

        self.conv20 = nn.Conv2d(2 * N, N, 1, 1, 0, bias=False)
        self.conv21 = nn.Conv2d(N, N, 3, 1, 1, bias=False)
        self.relu22 = nn.PReLU()
        self.conv23 = nn.Conv2d(N, N, 3, 1, 1, bias=False)
        self.relu24 = nn.PReLU()

        self.conv25 = nn.Conv2d(N, N, 3, 1, 1, bias=False)

    def forward(self, x):
        res1 = x

        out = self.conv1(x)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu4(out)
        out = torch.add(res1, out)
        cat1 = out

        out = self.conv5(out)
        out = self.relu6(out)
        res2 = out
        out = self.conv7(out)
        out = self.relu8(out)
        out = torch.add(res2, out)
        cat2 = out

        out = self.conv9(out)
        out = self.relu10(out)
        res3 = out
        out = self.conv11(out)
        out = self.relu12(out)
        out = torch.add(res3, out)

        out = self.conv13(out)
        out = self.up14(out)

        out = torch.cat([out, cat2], 1)
        out = self.conv15(out)
        res4 = out
        out = self.conv16(out)
        out = self.relu17(out)
        out = torch.add(res4, out)

        out = self.conv18(out)
        out = self.up19(out)

        out = torch.cat([out, cat1], 1)
        out = self.conv20(out)
        res5 = out
        out = self.conv21(out)
        out = self.relu22(out)
        out = self.conv23(out)
        out = self.relu24(out)
        out = torch.add(res5, out)

        out = self.conv25(out)
        out = torch.add(out, res1)

        return out


class Recon_Block(nn.Module):
    """
    Reconstruction block with 4 residual units for feature refinement.
    """
    def __init__(self, N=256):
        super(Recon_Block, self).__init__()

        self.conv1 = nn.Conv2d(N, N, 3, 1, 1, bias=False)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(N, N, 3, 1, 1, bias=False)
        self.relu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(N, N, 3, 1, 1, bias=False)
        self.relu6 = nn.PReLU()
        self.conv7 = nn.Conv2d(N, N, 3, 1, 1, bias=False)
        self.relu8 = nn.PReLU()

        self.conv9 = nn.Conv2d(N, N, 3, 1, 1, bias=False)
        self.relu10 = nn.PReLU()
        self.conv11 = nn.Conv2d(N, N, 3, 1, 1, bias=False)
        self.relu12 = nn.PReLU()

        self.conv13 = nn.Conv2d(N, N, 3, 1, 1, bias=False)
        self.relu14 = nn.PReLU()
        self.conv15 = nn.Conv2d(N, N, 3, 1, 1, bias=False)
        self.relu16 = nn.PReLU()

        self.conv17 = nn.Conv2d(N, N, 3, 1, 1, bias=False)

    def forward(self, x):
        res1 = x

        output = self.conv1(x)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.relu4(output)
        output = torch.add(output, res1)

        res2 = output
        output = self.conv5(output)
        output = self.relu6(output)
        output = self.conv7(output)
        output = self.relu8(output)
        output = torch.add(output, res2)

        res3 = output
        output = self.conv9(output)
        output = self.relu10(output)
        output = self.conv11(output)
        output = self.relu12(output)
        output = torch.add(output, res3)

        res4 = output
        output = self.conv13(output)
        output = self.relu14(output)
        output = self.conv15(output)
        output = self.relu16(output)
        output = torch.add(output, res4)

        output = self.conv17(output)
        output = torch.add(output, res1)

        return output


class DIDNAMIR(nn.Module):
    """Six DIDN DUBs, shared reconstruction and AMIR task routing."""
    def __init__(self):
        super().__init__()
        in_channels = 1
        self.in_channels = in_channels

        self.conv_input = nn.Conv2d(in_channels, 256, 3, 1, 1, bias=False)
        self.relu1 = nn.PReLU()
        self.conv_down = nn.Conv2d(256, 256, 3, 2, 1, bias=False)
        self.relu2 = nn.PReLU()

        self.recursive_A = _Residual_Block(N=256)
        self.recursive_B = _Residual_Block(N=256)
        self.recursive_C = _Residual_Block(N=256)
        self.recursive_D = _Residual_Block(N=256)
        self.recursive_E = _Residual_Block(N=256)
        self.recursive_F = _Residual_Block(N=256)

        self.recon = Recon_Block(N=256)

        self.conv_mid = nn.Conv2d(1536, 256, 1, 1, 0, bias=False)
        self.relu3 = nn.PReLU()
        self.conv_mid2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.relu4 = nn.PReLU()

        self.subpixel = nn.PixelShuffle(2)
        self.conv_output = nn.Conv2d(64, in_channels, 3, 1, 1, bias=False)

        self.dict_generator = RIN(in_channels, atom_num=16, atom_dim=256)
        self.spatial_routing = nn.ModuleList([Spatial_Routing(256, 256) for _ in range(3)])
        self.channel_routing = nn.ModuleList([Channel_Routing(256, 256) for _ in range(4)])

    def forward(self, x, return_aux=False):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("Expected an NCHW tensor matching the model's input channels")
        if min(x.shape[-2:]) < 16 or any(size % 8 for size in x.shape[-2:]):
            raise ValueError("DIDN+AMIR requires dimensions >=16 divisible by 8")
        prompt = self.dict_generator(x)
        out = self.relu2(self.conv_down(self.relu1(self.conv_input(x))))
        features, losses = [], []
        blocks = (self.recursive_A, self.recursive_B, self.recursive_C,
                  self.recursive_D, self.recursive_E, self.recursive_F)
        for stage in range(3):
            out = blocks[2 * stage](out)
            out, balance = self.spatial_routing[stage](out, prompt)
            features.append(self.recon(out))
            losses.append(balance)
            out = self.channel_routing[stage](blocks[2 * stage + 1](out), prompt)
            features.append(self.recon(out))
        out = self.relu3(self.conv_mid(torch.cat(features, 1)))
        out = self.channel_routing[3](out, prompt)
        out = out + self.relu4(self.conv_mid2(out))
        prediction = self.conv_output(self.subpixel(out)) + x
        return (prediction, torch.stack(losses).sum()) if return_aux else prediction


def get_model():
    """Build the single-channel DIDN+AMIR network."""
    return DIDNAMIR()
