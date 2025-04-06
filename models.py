from torch import Tensor
import torch
from torch import nn
import torch.utils.checkpoint
import contextlib
import torchvision
from einops import rearrange
import os

import math
from stgcn_layers import Graph, get_stgcn_chain
from deformable_attention_2d import DeformableAttention2D
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import warnings
from config import mt5_path

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: [B, T, C]
        x_t = x.transpose(0, 1)  # [T, B, C]
        attn_output, _ = self.attention(x_t, x_t, x_t, attn_mask=mask)
        attn_output = self.dropout(attn_output)
        attn_output = attn_output.transpose(0, 1)  # [B, T, C]
        return self.norm(x + attn_output)

class FeatureFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, features):
        # features: list of tensors [B, T, C]
        concat_features = torch.cat(features, dim=-1)
        gates = self.gate(concat_features)
        return self.linear(concat_features) * gates

class ISLRClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, T, C]
        # Global average pooling over time
        x = x.mean(1)  # [B, C]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class Uni_Sign(nn.Module):
    def __init__(self, args):
        super(Uni_Sign, self).__init__()
        self.args = args

        self.modes = ['body', 'left', 'right', 'face_all']

        self.graph, A = {}, []
        # project (x,y,score) to hidden dim
        hidden_dim = args.hidden_dim
        self.proj_linear = nn.ModuleDict()
        for mode in self.modes:
            self.graph[mode] = Graph(layout=f'{mode}', strategy='distance', max_hop=1)
            A.append(torch.tensor(self.graph[mode].A, dtype=torch.float32, requires_grad=False))
            self.proj_linear[mode] = nn.Linear(3, 64)

        self.gcn_modules = nn.ModuleDict()
        self.fusion_gcn_modules = nn.ModuleDict()
        spatial_kernel_size = A[0].size(0)
        for index, mode in enumerate(self.modes):
            self.gcn_modules[mode], final_dim = get_stgcn_chain(64, 'spatial', (1, spatial_kernel_size), A[index].clone(), True)
            self.fusion_gcn_modules[mode], _ = get_stgcn_chain(final_dim, 'temporal', (5, spatial_kernel_size), A[index].clone(), True)

        self.gcn_modules['left'] = self.gcn_modules['right']
        self.fusion_gcn_modules['left'] = self.fusion_gcn_modules['right']
        self.proj_linear['left'] = self.proj_linear['right']

        self.part_para = nn.Parameter(torch.zeros(hidden_dim*len(self.modes)))
        self.pose_proj = nn.Linear(256*4, 768)

        # Initialize ISLR-specific components
        self.initialize_islr_components(args)

        # Apply weight initialization
        self.apply(self._init_weights)

        if "CSL" in self.args.dataset:
            self.lang = 'Chinese'
        else:
            self.lang = 'English'

        if self.args.rgb_support:
            self.rgb_support_backbone = torch.nn.Sequential(*list(torchvision.models.efficientnet_b0(pretrained=True).children())[:-2])
            self.rgb_proj = nn.Conv2d(1280, hidden_dim, kernel_size=1)

            self.fusion_pose_rgb_linear = nn.Linear(hidden_dim, hidden_dim)

            # PGF
            self.fusion_pose_rgb_DA = DeformableAttention2D(
                                        dim = hidden_dim,            # feature dimensions
                                        dim_head = 32,               # dimension per head
                                        heads = 8,                   # attention heads
                                        dropout = 0.,                # dropout
                                        downsample_factor = 1,       # downsample factor (r in paper)
                                        offset_scale = None,         # scale of offset, maximum offset
                                        offset_groups = None,        # number of offset groups, should be multiple of heads
                                        offset_kernel_size = 1,      # offset kernel size
                                    )

            self.fusion_gate = nn.Sequential(nn.Conv1d(hidden_dim*2, hidden_dim, 1),
                                        nn.GELU(),
                                        nn.Conv1d(hidden_dim, 1, 1),
                                        nn.Tanh(),
                                        nn.ReLU(),
                                    )

            for layer in self.fusion_gate:
                if isinstance(layer, nn.Conv1d):
                    nn.init.constant_(layer.weight, 0)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

        # Only initialize MT5 if not doing ISLR task
        if not (hasattr(args, 'task') and args.task == 'ISLR'):
            self.mt5_model = MT5ForConditionalGeneration.from_pretrained(mt5_path)
            self.mt5_tokenizer = T5Tokenizer.from_pretrained(mt5_path, legacy=False)
        else:
            # For ISLR task, explicitly remove MT5 components if they exist
            # This helps when loading from a checkpoint that has MT5 components
            if hasattr(self, 'mt5_model'):
                delattr(self, 'mt5_model')
            if hasattr(self, 'mt5_tokenizer'):
                delattr(self, 'mt5_tokenizer')

    def initialize_islr_components(self, args):
        """Initialize ISLR-specific components"""
        # Add temporal attention modules for each body part
        if hasattr(args, 'task') and args.task == 'ISLR':
            # Add temporal attention modules for each body part
            self.temporal_attention = nn.ModuleDict()
            for mode in self.modes:
                self.temporal_attention[mode] = TemporalAttention(
                    hidden_dim=256,  # Match the output dim of spatial GCN
                    num_heads=4,
                    dropout=getattr(args, 'dropout', 0.1)
                )

            # Add feature fusion module
            self.feature_fusion = FeatureFusion(
                input_dim=256*4,  # 4 body parts with 256 dim each
                hidden_dim=768    # Match the original projection dimension
            )

            # Add ISLR-specific classification head
            num_classes = getattr(args, 'num_classes', 2000)  # Default to 2000 classes for WLASL
            self.islr_classifier = ISLRClassificationHead(
                input_dim=768,
                hidden_dim=512,
                num_classes=num_classes,
                dropout=getattr(args, 'dropout', 0.3)
            )

            # Initialize gloss vocabulary for WLASL dataset
            if args.dataset == 'WLASL':
                # Set the vocabulary path
                vocab_path = getattr(args, 'vocab_path', 'data/WLASL/gloss_vocab.json')
                reverse_path = vocab_path.replace('.json', '_reverse.json')

                # Try to load vocabulary from file
                if os.path.exists(vocab_path):
                    try:
                        import json
                        with open(vocab_path, 'r') as f:
                            self.gloss_to_idx = json.load(f)

                        # Load reverse mapping if available
                        if os.path.exists(reverse_path):
                            with open(reverse_path, 'r') as f:
                                self.idx_to_gloss = json.load(f)
                                # Convert keys from strings to integers
                                self.idx_to_gloss = {int(k): v for k, v in self.idx_to_gloss.items()}
                        else:
                            # Create reverse mapping
                            self.idx_to_gloss = {}
                            for gloss, idx in self.gloss_to_idx.items():
                                if not gloss.isdigit():  # Skip numeric keys
                                    self.idx_to_gloss[idx] = gloss

                        print(f"Loaded gloss vocabulary with {len(self.idx_to_gloss)} unique classes and {len(self.gloss_to_idx)} mappings from {vocab_path}")
                        print("Sample gloss->idx entries:")
                        sample_items = [(g, i) for g, i in list(self.gloss_to_idx.items())[:10] if not g.isdigit()]
                        for gloss, idx in sample_items[:5]:
                            print(f"  {gloss}: {idx}")

                        print("Sample idx->gloss entries:")
                        sample_items = list(self.idx_to_gloss.items())[:5]
                        for idx, gloss in sample_items:
                            print(f"  {idx}: {gloss}")
                    except Exception as e:
                        print(f"Error loading gloss vocabulary: {e}")
                        print("Creating default numeric mapping instead")
                        self.gloss_to_idx = {}
                        self.idx_to_gloss = {}
                        for i in range(num_classes):
                            self.gloss_to_idx[str(i)] = i
                            self.idx_to_gloss[i] = str(i)
                else:
                    print(f"Warning: Vocabulary file {vocab_path} not found. Creating default numeric mapping.")
                    self.gloss_to_idx = {}
                    self.idx_to_gloss = {}
                    for i in range(num_classes):
                        self.gloss_to_idx[str(i)] = i
                        self.idx_to_gloss[i] = str(i)

    def create_future_mask(self, T):
        """Create causal mask for temporal attention"""
        # Ensure T is a scalar value
        if isinstance(T, torch.Tensor):
            T = T.item()
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        return mask


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def maybe_autocast(self, dtype=torch.float32):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        # Check if CUDA is available
        enable_autocast = torch.cuda.is_available()

        if enable_autocast:
            from torch.cuda.amp import autocast
            return autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def gather_feat_pose_rgb(self, gcn_feat, rgb_feat, indices, rgb_len, pose_init):
        b, c, T, n = gcn_feat.shape
        assert rgb_feat.shape[0] == indices.shape[0]
        rgb_feat = self.rgb_proj(rgb_feat)

        assert len(rgb_len) == b
        start = 0
        for batch in range(b):
            index = indices[start:start + rgb_len[batch]].to(torch.long)
            # ignore some invalid rgb clip
            if rgb_len[batch] == 1 and -1 in index:
                start = start + rgb_len[batch]
                continue

            # index selection
            gcn_feat_selected = gcn_feat[batch, :, index]
            rgb_feat_selected = rgb_feat[start:start + rgb_len[batch]]
            pose_init_selected = pose_init[start:start + rgb_len[batch]]

            gcn_feat_selected = rearrange(gcn_feat_selected, 'c t n -> t c n')
            pose_init_selected = rearrange(pose_init_selected, 't n c -> t c n')

            # PGF forward
            with self.maybe_autocast():
                fused_transposed = self.fusion_pose_rgb_DA(pose_feat=gcn_feat_selected,
                                                            rgb_feat=rgb_feat_selected,
                                                            pose_init=pose_init_selected, )

            fused_transposed = fused_transposed.to(gcn_feat.dtype)
            gate_feature = torch.concat([fused_transposed, gcn_feat_selected,], dim=-2)
            gate_score = self.fusion_gate(gate_feature)
            fused_transposed_post = (gate_score) * fused_transposed + (1 - gate_score) * gcn_feat_selected

            gcn_feat = gcn_feat.clone()
            fused_transposed_post = rearrange(fused_transposed_post, 't c n -> c t n')

            # replace gcn feature
            gcn_feat[batch, :, index] = fused_transposed_post
            start = start + rgb_len[batch]

        assert start == rgb_feat.shape[0]
        return gcn_feat

    def forward(self, src_input, tgt_input):
        # Apply future masking if enabled
        future_mask = None
        if hasattr(self.args, 'use_future_mask') and self.args.use_future_mask:
            B, T = src_input['attention_mask'].shape
            future_mask = self.create_future_mask(T).to(src_input['attention_mask'].device)
            future_mask = future_mask[None, :, :].expand(B, -1, -1)
            src_input['attention_mask'] = src_input['attention_mask'].float().unsqueeze(-1) * (~future_mask).float()

        # RGB branch forward
        if self.args.rgb_support:
            rgb_support_dict = {}
            for index_key, rgb_key in zip(['left_sampled_indices', 'right_sampled_indices'], ['left_hands', 'right_hands']):
                rgb_feat = self.rgb_support_backbone(src_input[rgb_key])

                rgb_support_dict[index_key] = src_input[index_key]
                rgb_support_dict[rgb_key] = rgb_feat

        # Pose branch forward
        features = []

        body_feat = None
        for part in self.modes:
            # project position to hidden dim
            # Ensure input is Float32 and on the correct device
            input_tensor = src_input[part].to(dtype=torch.float32, device=self.proj_linear[part].weight.device)
            proj_feat = self.proj_linear[part](input_tensor).permute(0,3,1,2) #B,C,T,V
            # spatial gcn forward
            gcn_feat = self.gcn_modules[part](proj_feat)
            if part == 'body':
                body_feat = gcn_feat
            else:
                assert not body_feat is None
                if part == 'left':
                    # Pose RGB fusion
                    if self.args.rgb_support:
                        gcn_feat = self.gather_feat_pose_rgb(gcn_feat,
                                                            rgb_support_dict[f'{part}_hands'],
                                                            rgb_support_dict[f'{part}_sampled_indices'],
                                                            src_input[f'{part}_rgb_len'],
                                                            src_input[f'{part}_skeletons_norm'],
                                                            )

                    gcn_feat = gcn_feat + body_feat[..., -2][...,None].detach()

                elif part == 'right':
                    # Pose RGB fusion
                    if self.args.rgb_support:
                        gcn_feat = self.gather_feat_pose_rgb(gcn_feat,
                                                                rgb_support_dict[f'{part}_hands'],
                                                                rgb_support_dict[f'{part}_sampled_indices'],
                                                                src_input[f'{part}_rgb_len'],
                                                                src_input[f'{part}_skeletons_norm'],
                                                                )

                    gcn_feat = gcn_feat + body_feat[..., -1][...,None].detach()

                elif part == 'face_all':
                    gcn_feat = gcn_feat + body_feat[..., 0][...,None].detach()
                else:
                    raise NotImplementedError

            # temporal gcn forward
            gcn_feat = self.fusion_gcn_modules[part](gcn_feat) #B,C,T,V
            pool_feat = gcn_feat.mean(-1).transpose(1,2) #B,T,C

            # Apply temporal attention if available
            if hasattr(self, 'temporal_attention'):
                pool_feat = self.temporal_attention[part](pool_feat, mask=future_mask)

            features.append(pool_feat)

        # Use feature fusion if available for ISLR
        if hasattr(self, 'feature_fusion') and hasattr(self.args, 'task') and self.args.task == 'ISLR':
            inputs_embeds = self.feature_fusion(features)
        else:
            # concat sub-pose feature across token dimension
            inputs_embeds = torch.cat(features, dim=-1) + self.part_para
            inputs_embeds = self.pose_proj(inputs_embeds)

        # For ISLR task, use classification head
        if hasattr(self.args, 'task') and self.args.task == 'ISLR':
            # Use ISLR classifier if available
            if hasattr(self, 'islr_classifier'):
                logits = self.islr_classifier(inputs_embeds)
            else:
                # Fallback to simple classification
                logits = torch.nn.functional.linear(inputs_embeds.mean(1), self.pose_proj.weight.t()[:, :inputs_embeds.size(-1)])

            # Calculate loss if in training mode
            loss = None
            if self.training and tgt_input is not None and 'gt_gloss' in tgt_input:
                labels = self.prepare_labels(tgt_input['gt_gloss'])
                loss = self.compute_loss(logits, labels)

            return {
                'logits': logits,
                'loss': loss
            }

        # Original processing for other tasks (SLT, CSLR)
        prefix_token = self.mt5_tokenizer(
                                [f"Translate sign language video to {self.lang}: "] * len(tgt_input["gt_sentence"]),
                                padding="longest",
                                truncation=True,
                                return_tensors="pt",
                            ).to(inputs_embeds.device)

        prefix_embeds = self.mt5_model.encoder.embed_tokens(prefix_token['input_ids'])
        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

        attention_mask = torch.cat([prefix_token['attention_mask'],
                                    src_input['attention_mask']], dim=1)

        tgt_input_tokenizer = self.mt5_tokenizer(tgt_input['gt_sentence'],
                                                return_tensors="pt",
                                                padding=True,
                                                truncation=True,
                                                max_length=50)

        labels = tgt_input_tokenizer['input_ids']
        labels[labels == self.mt5_tokenizer.pad_token_id] = -100

        out = self.mt5_model(inputs_embeds = inputs_embeds,
                    attention_mask = attention_mask,
                    labels = labels.to(inputs_embeds.device),
                    return_dict = True,
                    )

        label = labels.reshape(-1)
        out_logits = out['logits']
        logits = out_logits.reshape(-1,out_logits.shape[-1])
        loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing, ignore_index=-100)
        loss = loss_fct(logits, label.to(out_logits.device, non_blocking=True))

        stack_out = {
            # use for inference
            'inputs_embeds':inputs_embeds,
            'attention_mask':attention_mask,
            'loss':loss,
            'logits':out_logits  # Include logits for top-k evaluation
        }

        return stack_out

    @torch.no_grad()
    def generate(self,pre_compute_item,max_new_tokens,num_beams):
        inputs_embeds = pre_compute_item['inputs_embeds']
        attention_mask = pre_compute_item['attention_mask']

        out = self.mt5_model.generate(inputs_embeds = inputs_embeds,
                                attention_mask = attention_mask,
                                max_new_tokens=max_new_tokens,
                                num_beams = num_beams,
                            )

        return out

def get_requires_grad_dict(model):
    param_requires_grad = {name: True for name, param in model.named_parameters()}
    param_requires_grad_right = {}
    for key in param_requires_grad.keys():
        if 'left' in key:
            param_requires_grad_right[key.replace("left", 'right')] = param_requires_grad[key]
    param_requires_grad = {**param_requires_grad,
                           **param_requires_grad_right}
    params_to_update = {k: v for k, v in model.state_dict().items() if param_requires_grad.get(k, True)}

    return params_to_update

