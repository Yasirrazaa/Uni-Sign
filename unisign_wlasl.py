import torch
from models import Uni_Sign
import torch.nn as nn

class WLASLUniSign(Uni_Sign):
    def __init__(self, args, num_classes=2000, gloss_vocab=None):
        super().__init__(args)
        
        # Remove MT5 components as we're doing classification
        if hasattr(self, 'mt5_model'):
            delattr(self, 'mt5_model')
        if hasattr(self, 'mt5_tokenizer'):
            delattr(self, 'mt5_tokenizer')
            
        # Add classification head
        hidden_dim = args.hidden_dim
        self.classifier = nn.Linear(hidden_dim*4, num_classes)
        
        # Initialize gloss vocabulary for classification
        self.num_classes = num_classes
        if gloss_vocab is not None:
            self.gloss_to_idx = {gloss: idx for idx, gloss in enumerate(sorted(gloss_vocab))}
            print(f"Initialized gloss vocabulary with {len(self.gloss_to_idx)} classes")
            self.initialized = True
        else:
            self.gloss_to_idx = None
            self.initialized = False
        
    def create_future_mask(self, T):
        """Create causal mask for temporal attention"""
        # Ensure T is a scalar value
        if isinstance(T, torch.Tensor):
            T = T.item()
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        return mask
        
    def forward(self, src_input, tgt_input=None):
        # Add future masking to attention mask
        B, T = src_input['attention_mask'].shape
        future_mask = self.create_future_mask(T).to(src_input['attention_mask'].device)
        future_mask = future_mask[None, :, :].expand(B, -1, -1)  # Expand to batch size
        src_input['attention_mask'] = src_input['attention_mask'].float().unsqueeze(-1) * (~future_mask).float()
        
        # RGB branch forward (unchanged)
        if self.args.rgb_support:
            rgb_support_dict = {}
            for index_key, rgb_key in zip(['left_sampled_indices', 'right_sampled_indices'], 
                                        ['left_hands', 'right_hands']):
                rgb_feat = self.rgb_support_backbone(src_input[rgb_key])
                rgb_support_dict[index_key] = src_input[index_key]
                rgb_support_dict[rgb_key] = rgb_feat
        
        # Pose branch forward
        features = []
        body_feat = None
        
        for part in self.modes:
            # Project position to hidden dim
            input_tensor = src_input[part]
            if input_tensor.dtype != torch.float32:
                input_tensor = input_tensor.float()  # Convert to float32
            proj_feat = self.proj_linear[part](input_tensor).permute(0,3,1,2)
            # Spatial GCN forward
            gcn_feat = self.gcn_modules[part](proj_feat)
            
            if part == 'body':
                body_feat = gcn_feat
            else:
                assert not body_feat is None
                if part == 'left':
                    if self.args.rgb_support:
                        gcn_feat = self.gather_feat_pose_rgb(
                            gcn_feat, 
                            rgb_support_dict[f'{part}_hands'],
                            rgb_support_dict[f'{part}_sampled_indices'],
                            src_input[f'{part}_rgb_len'],
                            src_input[f'{part}_skeletons_norm']
                        )
                    gcn_feat = gcn_feat + body_feat[..., -2][...,None].detach()
                    
                elif part == 'right':
                    if self.args.rgb_support:
                        gcn_feat = self.gather_feat_pose_rgb(
                            gcn_feat,
                            rgb_support_dict[f'{part}_hands'],
                            rgb_support_dict[f'{part}_sampled_indices'],
                            src_input[f'{part}_rgb_len'],
                            src_input[f'{part}_skeletons_norm']
                        )
                    gcn_feat = gcn_feat + body_feat[..., -1][...,None].detach()
                    
                elif part == 'face_all':
                    gcn_feat = gcn_feat + body_feat[..., 0][...,None].detach()
            
            # Temporal GCN forward
            gcn_feat = self.fusion_gcn_modules[part](gcn_feat)
            pool_feat = gcn_feat.mean(-1).transpose(1,2)
            features.append(pool_feat)
        
        # Concatenate features and get final embedding
        inputs_embeds = torch.cat(features, dim=-1) + self.part_para
        
        # Get class logits
        logits = self.classifier(inputs_embeds.mean(1))  # Average pooling over time
        
        if self.training:
            # Training mode
            loss = None
            if tgt_input is not None and tgt_input['gt_gloss'] is not None:
                try:
                    # Process space-separated gloss strings
                    batch_labels = []
                    for gloss_str in tgt_input['gt_gloss']:
                        # Split into individual tokens
                        gloss_tokens = gloss_str.split()
                        # Use the first token for classification
                        if gloss_tokens:
                            main_gloss = gloss_tokens[0]
                            if main_gloss in self.gloss_to_idx:
                                batch_labels.append(self.gloss_to_idx[main_gloss])
                            else:
                                print(f"Warning: Unknown gloss token: {main_gloss}")
                                batch_labels.append(0)  # Use index 0 for unknown tokens
                        else:
                            batch_labels.append(0)
                    
                    labels = torch.tensor(batch_labels, device=logits.device, dtype=torch.long)
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits, labels)
                    
                except Exception as e:
                    print(f"Error processing labels: {e}")
                    print(f"Current batch glosses: {tgt_input['gt_gloss']}")
                    print(f"Gloss mapping initialized: {self.initialized}")
                    if self.gloss_to_idx:
                        print(f"Known glosses: {list(self.gloss_to_idx.keys())[:10]}...")
                    loss = None
            
            return {
                'logits': logits,
                'loss': loss
            }
        else:
            # Inference mode
            return {
                'logits': logits
            }
