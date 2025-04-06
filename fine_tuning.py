import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models import Uni_Sign
import utils as utils
from datasets import S2T_Dataset
import os
import time
import argparse, json, datetime
from pathlib import Path
import math
import sys
from timm.optim import create_optimizer
from models import get_requires_grad_dict
from SLRT_metrics import translation_performance, islr_performance, wer_list
from transformers import get_scheduler
from config import *

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def mixup_data(x, y, alpha=0.2):
    """
    Applies mixup augmentation to the batch.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss function
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main(args):
    utils.init_distributed_mode_ds(args)

    print(args)
    utils.set_seed(args.seed)

    print(f"Creating dataset:")

    train_data = S2T_Dataset(path=train_label_paths[args.dataset],
                             args=args, phase='train')
    print(train_data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,shuffle=True)
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=train_data.collate_fn,
                                 sampler=train_sampler,
                                 pin_memory=args.pin_mem,
                                 drop_last=True)

    dev_data = S2T_Dataset(path=dev_label_paths[args.dataset],
                           args=args, phase='dev')
    print(dev_data)
    # dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data,shuffle=False)
    dev_sampler = torch.utils.data.SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=dev_data.collate_fn,
                                sampler=dev_sampler,
                                pin_memory=args.pin_mem)

    test_data = S2T_Dataset(path=test_label_paths[args.dataset],
                            args=args, phase='test')
    print(test_data)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,shuffle=False)
    test_sampler = torch.utils.data.SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=test_data.collate_fn,
                                 sampler=test_sampler,
                                 pin_memory=args.pin_mem)

    print(f"Creating model:")
    model = Uni_Sign(
                args=args
                )
    model.cuda()
    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    if args.finetune != '':
        print('***********************************')
        print('Load Checkpoint...')
        print('***********************************')
        state_dict = torch.load(args.finetune, map_location='cpu')['model']

        # For ISLR task with our improved model, use strict=False
        if args.task == 'ISLR' and (hasattr(model, 'temporal_attention') or
                                   hasattr(model, 'feature_fusion') or
                                   hasattr(model, 'islr_classifier')):
            print('Loading checkpoint with strict=False for improved ISLR model')
            ret = model.load_state_dict(state_dict, strict=False)

            # Initialize the new components with proper weight initialization
            if hasattr(model, 'temporal_attention'):
                print('Initializing temporal attention modules')
                for mode in model.modes:
                    if hasattr(model.temporal_attention, mode):
                        for m in model.temporal_attention[mode].modules():
                            if isinstance(m, (nn.Linear, nn.Conv1d)):
                                nn.init.xavier_uniform_(m.weight)
                                if m.bias is not None:
                                    nn.init.constant_(m.bias, 0)

            if hasattr(model, 'feature_fusion'):
                print('Initializing feature fusion module')
                for m in model.feature_fusion.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            if hasattr(model, 'islr_classifier'):
                print('Initializing ISLR classifier')
                for m in model.islr_classifier.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
        else:
            # For other tasks or original model, use strict=True
            ret = model.load_state_dict(state_dict, strict=True)

        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=optimizer,
                num_warmup_steps=int(args.warmup_epochs * len(train_dataloader)/args.gradient_accumulation_steps),
                num_training_steps=int(args.epochs * len(train_dataloader)/args.gradient_accumulation_steps),
            )

    model, optimizer, lr_scheduler = utils.init_deepspeed(args, model, optimizer, lr_scheduler)
    model_without_ddp = model.module.module
    # print(model_without_ddp)
    print(optimizer)

    output_dir = Path(args.output_dir)

    start_time = time.time()
    max_accuracy = 0
    if args.task == "CSLR":
        max_accuracy = 1000

    if args.eval:
        if utils.is_main_process():
            if args.task != "ISLR":
                print("📄 dev result")
                evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
            print("📄 test result")
            evaluate(args, test_dataloader, model, model_without_ddp, phase='test')

        return
    print(f"Start training for {args.epochs} epochs")

    for epoch in range(0, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch, model_without_ddp)

        if args.output_dir:
            checkpoint_paths = [output_dir / f'checkpoint_{epoch}.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': get_requires_grad_dict(model_without_ddp),
                }, checkpoint_path)

        # single gpu inference
        if utils.is_main_process():
            test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
            evaluate(args, test_dataloader, model, model_without_ddp, phase='test')

            if args.task == "SLT":
                if max_accuracy < test_stats["bleu4"]:
                    max_accuracy = test_stats["bleu4"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)

                print(f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['bleu4']:.2f}")
                print(f'Max BLEU-4: {max_accuracy:.2f}%')

            elif args.task == "ISLR":
                if max_accuracy < test_stats["top1_acc_pi"]:
                    max_accuracy = test_stats["top1_acc_pi"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)

                print(f"PI accuracy of the network on the {len(dev_dataloader)} dev videos: {test_stats['top1_acc_pi']:.2f}")
                print(f'Max PI accuracy: {max_accuracy:.2f}%')

            elif args.task == "CSLR":
                if max_accuracy > test_stats["wer"]:
                    max_accuracy = test_stats["wer"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)

                print(f"WER of the network on the {len(dev_dataloader)} dev videos: {test_stats['wer']:.2f}")
                print(f'Min WER: {max_accuracy:.2f}%')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(args, model, data_loader, optimizer, epoch, model_without_ddp=None):
    model.train()

    # If model_without_ddp is not provided, try to get it from model
    if model_without_ddp is None:
        if hasattr(model, 'module'):
            if hasattr(model.module, 'module'):
                model_without_ddp = model.module.module
            else:
                model_without_ddp = model.module
        else:
            model_without_ddp = model

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    optimizer.zero_grad()

    target_dtype = None
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if target_dtype != None:
            for key in src_input.keys():
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].to(target_dtype).cuda()

        if args.task == "CSLR":
            tgt_input['gt_sentence'] = tgt_input['gt_gloss']

        # Apply mixup if enabled and task is ISLR
        use_mixup = hasattr(args, 'use_mixup') and args.use_mixup and args.task == 'ISLR'
        use_focal_loss = hasattr(args, 'use_focal_loss') and args.use_focal_loss and args.task == 'ISLR'

        if use_mixup and 'gt_gloss' in tgt_input and tgt_input['gt_gloss'] is not None and len(tgt_input['gt_gloss']) > 0:
            # Get input embeddings first
            with torch.no_grad():
                # Forward pass to get embeddings
                temp_out = model(src_input, None)
                if 'inputs_embeds' in temp_out:
                    inputs_embeds = temp_out['inputs_embeds']
                    # Apply mixup to embeddings
                    labels = tgt_input['gt_gloss']
                    mixed_embeds, labels_a, labels_b, lam = mixup_data(inputs_embeds, labels, alpha=0.2)
                    # Replace inputs with mixed version
                    src_input['mixed_embeds'] = mixed_embeds
                    tgt_input['mixed_labels_a'] = labels_a
                    tgt_input['mixed_labels_b'] = labels_b
                    tgt_input['mixup_lambda'] = lam

        # Forward pass
        stack_out = model(src_input, tgt_input)

        # Get loss
        if use_focal_loss and args.task == 'ISLR':
            # Use focal loss for ISLR task
            if 'logits' in stack_out and tgt_input is not None and 'gt_gloss' in tgt_input:
                focal_loss = FocalLoss(gamma=2.0)
                if hasattr(model_without_ddp, 'gloss_to_idx'):
                    # Convert labels to indices
                    batch_labels = []
                    for gloss_str in tgt_input['gt_gloss']:
                        if not gloss_str or not gloss_str.strip():
                            # Handle empty gloss
                            batch_labels.append(0)
                        else:
                            # Split and get first token, defaulting to 0 if not found
                            gloss_tokens = gloss_str.strip().split()
                            if gloss_tokens and gloss_tokens[0] in model_without_ddp.gloss_to_idx:
                                batch_labels.append(model_without_ddp.gloss_to_idx[gloss_tokens[0]])
                            else:
                                batch_labels.append(0)  # Default to first class
                    labels = torch.tensor(batch_labels, device=stack_out['logits'].device, dtype=torch.long)
                else:
                    # Assume labels are already indices
                    labels = tgt_input['gt_gloss'].to(stack_out['logits'].device)

                total_loss = focal_loss(stack_out['logits'], labels)
                stack_out['loss'] = total_loss

        # Apply mixup loss if used
        if use_mixup and 'mixed_labels_a' in tgt_input and 'mixed_labels_b' in tgt_input:
            lam = tgt_input['mixup_lambda']
            loss_fct = nn.CrossEntropyLoss()
            total_loss = mixup_criterion(loss_fct, stack_out['logits'],
                                        tgt_input['mixed_labels_a'],
                                        tgt_input['mixed_labels_b'],
                                        lam)
            stack_out['loss'] = total_loss

        # Get final loss
        total_loss = stack_out['loss']
        model.backward(total_loss)
        model.step()

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def compute_topk_accuracy(predictions, targets, k_values):
    """Computes top-k accuracy for specified k values"""
    max_k = max(k_values)
    batch_size = targets.size(0)

    _, pred = predictions.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results = {}
    for k in k_values:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results[f'top{k}'] = correct_k.mul_(100.0 / batch_size).item()

    return results

class TopKAccuracyMeter:
    def __init__(self, k_values=(1, 3, 5, 7, 10)):
        self.k_values = k_values
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(self, preds, targets):
        self.predictions.append(preds)
        self.targets.append(targets)

    def compute(self):
        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        return compute_topk_accuracy(all_preds, all_targets, self.k_values)

def evaluate(args, data_loader, model, model_without_ddp, phase):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Test: {phase}'

    # Add accuracy meters for ISLR task
    if args.task == "ISLR":
        metric_logger.add_meter('top1_acc_pi', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('top1_acc_pc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    # Print model information
    print("\nModel Information:")
    if hasattr(model_without_ddp, 'gloss_to_idx'):
        print(f"Model has gloss_to_idx with {len(model_without_ddp.gloss_to_idx)} entries")
        print("Sample entries:")
        sample_items = list(model_without_ddp.gloss_to_idx.items())[:5]
        for gloss, idx in sample_items:
            print(f"  {gloss}: {idx}")
    else:
        print("Model does not have gloss_to_idx attribute")

    # Print dataset information
    print("\nDataset Information:")
    print(f"Number of samples: {len(data_loader.dataset)}")

    # For ISLR task, prepare for collecting predictions and references
    if args.task == "ISLR":
        tgt_pres = []
        tgt_refs = []

    with torch.no_grad():
        for src_input, tgt_input in metric_logger.log_every(data_loader, 10, header):
            # Move inputs to device
            src_input = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                        for k, v in src_input.items()}

            # Forward pass
            outputs = model(src_input, tgt_input)
            logits = outputs['logits']

            if args.task == "ISLR":
                # Process labels for ISLR task
                if hasattr(model_without_ddp, 'gloss_to_idx'):
                    # Print sample raw labels for debugging
                    if len(tgt_pres) == 0:  # Only print for the first batch
                        print("\nRaw labels sample:")
                        for i, gloss in enumerate(tgt_input['gt_gloss'][:10]):
                            print(f"  {i}: '{gloss}'")

                        # For WLASL dataset, print video paths
                        if args.dataset == 'WLASL' and 'name_batch' in src_input:
                            print("\nVideo paths sample:")
                            for i, name in enumerate(src_input['name_batch'][:5]):
                                print(f"  {i}: '{name}'")

                    # Process predictions and references
                    batch_preds = []
                    batch_refs = []

                    # Get top-1 predictions
                    _, pred_indices = logits.topk(1, dim=1)
                    pred_indices = pred_indices.squeeze().cpu().tolist()
                    if not isinstance(pred_indices, list):
                        pred_indices = [pred_indices]

                    # Process each prediction and reference
                    for i, (pred_idx, gloss) in enumerate(zip(pred_indices, tgt_input['gt_gloss'])):
                        # Add prediction
                        batch_preds.append(str(pred_idx))

                        # Process reference label
                        if gloss in model_without_ddp.gloss_to_idx:
                            ref_idx = model_without_ddp.gloss_to_idx[gloss]
                        elif gloss.isdigit():
                            ref_idx = int(gloss)
                        else:
                            # Try other methods to map gloss to index
                            if gloss.lower() in model_without_ddp.gloss_to_idx:
                                ref_idx = model_without_ddp.gloss_to_idx[gloss.lower()]
                            elif gloss and gloss.strip():
                                gloss_parts = gloss.strip().split()
                                if gloss_parts and gloss_parts[0] in model_without_ddp.gloss_to_idx:
                                    ref_idx = model_without_ddp.gloss_to_idx[gloss_parts[0]]
                                elif gloss_parts and gloss_parts[0].lower() in model_without_ddp.gloss_to_idx:
                                    ref_idx = model_without_ddp.gloss_to_idx[gloss_parts[0].lower()]
                                else:
                                    ref_idx = 0
                            else:
                                ref_idx = 0

                        batch_refs.append(str(ref_idx))

                    # Add batch results to overall results
                    tgt_pres.extend(batch_preds)
                    tgt_refs.extend(batch_refs)

                    # For compatibility with existing code, also create labels tensor
                    labels = torch.tensor([int(ref) for ref in batch_refs]).cuda()

                    # Print label distribution for debugging (only for first batch)
                    if len(tgt_pres) <= len(batch_preds):
                        unique_labels = torch.unique(labels)
                        print(f"\nUnique labels: {len(unique_labels)} out of {len(labels)}")
                        print(f"First few labels: {labels[:10]}")
                else:
                    # Handle the case where gt_gloss is a list
                    if isinstance(tgt_input['gt_gloss'], list):
                        try:
                            # First try to convert as integers (class indices)
                            labels = torch.tensor([int(gloss) for gloss in tgt_input['gt_gloss']]).cuda()
                        except ValueError:
                            # If that fails, assume they're gloss strings and use first token
                            print("Warning: Could not convert gloss labels to integers. Using default indices.")
                            labels = torch.tensor([0 for _ in tgt_input['gt_gloss']]).cuda()
                    else:
                        # If it's already a tensor
                        labels = tgt_input['gt_gloss'].cuda()

    # Compute final metrics for ISLR task
    if args.task == "ISLR":
        # Use the reference implementation's ISLR metrics
        from SLRT_metrics import islr_performance

        # Calculate per-instance and per-class accuracy
        top1_acc_pi, top1_acc_pc = islr_performance(tgt_refs, tgt_pres)

        # Update metrics
        metric_logger.meters['top1_acc_pi'].update(top1_acc_pi, n=1)
        metric_logger.meters['top1_acc_pc'].update(top1_acc_pc, n=1)

        # Print detailed results
        print("\nDetailed Evaluation Results:")
        print(f"Per-instance accuracy: {top1_acc_pi:.2f}%")
        print(f"Per-class accuracy: {top1_acc_pc:.2f}%")

        # Print prediction distribution
        pred_counts = {}
        for pred in tgt_pres:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1

        print("\nPrediction Distribution:")
        print(f"Number of unique predictions: {len(pred_counts)}")
        if pred_counts:
            most_common_pred = max(pred_counts.items(), key=lambda x: x[1])
            print(f"Most common prediction: class {most_common_pred[0]} ({most_common_pred[1]} times, {most_common_pred[1]*100/len(tgt_pres):.2f}% of all predictions)")

            # Map to gloss if possible
            if hasattr(model_without_ddp, 'gloss_to_idx'):
                idx_to_gloss = {idx: gloss for gloss, idx in model_without_ddp.gloss_to_idx.items()}
                if int(most_common_pred[0]) in idx_to_gloss:
                    print(f"Most common prediction gloss: '{idx_to_gloss[int(most_common_pred[0])]}'")

        # Print label distribution
        ref_counts = {}
        for ref in tgt_refs:
            ref_counts[ref] = ref_counts.get(ref, 0) + 1

        print("\nLabel Distribution:")
        print(f"Number of unique labels: {len(ref_counts)}")
        if ref_counts:
            most_common_label = max(ref_counts.items(), key=lambda x: x[1])
            print(f"Most common label: class {most_common_label[0]} ({most_common_label[1]} times, {most_common_label[1]*100/len(tgt_refs):.2f}% of all labels)")

            # Map to gloss if possible
            if hasattr(model_without_ddp, 'gloss_to_idx'):
                idx_to_gloss = {idx: gloss for gloss, idx in model_without_ddp.gloss_to_idx.items()}
                if int(most_common_label[0]) in idx_to_gloss:
                    print(f"Most common label gloss: '{idx_to_gloss[int(most_common_label[0])]}'")

    # Save detailed results if in evaluation mode
    if args.eval and utils.is_main_process() and args.task == "ISLR":
        # Create results dictionary
        results = {
            'top1_acc_pi': top1_acc_pi,
            'top1_acc_pc': top1_acc_pc,
            'num_unique_predictions': len(pred_counts),
            'num_unique_labels': len(ref_counts),
            'most_common_prediction': most_common_pred[0],
            'most_common_label': most_common_label[0]
        }

        # Save results to file
        results_path = os.path.join(args.output_dir, f'{phase}_detailed_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
