"""
Finetuning script for WLASL with pretrained Uni-Sign checkpoints and future masking
"""
import os
import time
import torch
from utils import (
    init_distributed_mode, get_args_parser, set_seed,
    save_on_master, is_main_process,
    MetricLogger, SmoothedValue
)
from unisign_wlasl import WLASLUniSign
from datasets import S2T_Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import datetime
import json
from timm.optim import create_optimizer
from transformers import get_scheduler
import math
import sys

def train_one_epoch(model, data_loader, optimizer, lr_scheduler, epoch, args):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Move inputs to cuda
        for key in src_input:
            if isinstance(src_input[key], torch.Tensor):
                src_input[key] = src_input[key].cuda()
        # Don't move gt_gloss to cuda since they are strings

        # Forward pass
        outputs = model(src_input, tgt_input)
        logits = outputs['logits']
        loss = outputs['loss']

        if loss is None:
            # Handle loss computation here if model doesn't compute it
            loss_fct = torch.nn.CrossEntropyLoss()
            batch_glosses = tgt_input['gt_gloss']
            label_indices = [model.module.gloss_to_idx.get(g.split()[0], 0) for g in batch_glosses]
            labels = torch.tensor(label_indices, device=logits.device)
            loss = loss_fct(logits, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        lr_scheduler.step()  # Update learning rate scheduler every step

        # Log metrics
        current_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=current_lr)
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, data_loader, args):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    metric_logger.add_meter('acc1', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('acc5', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    
    num_processed_samples = 0
    num_correct1 = 0
    num_correct5 = 0

    for src_input, tgt_input in metric_logger.log_every(data_loader, 10, header):
        # Move inputs to cuda
        for key in src_input:
            if isinstance(src_input[key], torch.Tensor):
                src_input[key] = src_input[key].cuda()
        # Don't move gt_gloss to cuda since they are strings

        # Forward pass
        outputs = model(src_input, tgt_input)
        logits = outputs['logits']
# Get top-k predictions
maxk = max((1, 5))
_, pred = logits.topk(maxk, 1, True, True)
batch_size = len(tgt_input['gt_gloss'])

# Calculate accuracies for batch
correct1 = 0
correct5 = 0
for i in range(batch_size):
    true_gloss = tgt_input['gt_gloss'][i].split()[0] if tgt_input['gt_gloss'][i] else ""
    true_idx = model.module.gloss_to_idx.get(true_gloss, -1)
    
    if true_idx >= 0:  # Skip invalid glosses
        pred_indices = pred[i].tolist()
        if pred_indices[0] == true_idx:
            correct1 += 1
        if true_idx in pred_indices:
            correct5 += 1
        
        # Debug output for first few samples
        if i < 3 and num_processed_samples % 100 == 0:
            pred_gloss = [k for k,v in model.module.gloss_to_idx.items() if v == pred_indices[0]][0]
            print(f"\nSample {i}: True={true_gloss}, Pred={pred_gloss}")

num_correct1 += correct1
num_correct5 += correct5
num_processed_samples += batch_size

acc1 = correct1 / batch_size * 100
acc5 = correct5 / batch_size * 100
metric_logger.meters['acc1'].update(acc1, n=batch_size)
metric_logger.meters['acc5'].update(acc5, n=batch_size)
        metric_logger.meters['acc1'].update(acc1, n=batch_size)

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print(f'* Acc@1 {metric_logger.acc1.global_avg:.3f}')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main(args):
    # Initialize distributed training
    init_distributed_mode(args)
    print(args)
    set_seed(args.seed)
    # Create datasets
    print("Loading training dataset...")
    train_data = S2T_Dataset(path=args.train_label_path, args=args, phase='train')

    # Collect all unique glosses from training set
    print("Collecting gloss vocabulary...")
    all_glosses = set()
    for _, _, _, gloss, _ in train_data:
        if gloss:  # Skip empty glosses
            # Split space-separated glosses and add each one
            gloss_tokens = gloss.split()
            all_glosses.update(gloss_tokens)
    print(f"Found {len(all_glosses)} unique glosses in training set")
    print("Sample glosses:", list(all_glosses)[:5])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_data.collate_fn,
        sampler=train_sampler,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    val_data = S2T_Dataset(path=args.val_label_path, args=args, phase='dev')
    val_sampler = torch.utils.data.SequentialSampler(val_data)
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=val_data.collate_fn,
        sampler=val_sampler,
        pin_memory=args.pin_mem
    )

    # Create model
    print("Creating model")
    model = WLASLUniSign(args, num_classes=args.num_classes, gloss_vocab=all_glosses)
    model.cuda()
    print(f"Model initialized with {args.num_classes} output classes")
    
    # Load pretrained weights
    if args.finetune:
        print('Loading checkpoint:', args.finetune)
        checkpoint = torch.load(args.finetune, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print('Missing keys:', msg.missing_keys)
        print('Unexpected keys:', msg.unexpected_keys)

    # Setup distributed training
    model_without_ddp = model
    if args.distributed:
        # Convert batch norm layers
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # Create optimizer with explicit learning rate
    args.opt_betas = (0.9, 0.999)  # Add default Adam betas
    args.opt_eps = 1e-8  # Add default epsilon
    args.weight_decay = 0.01  # Add default weight decay
    optimizer = create_optimizer(args, model_without_ddp)
    
    # Calculate scheduler steps
    num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    num_warmup_steps = int(args.warmup_epochs * num_update_steps_per_epoch)
    num_training_steps = int(args.epochs * num_update_steps_per_epoch)
    
    print(f"Warmup steps: {num_warmup_steps}")
    print(f"Total steps: {num_training_steps}")
    
    # Create scheduler with explicit steps
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Training loop
    output_dir = Path(args.output_dir)
    print(f"Start training for {args.epochs} epochs, lr={args.lr}, warmup_epochs={args.warmup_epochs}")
    print(f"GPU: {args.gpu}")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, train_loader, optimizer, epoch, args
        )

        if args.output_dir:
            checkpoint_path = output_dir / f'checkpoint_{epoch}.pth'
            save_on_master({'model': model_without_ddp.state_dict()}, checkpoint_path)

        test_stats = evaluate(model, val_loader, args)
        print(f"Accuracy of the network on the {len(val_loader)} test images: {test_stats['acc1']:.1f}%")
        
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                checkpoint_path = output_dir / 'best_checkpoint.pth'
                save_on_master({'model': model_without_ddp.state_dict()}, checkpoint_path)

        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': sum(p.numel() for p in model.parameters())
        }

        if args.output_dir and is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = get_args_parser()
    # Add WLASL specific arguments
    parser.add_argument('--num_classes', default=2000, type=int)
    parser.add_argument('--train_label_path', default='data/WLASL/labels-2000.train')
    parser.add_argument('--val_label_path', default='data/WLASL/labels-2000.dev')
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
