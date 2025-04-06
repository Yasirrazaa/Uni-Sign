import torch
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
from SLRT_metrics import translation_performance, islr_performance, wer_list, islr_performance_topk # Import new function
from transformers import get_scheduler
from config import *

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

        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch)

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
                # Use top1 per-instance accuracy for tracking best model
                if max_accuracy < test_stats["top1_acc_pi"]:
                    max_accuracy = test_stats["top1_acc_pi"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)
                
                # Print all accuracies (both per-instance and per-class)
                print(f"\nAccuracies on {len(dev_dataloader)} dev videos:")
                # Print per-instance accuracies
                pi_metrics = [f"Top-{k} PI: {test_stats[f'top{k}_acc_pi']:.2f}%"
                            for k in [1, 3, 5, 7, 10]]
                print("Per Instance:", ", ".join(pi_metrics))
                
                # Print per-class accuracies
                pc_metrics = [f"Top-{k} PC: {test_stats[f'top{k}_acc_pc']:.2f}%"
                            for k in [1, 3, 5, 7, 10]]
                print("Per Class:", ", ".join(pc_metrics))
                
                print(f'Max Top-1 PI accuracy: {max_accuracy:.2f}%')

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

def train_one_epoch(args, model, data_loader, optimizer, epoch):
    model.train()

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
        stack_out = model(src_input, tgt_input)

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

def evaluate(args, data_loader, model, model_without_ddp, phase):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    target_dtype = None
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16

    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []

        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            if target_dtype != None:
                for key in src_input.keys():
                    if isinstance(src_input[key], torch.Tensor):
                        src_input[key] = src_input[key].to(target_dtype).cuda()

            if args.task == "CSLR":
                tgt_input['gt_sentence'] = tgt_input['gt_gloss']
            stack_out = model(src_input, tgt_input)

            total_loss = stack_out['loss']
            metric_logger.update(loss=total_loss.item())

            # --- Start ISLR Top-K Modification ---
            if args.task == "ISLR":
                # Collect logits and target IDs instead of generating text
                tgt_pres.append(stack_out['logits']) # Logits for first token prediction
                tgt_refs.append(stack_out['target_ids']) # Target ID for first token
            # --- End ISLR Top-K Modification ---
            else: # Keep original logic for SLT/CSLR
                output = model_without_ddp.generate(stack_out,
                                                    max_new_tokens=100,
                                                    num_beams = 4,
                            )

                for i in range(len(output)):
                    tgt_pres.append(output[i])
                    tgt_refs.append(tgt_input['gt_sentence'][i])

    # --- Start ISLR Top-K Modification ---
    if args.task == "ISLR":
        # Concatenate collected logits and targets from all batches
        all_logits = torch.cat(tgt_pres, dim=0)
        all_targets = torch.cat(tgt_refs, dim=0)
        # Calculate Top-K accuracies
        topk_accuracies = islr_performance_topk(all_logits, all_targets, ks=[1, 3, 5, 7, 10])
        for k, acc in topk_accuracies.items():
             metric_logger.meters[k].update(acc)
    # --- End ISLR Top-K Modification ---
    else: # Keep original logic for SLT/CSLR
        tokenizer = model_without_ddp.mt5_tokenizer
        padding_value = tokenizer.eos_token_id

        # Handle potential empty tgt_pres list if evaluation set is small/empty
        if tgt_pres:
            max_len_in_batch = max(len(t) for t in tgt_pres)
            pad_tensor = torch.ones(max_len_in_batch - len(tgt_pres[0]), device=tgt_pres[0].device, dtype=torch.long) * padding_value
            tgt_pres[0] = torch.cat((tgt_pres[0], pad_tensor), dim=0)
            tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
            tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)
        else:
            tgt_pres = [] # Ensure tgt_pres is an empty list if no predictions were made

        # fix mt5 tokenizer bug
        if args.dataset == 'CSL_Daily' and args.task == "SLT":
            tgt_pres = [' '.join(list(r.replace(" ",'').replace("\n",''))) for r in tgt_pres]
            tgt_refs = [' '.join(list(r.replace("，", ',').replace("？","?").replace(" ",''))) for r in tgt_refs]

        if args.task == "SLT":
            bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)
            # Indent these lines to be inside the SLT block
            for k,v in bleu_dict.items():
                metric_logger.meters[k].update(v)
            metric_logger.meters['rouge'].update(rouge_score)

    # Removed ISLR section here as it's handled above by islr_performance_topk

        # Un-indent this block to align with the 'if args.task == "SLT":'
        elif args.task == "CSLR":
            wer_results = wer_list(hypotheses=tgt_pres, references=tgt_refs)
            print(wer_results)
            for k,v in wer_results.items():
                metric_logger.meters[k].update(v)

    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()

    # Only write temporary prediction/reference files for non-ISLR tasks,
    # as tgt_pres/tgt_refs contain tensors for ISLR after the Top-K modification.
    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval and args.task != "ISLR":
        # Ensure tgt_pres and tgt_refs are lists of strings here (should be true for SLT/CSLR)
        if tgt_pres and isinstance(tgt_pres[0], str):
            with open(args.output_dir+f'/{phase}_tmp_pres.txt','w') as f:
                for i in range(len(tgt_pres)):
                    f.write(tgt_pres[i]+'\n')
        if tgt_refs and isinstance(tgt_refs[0], str):
            with open(args.output_dir+f'/{phase}_tmp_refs.txt','w') as f:
                for i in range(len(tgt_refs)):
                    f.write(tgt_refs[i]+'\n')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)