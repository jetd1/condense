import torch
import numpy as np
import wandb
import logging
import os
import torch.distributed.nn
import torch.nn.functional as F
from tqdm import tqdm

from backbone3.minkunet import MinkUNet14
from utils.arg import get_config


class Trainer(object):
    def __init__(self, rank, config, model, image_proj, optimizer, scheduler, train_loader,
                 modelnet40_loader, scanobjectnn_loader):
        self.rank = rank
        self.config = config
        self.model = model
        self.image_proj = image_proj
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.modelnet40_loader = modelnet40_loader
        self.scanobjectnn_loader = scanobjectnn_loader
        self.epoch = 0
        self.step = 0
        self.best_img_contras_acc = 0
        self.best_text_contras_acc = 0
        self.best_modelnet40_overall_acc = 0
        self.best_modelnet40_class_acc = 0
        self.best_lvis_acc = 0

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.config.training.use_image_proj:
            self.image_proj.load_state_dict(checkpoint['image_proj'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.config.training.use_openclip_optimizer_scheduler == False:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_img_contras_acc = checkpoint['best_img_contras_acc']
        self.best_text_contras_acc = checkpoint['best_text_contras_acc']
        self.best_modelnet40_overall_acc = checkpoint['best_modelnet40_overall_acc']
        self.best_modelnet40_class_acc = checkpoint['best_modelnet40_class_acc']
        self.best_lvis_acc = checkpoint['best_lvis_acc']

        logging.info("Loaded checkpoint from {}".format(path))
        logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))
        logging.info("----Best img contras acc: {}".format(self.best_img_contras_acc))
        logging.info("----Best text contras acc: {}".format(self.best_text_contras_acc))
        logging.info("----Best modelnet40 overall acc: {}".format(self.best_modelnet40_overall_acc))
        logging.info("----Best modelnet40 class acc: {}".format(self.best_modelnet40_class_acc))
        logging.info("----Best lvis acc: {}".format(self.best_lvis_acc))

    def contras_loss(self, feat1, feat2, logit_scale=1, mask=None):
        if self.config.ngpu > 1:
            feat1 = F.normalize(feat1, dim=1)
            feat2 = F.normalize(feat2, dim=1)
            all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
            all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
            logits = logit_scale * all_feat1 @ all_feat2.T
        else:
            logits = logit_scale * F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).T
        if mask is not None:
            logits = logits * mask
        labels = torch.arange(logits.shape[0]).to(self.config.device)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss, accuracy

    def train_one_epoch(self):
        self.model.train()
        if self.config.training.use_image_proj:
            self.image_proj.train()

        text_contras_acc_list = []
        img_contras_acc_list = []
        if self.config.training.use_mask:
            k = self.config.dataset.negative_sample_num
            s = self.config.dataset.train_batch_size
            mask1 = np.eye(k * s).astype(np.bool)
            mask2 = np.kron(np.eye(s), np.ones((k, k))).astype(np.bool)
            mask_other = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)

        for data in self.train_loader:
            self.step += 1
            self.optimizer.zero_grad()
            loss = 0
            if not self.config.model.get("use_dense", False):
                pred_feat = self.model(data['xyz'], data['features'], \
                                       device=self.config.device, \
                                       quantization_size=self.config.model.voxel_size)
            else:
                pred_feat = self.model(data['xyz_dense'], data['features_dense'])
            idx = data['has_text_idx']

            text_feat = torch.vstack(data['text_feat']).to(self.config.device)
            img_feat = torch.vstack(data['img_feat']).to(self.config.device)

            if self.config.training.use_mask:
                img_text_sim = F.normalize(img_feat, dim=-1) @ F.normalize(text_feat, dim=-1).T
                mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
                mask = torch.logical_or(mask, mask_other).detach()
            else:
                mask = None

            if len(idx) > 0:
                text_contras_loss, text_contras_acc = self.contras_loss(pred_feat[idx], text_feat, mask=mask)
                loss += text_contras_loss * self.config.training.lambda_text_contras
                text_contras_acc_list.append(text_contras_acc.item())

            if self.config.training.use_image_proj:
                img_feat = self.image_proj(img_feat)
            img_contras_loss, img_contras_acc = self.contras_loss(pred_feat, img_feat, mask=mask)
            loss += img_contras_loss * self.config.training.lambda_img_contras
            loss.backward()
            self.optimizer.step()
            if self.config.training.use_openclip_optimizer_scheduler:
                self.scheduler(self.step)
            else:
                self.scheduler.step()

            img_contras_acc_list.append(img_contras_acc.item())

            if self.rank == 0 and self.step % self.config.training.log_freq == 0:
                try:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/text_contras_loss": text_contras_loss.item() if len(idx) > 0 else 0,
                        "train/img_contras_loss": img_contras_loss.item(),
                        "train/text_contras_acc": text_contras_acc.item() if len(idx) > 0 else 0,
                        "train/img_contras_acc": img_contras_acc.item(),
                        "train/lr": self.optimizer.param_groups[0]['lr'],
                        "train/epoch": self.epoch,
                        "train/step": self.step,
                        "train/has_text": len(idx),
                        "train/filtered_pair": (mask == False).sum() if mask is not None else 0
                    })
                except:
                    print("wandb log error", flush=True)
        if self.rank == 0:
            logging.info('Train: text_cotras_acc: {0} image_contras_acc: {1}' \
                         .format(np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0,
                                 np.mean(img_contras_acc_list)))

    def save_model(self, name):
        torch.save({
            "state_dict": self.model.state_dict(),
            "image_proj": self.image_proj.state_dict() if self.config.training.use_image_proj else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.config.training.use_openclip_optimizer_scheduler == False else None,
            "epoch": self.epoch,
            "step": self.step,
            "best_img_contras_acc": self.best_img_contras_acc,
            "best_text_contras_acc": self.best_text_contras_acc,
            "best_modelnet40_overall_acc": self.best_modelnet40_overall_acc,
            "best_modelnet40_class_acc": self.best_modelnet40_class_acc,
            "best_lvis_acc": self.best_lvis_acc,
        }, os.path.join(self.config.ckpt_dir, '{}.pt'.format(name)))

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res, correct

    def train(self):
        for epoch in range(self.epoch, self.config.training.max_epoch):
            self.epoch = epoch
            if self.rank == 0:
                logging.info("Epoch: {}".format(self.epoch))
            self.train_one_epoch()
            if self.rank == 0:
                self.save_model('latest')
                self.test_modelnet40()
            if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                self.save_model('epoch_{}'.format(self.epoch))

    def test_modelnet40(self):
        self.model.eval()
        clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).to(self.config.device)
        per_cat_correct = torch.zeros(40).to(self.config.device)
        per_cat_count = torch.zeros(40).to(self.config.device)
        category2idx = self.modelnet40_loader.dataset.category2idx

        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in self.modelnet40_loader:
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)

                for i in range(40):
                    idx = (labels == i)
                    if idx.sum() > 0:
                        per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                        per_cat_count[i] += idx.sum()
        topk_acc, correct = self.accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1, 3, 5,))

        overall_acc = per_cat_correct.sum() / per_cat_count.sum()
        per_cat_acc = per_cat_correct / per_cat_count
        # for i in range(40):
        #    print(idx2category[i], per_cat_acc[i])

        if overall_acc > self.best_modelnet40_overall_acc:
            self.best_modelnet40_overall_acc = overall_acc
            self.save_model('best_modelnet40_overall')
        if per_cat_acc.mean() > self.best_modelnet40_class_acc:
            self.best_modelnet40_class_acc = per_cat_acc.mean()
            self.save_model('best_modelnet40_class')

        logging.info('Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})'.format(overall_acc,
                                                                                         self.best_modelnet40_overall_acc,
                                                                                         per_cat_acc.mean(),
                                                                                         self.best_modelnet40_class_acc))
        logging.info(
            'Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(), topk_acc[1].item(),
                                                                                topk_acc[2].item()))
        wandb.log({"test/epoch": self.epoch,
                   "test/step": self.step,
                   "test/ModelNet40_overall_acc": overall_acc,
                   "test/ModelNet40_class_acc": per_cat_acc.mean(),
                   "test/top3_acc": topk_acc[1],
                   "test/top5_acc": topk_acc[2], })

    def test_scanobjectnn(self):
        self.model.eval()
        clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
        per_cat_correct = torch.zeros(15).to(self.config.device)
        per_cat_count = torch.zeros(15).to(self.config.device)
        category2idx = self.scanobjectnn_loader.dataset.category2idx

        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in self.scanobjectnn_loader:
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)
                # calculate per class accuracy
                for i in range(15):
                    idx = (labels == i)
                    if idx.sum() > 0:
                        per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                        per_cat_count[i] += idx.sum()

        topk_acc, correct = self.accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1, 3, 5,))

        overall_acc = per_cat_correct.sum() / per_cat_count.sum()
        per_cat_acc = per_cat_correct / per_cat_count

        logging.info('Test ScanObjectNN: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
        logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
                                                                                           topk_acc[1].item(),
                                                                                           topk_acc[2].item()))
        wandb.log({"test_scanobjectnn/epoch": self.epoch,
                   "test_scanobjectnn/step": self.step,
                   "test_scanobjectnn/overall_acc": overall_acc,
                   "test_scanobjectnn/class_acc": per_cat_acc.mean(),
                   "test_scanobjectnn/top3_acc": topk_acc[1],
                   "test_scanobjectnn/top5_acc": topk_acc[2], })

    # def test_scannet(self):
    #     self.model.eval()
    #     if self.config.training.use_text_proj:
    #         self.text_proj.eval()
    #     clip_text_feat = torch.from_numpy(self.scannet_loader.dataset.clip_cat_feat).to(self.config.device)
    #     if self.config.training.use_text_proj:
    #         clip_text_feat = self.text_proj(clip_text_feat)
    #     per_cat_correct = torch.zeros(20).to(self.config.device)
    #     per_cat_count = torch.zeros(20).to(self.config.device)
    #     category2idx = self.scannet_loader.dataset.category2idx
    #     idx2category = {v: k for k, v in category2idx.items()}
    #     logits_all = []
    #     labels_all = []
    #     with torch.no_grad():
    #         for data in self.scannet_loader:
    #             if not self.config.model.get("use_dense", False):
    #                 pred_feat = self.model(data['xyz'], data['features'], \
    #                                        device=self.config.device, \
    #                                        quantization_size=self.config.model.voxel_size)
    #             else:
    #                 pred_feat = self.model(data['xyz_dense'], data['features_dense'])
    #             logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
    #             labels = data['category'].to(self.config.device)
    #             logits_all.append(logits.detach())
    #             labels_all.append(labels)
    #             # calculate per class accuracy
    #             for i in range(20):
    #                 idx = (labels == i)
    #                 if idx.sum() > 0:
    #                     per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
    #                     per_cat_count[i] += idx.sum()
    #
    #     topk_acc, correct = self.accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1, 3, 5,))
    #     overall_acc = per_cat_correct.sum() / per_cat_count.sum()
    #     per_cat_acc = per_cat_correct / per_cat_count
    #
    #     logging.info('Test ScanNet: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
    #     logging.info('Test ScanNet: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
    #                                                                                        topk_acc[1].item(),
    #                                                                                        topk_acc[2].item()))
    #     wandb.log({"test_scannet/epoch": self.epoch,
    #                  "test_scannet/step": self.step,
    #                  "test_scannet/overall_acc": overall_acc,
    #                  "test_scannet/class_acc": per_cat_acc.mean(),
    #                  "test_scannet/top3_acc": topk_acc[1],
    #                  "test_scannet/top5_acc": topk_acc[2], })
    #
    # def test_s3dis(self):
    #     self.model.eval()
    #     if self.config.training.use_text_proj:
    #         self.text_proj.eval()
    #     clip_text_feat = torch.from_numpy(self.s3dis_loader.dataset.clip_cat_feat).to(self.config.device)
    #     if self.config.training.use_text_proj:
    #         clip_text_feat = self.text_proj(clip_text_feat)
    #     per_cat_correct = torch.zeros(13).to(self.config.device)
    #     per_cat_count = torch.zeros(13).to(self.config.device)
    #     category2idx = self.s3dis_loader.dataset.category2idx
    #     idx2category = {v: k for k, v in category2idx.items()}
    #     logits_all = []
    #     labels_all = []
    #     with torch.no_grad():
    #         for data in self.s3dis_loader:
    #             if not self.config.model.get("use_dense", False):
    #                 pred_feat = self.model(data['xyz'], data['features'], \
    #                                        device=self.config.device, \
    #                                        quantization_size=self.config.model.voxel_size)
    #             else:
    #                 pred_feat = self.model(data['xyz_dense'], data['features_dense'])
    #             logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
    #             labels = data['category'].to(self.config.device)
    #             logits_all.append(logits.detach())
    #             labels_all.append(labels)
    #             # calculate per class accuracy
    #             for i in range(13):
    #                 idx = (labels == i)
    #                 if idx.sum() > 0:
    #                     per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
    #                     per_cat_count[i] += idx.sum()
    #
    #     topk_acc, correct = self.accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1, 3, 5,))
    #     overall_acc = per_cat_correct.sum() / per_cat_count.sum()
    #     per_cat_acc = per_cat_correct / per_cat_count
    #
    #     logging.info('Test S3DIS: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
    #     logging.info('Test S3DIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
    #                                                                                        topk_acc[1].item(),
    #                                                                                        topk_acc[2].item()))
    #     wandb.log({"test_s3dis/epoch": self.epoch,
    #                     "test_s3dis/step": self.step,
    #                     "test_s3dis/overall_acc": overall_acc,
    #                     "test_s3dis/class_acc": per_cat_acc.mean(),
    #                     "test_s3dis/top3_acc": topk_acc[1],
    #                     "test_s3dis/top5_acc": topk_acc[2], })
    #

import sys
import os
import logging
import shutil
import data
import MinkowskiEngine as ME
import torch
import wandb
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size):
    setup(rank, world_size)

    config, _, _ = get_config()
    if config.autoresume:
        config.trial_name = config.get('trial_name') + "@autoresume"
    else:
        config.trial_name = config.get('trial_name') + datetime.now().strftime('@%Y%m%d-%H%M%S')
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
    config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')

    if rank == 0:
        os.makedirs(os.path.join(config.exp_dir, config.trial_name), exist_ok=config.autoresume)
        os.makedirs(config.ckpt_dir, exist_ok=True)
        if os.path.exists(config.code_dir):
            shutil.rmtree(config.code_dir)
        shutil.copytree("./src", config.code_dir)

    config.device = 'cuda:{0}'.format(rank)

    if rank == 0:
        config.log_path = config.get('log_path') or os.path.join(config.exp_dir, config.trial_name, 'log.txt')
        config.log_level = logging.DEBUG if config.debug else logging.INFO
        logging.info("Using {} GPU(s).".format(config.ngpu))
        wandb.init(project=config.project_name, name=config.trial_name, config=config)

    if config.train:
        model = MinkUNet14(in_channels=3, out_channels=config.n_channels).to(config.device)
        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            logging.info(model)
            logging.info("Network:{}, Number of parameters: {}".format(config.model.name, total_params))

        torch.cuda.set_device(rank)
        model.cuda(rank)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        if config.model.name.startswith('Mink'):
            model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)  # minkowski only
            logging.info("Using MinkowskiSyncBatchNorm")
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logging.info("Using SyncBatchNorm")

        image_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(config.device)
        image_proj = DDP(image_proj, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        train_loader = data.make(config, 'train', rank, world_size)

        if rank == 0:
            modelnet40_loader = data.make_modelnet40test(config)
            scanobjectnn_loader = data.make_scanobjectnntest(config)
        else:
            modelnet40_loader = None
            scanobjectnn_loader = None

        if rank == 0:
            if train_loader is not None:
                logging.info("Train iterations: {}".format(len(train_loader)))

        params = list(model.parameters()) + list(image_proj.parameters())
        optimizer = torch.optim.AdamW(params, lr=config.training.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.training.lr_decay_step,
                                                    gamma=config.training.lr_decay_rate)

        trainer = Trainer(rank, config, model, image_proj, optimizer, scheduler,
                          train_loader, modelnet40_loader, scanobjectnn_loader)

        if config.resume is not None:
            trainer.load_from_checkpoint(config.resume)
        elif config.autoresume:
            if os.path.exists(os.path.join(config.ckpt_dir, '{}.pt'.format('latest'))):
                trainer.load_from_checkpoint(os.path.join(config.ckpt_dir, '{}.pt'.format('latest')))

        trainer.train()

    if rank == 0:
        wandb.finish()
    cleanup()


if __name__ == '__main__':
    mp.spawn(
        main,
        args=(0, 1),
        nprocs=1
    )
