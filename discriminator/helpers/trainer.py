"""Wrapper trainer class for training our models."""

import sys
import os
import re
import time
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from helpers.logger import Logger
from helpers.processor import SpecProcessor
from utils.augment import add_random_noise, degrade_some, frame_dropout
from utils.plot import plot_alignment, plot_spectrogram, plot_projection
from utils.util import last_checkpoint, get_symbols

from datasets.dataset import SpeechDataset, SpeechCollate
from models import CC, CCLoss
from models import Discriminator, NoamScheduler

class Trainer:
    def __init__(self,
                 model=None,
                 datasets=None,
                 criterion=None,
                 optimizers=None,
                 checkpoint=None,
                 device=None
                 ):
        # model, criterion, optimizer
        self.model = model
        self.criterion = criterion
        self.optimizer, self.scheduler = optimizers

        # dataset
        self.trainset, self.validset = datasets

        # device
        self.device = device
        self.model.to(self.device)
        print(f'Model sent to {self.device}')

        # helper vars
        self.checkpoint = None
        self.epoch, self.step = 0, 0
        if checkpoint is not None:
            self.load_checkpoint(self.checkpoint)

    def to_device(self, device):
        print(f'Sending network to {device}')
        self.device = device
        self.model.to(device)
        return self

    def setup_seed(self, seed):
        # python
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        # numpy
        np.random.seed(seed)
        # pytorch
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed) # current gpu
            #torch.cuda.manual_seed_all(seed) # all gpus

    def save_checkpoint(self,checkpoint):
        self.checkpoint = checkpoint
        print("Saving the checkpoint file '%s'..." % self.checkpoint)
        torch.save(
            {
                'epoch': self.epoch,
                'step': self.step,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
            },
            self.checkpoint)

    def load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        print("Loaded checkpoint epoch=%d step=%d" % (self.epoch, self.step))

        self.checkpoint = None  # prevent overriding old checkpoint

    def dataloader(self, dataset, batch_size, collate_fn, num_workers=8, **kwargs):
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                          num_workers=0 if sys.platform.startswith('win') else num_workers,
                          shuffle=True, pin_memory=False, drop_last=False, **kwargs)



class CCTrainer(Trainer):
    def __init__(self,
                 hparams,device=torch.device('cuda')
                 ):
        self._create_hparams(hparams)

        model = CC(self.model_hparams)

        if (self.data_hparams.load_data_from_disk is False) \
            or (self.data_hparams.compute_statistics is True):
            listfile = os.path.join(self.data_hparams.etl_path, self.data_hparams.wave_files)
            with open(listfile, 'r', encoding='utf-8') as f:
                filelist = [line.strip() for line in f]
            if self.data_hparams.load_data_from_disk is True:
                filelist = [re.sub(r'(.*?)wav(.*?).wav', r'\1mel\2.npy', line) for line in filelist]
            done = SpecProcessor(self.data_hparams)(filelist)
            self.data_hparams.load_data_from_disk = True
            self.data_hparams.compute_statistics = False
        trainset = SpeechDataset(self.data_hparams.training_files, self.data_hparams)
        validset = SpeechDataset(self.data_hparams.validation_files, self.data_hparams)
        datasets = (trainset, validset)
        criterion = CCLoss(self.model_hparams)


        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.optim_hparams.lr,
                                     betas=self.optim_hparams.betas,
                                     eps=self.optim_hparams.eps,
                                     weight_decay=self.optim_hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer,
                                      factor=self.optim_hparams.factor,
                                      patience=self.optim_hparams.patience)
        optimizers = (optimizer, scheduler)

        checkpoint = self.train_hparams.checkpoint or last_checkpoint(self.train_hparams.out_path)

        super(CCTrainer, self).__init__(
            model=model,
            datasets=datasets,
            criterion=criterion,
            optimizers=optimizers,
            checkpoint=checkpoint,
            device=device
        )
    def _create_hparams(self, hparams):
        # Data Hparams
        self.data_hparams = hparams.data
        text_hparams, audio_hparams = hparams.text, hparams.audio
        symbols_list = os.path.join(hparams.data.etl_path, hparams.data.symbols_list)
        text_hparams.symbols, _ = get_symbols(symbols_list)
        text_hparams.symbols = text_hparams.specials + text_hparams.symbols + text_hparams.punctuations
        self.data_hparams.update(text_hparams)
        self.data_hparams.update(audio_hparams)
        # Model HParams
        self.model_hparams = hparams.model
        self.model_hparams.n_symbols = len(text_hparams.symbols)
        # Train HParams
        self.train_hparams = hparams.train
        # Optim HParams
        self.optim_hparams = hparams.optim
    def fit(self, checkpoint=None, loggers=None):
        # random seed
        self.setup_seed(self.train_hparams.seed)
        # cudnn settings
        if self.device.type == 'cuda':
            torch.backends.cudnn.enabled = self.train_hparams.cudnn_enabled
            torch.backends.cudnn.benchmark = self.train_hparams.cudnn_benchmark
            print('cuDNN Enabled:', self.train_hparams.cudnn_enabled)
            print('cuDNN Benchmark:', self.train_hparams.cudnn_benchmark)



        self.loggers = loggers or Logger(self.train_hparams)

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

        # save initial model weights
        self.save_checkpoint(remove=False)

        collate_fn = SpeechCollate(self.data_hparams, device=torch.device('cpu'))


        train_loader = self.dataloader(self.trainset, batch_size=self.data_hparams.batch_size,collate_fn=collate_fn)
        valid_loader = self.dataloader(self.validset, batch_size=self.data_hparams.batch_size,collate_fn=collate_fn)


        for e in range(self.epoch + 1, self.epoch + 1 + self.train_hparams.epochs):
            self.epoch = e

            train_losses = self._train_epoch(train_loader)
            valid_losses = self._validate(valid_loader)

            self.scheduler.step(sum(valid_losses.values()))

            if self.epoch % self.train_hparams.epochs_per_chkpt == 0:
                # checkpoint at every n epoch
                self.save_checkpoint(remove=False)

            self.loggers.log_epoch('train', 'epoch', self.epoch, train_losses)
            self.loggers.log_epoch('valid', 'epoch', self.epoch, valid_losses)

            print(f'Epoch {e} ')
            print(f"[ Train ] {', '.join([f'{k}: {v:.4f}' for k, v in train_losses.items()])}")
            print(f"[ Valid ] {', '.join([f'{k}: {v:.4f}' for k, v in valid_losses.items()])}")

        # save final checkpoint

        self.save_checkpoint(remove=False)

    LOSSES = ['mse', 'mae', 'ssim', 'drn', 'cos', 'spd','f0', 'nrg'] # the loss keys


    def _train_epoch(self, dataloader):
        self.model.train()

        running_losses = {key: 0.0 for key in self.LOSSES}


        pbar = tqdm(dataloader, unit="audios", unit_scale=dataloader.batch_size, \
                    disable=self.train_hparams.disable_progress_bar)
        for it, batch in enumerate(pbar, start=1):
            self.optimizer.zero_grad()
            inputs, targets, lengths = self.model.parse_batch(batch,device=self.device)
            outputs = self.model(inputs)
            outputs_and_targets = (outputs, targets, lengths)

            losses = self.criterion(outputs_and_targets, use_dtw=False)
            loss = sum(losses.values())
            loss.backward()

            if self.step % self.optim_hparams.grad_acc_step == 0:
                # clipping gradients to avoid gradient explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.optim_hparams.grad_clip_thresh)
                self.optimizer.step()



            losses = {key: val.item() for key, val in losses.items()}
            running_losses = {key: running_losses[key] + val for key, val in losses.items()}
            epoch_losses = {key: val / it for key, val in running_losses.items()}


            # update the progress bar
            pbar.set_postfix({
                key: f'{val:.5f}' for key, val in epoch_losses.items()
            })

            # logging
            learning_rate = self._get_learning_rate(self.optimizer)

            self.loggers.log_train('train','step', self.step, losses, lr_dict={'lr',learning_rate})

            self.step += 1


        return epoch_losses

    def _get_learning_rate(self, optimizer):
        learning_rate = 0.0
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']

        return learning_rate

    def _validate(self, dataloader):
        self.model.eval()

        running_losses = {key: 0.0 for key in self.LOSSES}

        pbar = tqdm(dataloader, unit="audios", unit_scale=dataloader.batch_size, \
                    disable=self.train_hparams.disable_progress_bar)
        for it, batch in enumerate(pbar, start=1):
            inputs, targets, lengths = self.model.parse_batch(batch, device=self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
            outputs_and_targets = (outputs, targets, lengths)

            losses = self.criterion(outputs_and_targets, use_dtw=False)

            losses = {key: val.item() for key, val in losses.items()}
            running_losses = {key: running_losses[key] + val for key, val in losses.items()}
            epoch_losses = {key: val / it for key, val in running_losses.items()}

            # plot spectrogram
            spec_outputs, specs = outputs[0], targets[0]
            spec_outputs, specs = spec_outputs.cpu().detach(), specs.cpu().detach()
            input_lengths, output_lengths = lengths
            predicted_lengths = outputs[1]
            index = random.randint(0, specs.size(0) - 1) # -1
            output_length, input_length = output_lengths[index].item(), input_lengths[index].item()
            predicted_length = predicted_lengths[index].item()
            specs_fig, _ = plot_spectrogram(spec_outputs[index, :predicted_length, :],
                                           target=specs[index, :output_length, :])

            self.loggers.log_train('valid','step', self.step, losses, image_dict={'spec':specs_fig})


        return epoch_losses

    # overwrite load_checkpoint to remove prev one and create soft-link
    def save_checkpoint(self,remove=True):
        # remove prev one
        if remove and self.checkpoint is not None:
            os.remove(self.checkpoint)

        checkpoint = os.path.join(self.train_hparams.out_path, f'{time.strftime("%Y-%m-%d")}_chkpt_step{self.step:05d}.pth')
        super().save_checkpoint(checkpoint)

        # create soft-link
        latest_checkpoint = os.path.join(self.train_hparams.out_path, 'as_latest')
        os.system("ln -fs {} {}".format(self.checkpoint,latest_checkpoint))
        return checkpoint

    # overwrite load_checkpoint to avoid loading error
    def load_checkpoint(self, checkpoint):
        try:
            super().load_checkpoint(checkpoint)
        except Exception as err:
            print(err)
            print("Loaded checkpoint '{}' failed !".format(checkpoint))
        finally:
            return checkpoint


class CCGANTrainer(Trainer):
    def __init__(self,
                 hparams, device=torch.device('cuda')
                 ):
        super(CCGANTrainer, self).__init__(hparams,device=device)
        self._create_hparams(hparams)


        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.disc_hparams.lr,
                                     betas=self.disc_hparams.betas,
                                     eps=self.disc_hparams.eps,
                                     weight_decay=self.disc_hparams.weight_decay)
        self.scheduler = NoamScheduler(self.optimizer,
                                      n_warmup=self.disc_hparams.n_warmup+self.disc_hparams.n_train_start,
                                      init_scale=self.disc_hparams.init_scale)

        # Prepare Discriminator
        self.d_model = Discriminator(self.disc_hparams).to(self.device)
        self.d_criterion = nn.MSELoss()
        self.d_optimizer = torch.optim.Adam(self.d_model.parameters(),
                                     lr=self.disc_hparams.lr,
                                     betas=self.disc_hparams.betas,
                                     eps=self.disc_hparams.eps,
                                     weight_decay=self.disc_hparams.weight_decay)
        self.d_scheduler = NoamScheduler(self.d_optimizer,
                                       n_warmup=self.disc_hparams.n_warmup,
                                       init_scale=self.disc_hparams.init_scale)




    def _create_hparams(self, hparams):
        # Base Hparams
        super()._create_hparams(hparams)
        # Optim HParams
        self.disc_hparams = hparams.discriminator

    def fit(self, checkpoint=None, loggers=None):
        # random seed
        self.setup_seed(self.train_hparams.seed)
        # cudnn settings
        if self.device.type == 'cuda':
            torch.backends.cudnn.enabled = self.train_hparams.cudnn_enabled
            torch.backends.cudnn.benchmark = self.train_hparams.cudnn_benchmark
            print('cuDNN Enabled:', self.train_hparams.cudnn_enabled)
            print('cuDNN Benchmark:', self.train_hparams.cudnn_benchmark)

        self.loggers = loggers or Logger(self.train_hparams)

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

        # save initial model weights
        self.save_checkpoint(remove=False)
        base_step = self.step

        collate_fn = SpeechCollate(self.data_hparams, device=torch.device('cpu'))

        train_loader = self.dataloader(self.trainset, batch_size=self.data_hparams.batch_size, collate_fn=collate_fn)
        valid_loader = self.dataloader(self.validset, batch_size=self.data_hparams.batch_size, collate_fn=collate_fn)

        for e in range(self.epoch + 1, self.epoch + 1 + self.train_hparams.epochs):
            self.epoch = e

            train_losses = self._train_epoch(train_loader, base_step)
            valid_losses = self._validate(valid_loader)

            # if self.epoch % self.train_hparams.epochs_per_chkpt == 0:
            #     # checkpoint at every n epoch
            #     self.save_checkpoint(remove=False)



            self.loggers.log_epoch('train', 'epoch', self.epoch, train_losses)
            self.loggers.log_epoch('valid', 'epoch', self.epoch, valid_losses)

            print(f'Epoch {e} ')
            print(f"[ Train ] {', '.join([f'{k}: {v:.4f}' for k, v in train_losses.items()])}")
            print(f"[ Valid ] {', '.join([f'{k}: {v:.4f}' for k, v in valid_losses.items()])}")

        # save final checkpoint

        self.save_checkpoint(remove=False)

    LOSSES = ['mse', 'mae', 'ssim', 'drn', 'cos', 'spd', 'f0', 'nrg', 'gan', 'hdn', 'disc', 'attention']  # the loss keys

    def _train_epoch(self, dataloader, base_step):
        self.model.train()

        running_losses = {key: 0.0 for key in self.LOSSES}

        pbar = tqdm(dataloader, unit="audios", unit_scale=dataloader.batch_size, \
                    disable=self.train_hparams.disable_progress_bar)
        for it, batch in enumerate(pbar, start=1):
            self.optimizer.zero_grad()
            self.d_optimizer.zero_grad()

            inputs, targets, lengths = self.model.parse_batch(batch, device=self.device)
            outputs = self.model(inputs)
            outputs_and_targets = (outputs, targets, lengths)

            losses = self.criterion(outputs_and_targets, use_dtw=False)

            # Discriminator Fake
            if self.step >= self.disc_hparams.n_train_start and lengths[1].min >= max(self.disc_hparams.time_lengths):
                d_fake, start_frames_wins, h_fake = self.d_model(outputs[0], lengths[1])
                gan_loss = self.d_criterion(d_fake, torch.tensor(1.0).expand_as(d_fake).to(d_fake.device))
                losses['gan'] = gan_loss * self.disc_hparams.gan_loss_lambda
                #feature mapping loss
                if self.disc_hparams.enable_hdn_loss:
                    *__,h_real = self.d_model(targets[0], lengths[1],start_frames_wins=start_frames_wins)
                    h_fake = [h_in for h_out in h_fake for h_in in h_out]
                    h_real = [h_in for h_out in h_real for h_in in h_out]
                    hdn_loss = self.d_criterion(torch.stack(h_fake, dim=-1), torch.satack(h_real, dim=-1))
                    losses['hdn'] = hdn_loss * self.disc_hparams.gan_loss_lambda
            else:
                losses['gan'] = torch.tensor(0.0)
                if self.disc_hparams.enable_hdn_loss:
                    losses['hdn'] = torch.tensor(0.0)

            loss = sum(losses.values()) / self.optim_hparams.grad_acc_step
            loss.backward()

            if self.step % self.optim_hparams.grad_acc_step == 0:
                # clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(self.model.parameters(), self.optim_hparams.grad_clip_thresh)
                self.optimizer.step()
                self.scheduler.step()


            # Discriminator Process
            if self.step >= self.disc_hparams.n_train_start and lengths[1].min >= max(self.disc_hparams.time_lengths):
                d_real, *__ = self.d_model(targets[0],lengths[1])
                d_fake, *__ = self.d_model(outputs[0].detach(),lengths[1])
                d_real_loss = self.d_criterion(d_real, torch.tensor(1.0).expand_as(d_real).to(d_real.device))
                d_fake_loss = self.d_criterion(d_fake, torch.tensor(0.0).expand_as(d_fake).to(d_fake.device))
                disc_loss = d_real_loss + d_fake_loss
                losses['disc'] = disc_loss * self.disc_hparams.gan_loss_lambda

                disc_loss = disc_loss / self.disc_hparams.grad_acc_step
                disc_loss.backward()
                if self.step % self.disc_hparams.grad_acc_step == 0:
                    # clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(self.d_model.parameters(), self.disc_hparams.grad_clip_thresh)
                    self.d_optimizer.step()
                    self.d_scheduler.step()
            else:
                losses['disc'] = torch.tensor(0.0)


            losses = {key: val.item() for key, val in losses.items()}
            running_losses = {key: running_losses[key] + val for key, val in losses.items()}
            epoch_losses = {key: val / it for key, val in running_losses.items()}

            # update the progress bar
            pbar.set_postfix({
                key: f'{val:.5f}' for key, val in epoch_losses.items()
            })

            # logging
            learning_rate = self._get_learning_rate(self.optimizer)
            d_learning_rate = self._get_learning_rate(self.d_optimizer)
            lr_dict = {'lr': learning_rate, 'd_lr': d_learning_rate}


            self.loggers.log_train('train', 'step', self.step, losses, lr_dict=lr_dict)

            if (self.step - base_step) % self.train_hparams.step_interval == 0:
                self.save_checkpoint(remove=False)

            if (self.step - base_step) % self.train_hparams.target_end_step:
                os.symlink(self.checkpoint, os.path.join(self.train_hparams.out_path,f'step_{self.train_hparams.target_end_step}'))

            self.step += 1

        return epoch_losses


    def _validate(self, dataloader):
        self.model.eval()

        running_losses = {key: 0.0 for key in self.LOSSES}

        pbar = tqdm(dataloader, unit="audios", unit_scale=dataloader.batch_size, \
                    disable=self.train_hparams.disable_progress_bar)
        for it, batch in enumerate(pbar, start=1):
            inputs, targets, lengths = self.model.parse_batch(batch, device=self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
            outputs_and_targets = (outputs, targets, lengths)

            losses = self.criterion(outputs_and_targets, use_dtw=False)

            # Discriminator Fake
            if self.step >= self.disc_hparams.n_train_start and lengths[1].min >= max(self.disc_hparams.time_lengths):
                with torch.no_grad():
                    d_fake, start_frames_wins, h_fake = self.d_model(outputs[0], lengths[1])
                gan_loss = self.d_criterion(d_fake, torch.tensor(1.0).expand_as(d_fake).to(d_fake.device))
                losses['gan'] = gan_loss * self.disc_hparams.gan_loss_lambda
                # feature mapping loss
                if self.disc_hparams.enable_hdn_loss:
                    with torch.no_grad():
                        *__, h_real = self.d_model(targets[0], lengths[1], start_frames_wins=start_frames_wins)
                    h_fake = [h_in for h_out in h_fake for h_in in h_out]
                    h_real = [h_in for h_out in h_real for h_in in h_out]
                    hdn_loss = self.d_criterion(torch.stack(h_fake, dim=-1), torch.satack(h_real, dim=-1))
                    losses['hdn'] = hdn_loss * self.disc_hparams.gan_loss_lambda
            else:
                losses['gan'] = torch.tensor(0.0)
                if self.disc_hparams.enable_hdn_loss:
                    losses['hdn'] = torch.tensor(0.0)
            # Discriminator Process
            if self.step >= self.disc_hparams.n_train_start and lengths[1].min >= max(
                    self.disc_hparams.time_lengths):
                with torch.no_grad():
                    d_real, *__ = self.d_model(targets[0], lengths[1])
                    d_fake, *__ = self.d_model(outputs[0].detach(), lengths[1])
                d_real_loss = self.d_criterion(d_real, torch.tensor(1.0).expand_as(d_real).to(d_real.device))
                d_fake_loss = self.d_criterion(d_fake, torch.tensor(0.0).expand_as(d_fake).to(d_fake.device))
                disc_loss = d_real_loss + d_fake_loss
                losses['disc'] = disc_loss * self.disc_hparams.gan_loss_lambda
            else:
                losses['disc'] = torch.tensor(0.0)

            losses = {key: val.item() for key, val in losses.items()}
            running_losses = {key: running_losses[key] + val for key, val in losses.items()}
            epoch_losses = {key: val / it for key, val in running_losses.items()}

            # plot spectrogram
            spec_outputs, specs = outputs[0], targets[0]
            spec_outputs, specs = spec_outputs.cpu().detach(), specs.cpu().detach()
            input_lengths, output_lengths = lengths
            predicted_lengths = outputs[1]
            index = random.randint(0, specs.size(0) - 1)  # -1
            output_length, input_length = output_lengths[index].item(), input_lengths[index].item()
            predicted_length = predicted_lengths[index].item()
            specs_fig, _ = plot_spectrogram(spec_outputs[index, :predicted_length, :],
                                            target=specs[index, :output_length, :])

            self.loggers.log_train('valid', 'step', self.step, losses, image_dict={'spec': specs_fig})

        return epoch_losses

    def save_d_checkpoint(self, checkpoint):
        self.d_checkpoint = checkpoint
        print("Saving the disc checkpoint file '%s'..." % self.d_checkpoint)
        torch.save(
            {
                # 'epoch': self.epoch,
                # 'step': self.step,
                'model': self.d_model.state_dict(),
                'optimizer': self.d_optimizer.state_dict(),
                'scheduler': self.d_scheduler.state_dict()
            },
            self.d_checkpoint)

    def load_d_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint, map_location=self.device)
        # self.epoch = checkpoint['epoch']
        # self.step = checkpoint['step']
        self.d_model.load_state_dict(checkpoint['model'])
        self.d_optimizer.load_state_dict(checkpoint['optimizer'])
        self.d_scheduler.load_state_dict(checkpoint['scheduler'])
        print("Loaded checkpoint epoch=%d step=%d" % (self.epoch, self.step))

        self.d_checkpoint = None  # prevent overriding old checkpoint

    # overwrite load_checkpoint to remove prev one and create soft-link
    def save_checkpoint(self, remove=True):
        checkpoint = super().save_checkpoint(remove=remove)
        # disc checkpoint
        if remove and self.d_checkpoint is not None:
            os.remove(self.d_checkpoint)
        dirname, filename = os.path.split(checkpoint)
        d_chekpoint = os.path.join(dirname, f'd_{filename}')

        self.save_d_checkpoint(d_chekpoint)

        # create soft-link
        latest_checkpoint = os.path.join(self.train_hparams.out_path, 'd_as_latest')
        os.system("ln -fs {} {}".format(self.d_checkpoint, latest_checkpoint))

    # overwrite load_checkpoint to avoid loading error
    def load_checkpoint(self, checkpoint):
        try:
            dirname, filename = os.path.split(checkpoint)
            d_chekpoint = os.path.join(dirname, f'd_{filename}')
            self.load_d_checkpoint(d_chekpoint)
        except Exception as err:
            print(err)
            print("Loaded disc checkpoint '{}' failed !".format(checkpoint))

