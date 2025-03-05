import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Optional, Tuple


class TorchTrainer:
    def __init__(
            self,
            model_name: str,
            learning_rate: float,
            loss: Tuple[str, torch.nn.Module],
            metric: Tuple[str, torch.nn.Module],
            device: Optional[torch.device],
            epochs: int,
            lr_decay_patience: Optional[int],
            lr_decay_factor: float,
            early_stopping_patience: Optional[int],
            verbose: bool,
            resume_training: bool,
            checkpoint_directory: str,
            tensorboard_directory: str
    ):
        self._model_name = model_name
        self._learning_rate = learning_rate
        self._loss_name, self._loss_fn = loss
        self._metric_name, self._metric_fn = metric
        self._device = device
        self._epochs = epochs
        self._lr_decay_patience = lr_decay_patience
        self._lr_decay_factor = lr_decay_factor
        self._early_stopping_patience = early_stopping_patience
        self._verbose = verbose
        self._resume_training = resume_training
        self._checkpoint_directory = f'{checkpoint_directory}/{model_name}'
        self._tensorboard_directory = f'{tensorboard_directory}/{model_name}'

        self._best_model_filename = 'best_model.pt'
        self._last_model_filename = 'last_model.pt'
        self._use_device = device is not None

        os.makedirs(name=self._checkpoint_directory, exist_ok=True)
        os.makedirs(name=self._tensorboard_directory, exist_ok=True)

    def _reset_training_state(
            self,
            model: torch.nn.Module
    ) -> Tuple[
        torch.optim.Optimizer,
        Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
        int,
        int,
        float
    ]:
        if self._use_device is not None:
            model = model.to(self._device)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=self._learning_rate)
        lr_scheduler = None if self._lr_decay_patience is None else torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=self._lr_decay_factor,
            patience=self._lr_decay_patience
        )

        if self._resume_training and os.path.exists(path=f'{self._checkpoint_directory}/{self._last_model_filename}'):
            checkpoint = torch.load(f'{self._checkpoint_directory}/{self._last_model_filename}')
            current_epoch = checkpoint['epoch']
            best_epoch = checkpoint['best_epoch']
            min_loss = checkpoint[f'test_loss_{self._loss_name}']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if lr_scheduler is not None:
                lr_scheduler_state_dict = checkpoint['lr_scheduler_state_dict']

                if lr_scheduler_state_dict is not None:
                    lr_scheduler.load_state_dict(lr_scheduler_state_dict)
        else:
            current_epoch = 0
            best_epoch = 0
            min_loss = np.inf

        return optimizer, lr_scheduler, current_epoch, best_epoch, min_loss

    def fit(self, model: torch.nn.Module, train_dataloader: DataLoader, eval_dataloader: DataLoader):
        optimizer, lr_scheduler, current_epoch, best_epoch, min_loss = self._reset_training_state(model=model)

        num_train_batches = len(train_dataloader)
        num_eval_batches = len(eval_dataloader)
        writer = SummaryWriter(log_dir=self._tensorboard_directory)

        for epoch in tqdm(range(current_epoch + 1, self._epochs + 1), desc='Epoch'):
            # --- Training Phase ---

            model.train()
            total_train_loss = 0.0
            for (inputs, targets) in tqdm(train_dataloader, desc='Train Batch'):
                optimizer.zero_grad()

                if self._use_device is not None:
                    targets = targets.to(self._device)
                    for key, inp in inputs.items():
                        inputs[key] = inp.to(self._device)

                y_pred = model.forward(inputs)
                loss = self._loss_fn(y_pred, targets)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            total_train_loss /= num_train_batches
            writer.add_scalar(f'Loss-{self._loss_name}/train', total_train_loss, epoch)

            # --- Evaluation Phase ---

            model.eval()
            total_eval_loss = 0.0
            total_eval_metric = 0.0
            with torch.no_grad():
                for (inputs, targets) in tqdm(eval_dataloader, desc='Eval Batch'):
                    if self._use_device is not None:
                        targets = targets.to(self._device)
                        for key, inp in inputs.items():
                            inputs[key] = inp.to(self._device)

                    outputs = model.forward(inputs)
                    total_eval_loss += self._loss_fn.forward(outputs, targets).item()
                    total_eval_metric += self._metric_fn.forward(outputs, targets).item()
            total_eval_loss /= num_eval_batches
            total_eval_metric /= num_eval_batches

            if lr_scheduler is not None:
                lr_scheduler.step(metrics=total_eval_loss)

            writer.add_scalar(f'Loss-{self._loss_name}/eval', total_eval_loss, epoch)
            writer.add_scalar(f'Metric-{self._loss_name}/eval', total_eval_loss, epoch)
            checkpoint = {
                'epoch': epoch,
                'best_epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': None if lr_scheduler is None else lr_scheduler.state_dict(),
                f'test_loss_{self._loss_name}': total_eval_loss,
            }

            if total_eval_loss < min_loss:
                if self._verbose:
                    logging.info(f'\nNew best eval loss {total_eval_loss} at epoch: {epoch}\n')

                min_loss = total_eval_loss
                best_epoch = epoch
                checkpoint['best_epoch'] = best_epoch
                torch.save(checkpoint, f'{self._checkpoint_directory}/{self._best_model_filename}')

            torch.save(checkpoint, f'{self._checkpoint_directory}/{self._last_model_filename}')

            if epoch - best_epoch > self._early_stopping_patience:
                break

            if self._verbose:
                logging.info(f'\nEpoch: {epoch}\tTrain Loss: {total_train_loss}\tEval Loss: {total_eval_loss}\t Eval Metric: {total_eval_metric}\n')

        writer.flush()
        writer.close()
