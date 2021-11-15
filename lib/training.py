"""
utility and convenience functions for pytorch
"""

import os
import re
import time
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm

OPTIMIZER_STATE_KEY = "optimizer_state"
SCHEDULER_STAT_KEY = "scheduler_state"
EPOCH_KEY = "epochs_trained"
STEPS_KEY = "steps_trained"
TRAINING_LOSS_KEY = "training_loss"
VALID_LOSS_KEY = "validation_loss"

MODEL_SUFFIX = "pth"
CHECKPOINT_SUFFIX = "ckp"


class ModelCheckpoint:
    """
    Convenience class for handling model checkpoint serialization.

    The model and checkpoint are saved as pickled obj and dictionary in the background and can\
    be loaded using 'torch.load(checkpoint_path)' outside this class. However, this class handles\
    key assignment, checkpoint_path handling, etc.
    """

    def __init__(
        self, model_checkpoint_path=None, save_model_mode="best", device_loc="cpu"
    ):
        """
        Create ModelCheckpoint class wrapping training progress and model serialization.

        Training progress and the model itself are saved in separate files in order to/
         spare memory access, as the model does not need to be updated in each training/
          loop.

        :param model_checkpoint_path: path to checkpoint file
        :param save_model_mode: mode of saving model intermediates one of /
        ['best', 'last', 'each']
        :param device_loc: device localization (either 'cpu' or 'cuda:{device number}')
        """
        self.training_progress = (
            model_checkpoint_path + "_training_progress." + CHECKPOINT_SUFFIX
        )
        suffix = "_model_best" if save_model_mode == "best" else "_model"
        self.model_path = model_checkpoint_path + suffix + "." + MODEL_SUFFIX
        self.save_model_mode = save_model_mode
        self.device_loc = device_loc
        self.best_validation_loss = None

    def training_step(
        self,
        model,
        optimizer,
        scheduler,
        epoch,
        step,
        training_loss,
        valid_loss,
        model_path=None,
    ):
        """wrapper to call after each training step to handle serialization based on the settings.

        model_path is ignored if save_model_modes is not set to 'each'.
         Note: the existing file might be overwritten.
        """
        self.serialize_checkpoint(
            optimizer, scheduler, epoch, step, training_loss, valid_loss
        )
        if (
            self.save_model_mode in ["last", "each"]
            or not self.best_validation_loss  # first model
            or self.best_validation_loss < valid_loss  # model improved
        ):
            model_path = model_path if model_path else self.model_path
            self.serialize_model(model, model_path)

    def serialize_all(
        self,
        model,
        optimizer,
        scheduler,
        epoch,
        step,
        training_loss,
        valid_loss,
    ):
        """Wrapper to serialize both the model and the training progress checkpoint"""
        self.serialize_checkpoint(
            optimizer, scheduler, epoch, step, training_loss, valid_loss
        )
        self.serialize_model(model)

    def serialize_checkpoint(
        self,
        optimizer,
        scheduler,
        epoch,
        step,
        training_loss,
        valid_loss,
    ) -> None:
        """serializes training progress checkpoint without saving the model itself"""
        checkpoint = {
            EPOCH_KEY: epoch,
            STEPS_KEY: step,
            OPTIMIZER_STATE_KEY: optimizer.state_dict(),
            SCHEDULER_STAT_KEY: scheduler.state_dict(),
            VALID_LOSS_KEY: valid_loss,
            TRAINING_LOSS_KEY: training_loss,
        }
        checkpoint = remove_none_values_from_dict(checkpoint)
        torch.save(checkpoint, self.training_progress)

    def serialize_model(self, model, model_path=None) -> None:
        """serializes the model"""
        path = self.model_path
        if self.save_model_mode == "each":
            path = (
                model_path
                if model_path
                else re.sub(
                    "_[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}",
                    "_" + datetime.now().strftime("%y-%m-%d_%H-%M-%S"),
                    self.model_path,
                )  # name model after current date and time
            )
        torch.save(model.state_dict(), path)

    def deserialize_all(self, model, optimizer, scheduler):
        """
        load serialized model to model, optimizer, and scheduler and retrieve training
        progress checkpoint

        :return
            model: model with loaded state
            optimizer: optimizer with loaded state
            scheduler: scheduler wth loaded state
            epoch: number of trained epochs
            steps: number of trained steps
            validation_loss: train_loss of saved model (might not be the latest model state)
        """
        epoch = 0
        steps = 0

        if os.path.exists(self.training_progress):
            checkpoint = torch.load(self.training_progress)
            print(
                f"checkpoint restored (last modified "
                + time.strftime(
                    "%Y/%m/%d-%H:%M:%S",
                    time.localtime(os.path.getmtime(self.training_progress)),
                )
                + ")"
            )
            self.deserialize_model(model)
            if OPTIMIZER_STATE_KEY in checkpoint.keys() and optimizer:
                optimizer.load_state_dict(checkpoint[OPTIMIZER_STATE_KEY])
            if SCHEDULER_STAT_KEY in checkpoint.keys() and scheduler:
                scheduler.load_state_dict(checkpoint[SCHEDULER_STAT_KEY])
            if EPOCH_KEY in checkpoint.keys():
                epoch = checkpoint[EPOCH_KEY]
            if STEPS_KEY in checkpoint.keys():
                steps = checkpoint[STEPS_KEY]
            if VALID_LOSS_KEY in checkpoint.keys():
                self.best_validation_loss = checkpoint[VALID_LOSS_KEY]

        return model, optimizer, scheduler, epoch, steps, self.best_validation_loss

    def delete_model_checkpoints(self):
        """deletes existing model checkpoints"""
        if os.path.exists(self.training_progress):
            os.remove(self.training_progress)
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def deserialize_model(self, model):
        self.deserialize_model_from_path(model, self.model_path, self.device_loc)
        
#         model_state = torch.load(self.model_path, de)
#         print(
#             f"model restored (last modified on "
#             + time.strftime(
#                 "%Y/%m/%d-%H:%M:%S",
#                 time.localtime(os.path.getmtime(self.model_path)),
#             )
#             + ")"
#         )
#         if self.device_loc == "cpu":
#             model.load_state_dict(model_state)
#         else:
#             model.load_state_dict(model_state, map_location=self.device_loc)

    @staticmethod
    def deserialize_model_from_path(model, path, device_loc="cpu"):
        model_state = torch.load(path, map_location=device_loc)
        print(
            f"model restored (last modified "
            + time.strftime(
                "%Y/%m/%d-%H:%M:%S",
                time.localtime(os.path.getmtime(path)),
            )
            + ")"
        )
        model.load_state_dict(model_state)
        model.to(torch.device(device_loc))

def remove_none_values_from_dict(d):
    return {k: v for k, v in d.items() if v is not None}


class ModelTrainer:
    """
    Callable to train model wrapping model checkpoint and training tracking via
    tensorboard.

    It is assumed that the samples from the training and validation sets are
    given as an ordered dict including the ground truth label (y).
    """

    def __init__(
        self,
        step_size,
        loss_criterion=nn.MSELoss(reduction="sum"),
        y_key="y",
        checkpoint_path=None,
        save_model_mode="best",
        device_loc="cpu",
        writer=None,
        loss_renorm_factor=1,
        accuracy_metrics={},
        train_from_scratch=False,
    ):
        """
        Init ModelTrainer with model hyper-parameters and metadata settings.

        :param step_size: number of batches  after which to validate the model (should be multiple of batch size)
        :param loss_criterion: train_loss criterion to optimize (must not be reduced)
        :param train_from_scratch: If set to True the existing model will be ignored.
         Note: Existing models might still be overwritten!
        :param checkpoint_path: checkpoint_path to deserialize and serialize the model (if None /
        model will not be saved)
        :param save_model_mode: mode of saving model intermediates one of /
        ['best', 'last', 'each']
        :param writer: reference to initialized 'torch.utils.tensorboard.SummaryWriter'/
        if None training will not be tracked
        :param loss_renorm_factor: factor to reverse renorm of factor (i.e. if /
        y was normalized to sd = 1). Set to 1 to ignore param.
        :param accuracy_metrics: additional model accuracy metrics to evaluate the /
        model. The keys of the dict will be shown as metric in tensorboard.
        """
        self.step_size = step_size
        self.loss_criterion = loss_criterion
        self.y_key = y_key
        self.checkpoint_path = checkpoint_path
        self.writer = writer
        self.loss_renorm_factor = loss_renorm_factor
        self.accuracy_metrics = accuracy_metrics
        self.force_train = train_from_scratch
        self.device_loc = device_loc
        self.checkpoint = ModelCheckpoint(
            model_checkpoint_path=self.checkpoint_path,
            save_model_mode=save_model_mode,
            device_loc=self.device_loc,
        )
        if train_from_scratch:
            self.checkpoint.delete_model_checkpoints()

    def __call__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        n_epochs,
    ) -> None:
        ckp = self.checkpoint.deserialize_all(model, optimizer, scheduler)
        model, optimizer, scheduler, epoch, steps, best_loss = ckp
        
        print("training on " + self.device_loc)
        device = torch.device(self.device_loc)
        model.to(device)
        model.train()

        sum_train_accuracies = {k: 0 for k in self.accuracy_metrics.keys()}
        for epoch in tqdm(range(epoch, n_epochs)):
            sum_loss = n_samples = 0
            sum_train_accuracies = {k: 0 for k in sum_train_accuracies.keys()}
            for i, sample in enumerate(train_loader):
                # forward prop
                x = self.without_y_key(sample)  # input without ground truth
                x = {k: v.to(device) for k, v in x.items()}
                y = sample[self.y_key].to(device)  # retrieve ground truth

                optimizer.zero_grad()
                y_hat = model(*x.values())

                # train_loss calculation
                loss = self.loss_criterion(y_hat, y)
                sum_loss += loss.item()

                # calculate additional acc metrics
                for key, metric in self.accuracy_metrics.items():
                    sum_train_accuracies[key] += metric(y_hat, y).item()
                # get actual batch size for loss computation (batch might be incomplete)
                n_samples += y.shape[0]

                # back prop
                loss.backward()
                optimizer.step()

                if (i + 1) % self.step_size == 0:
                    train_loss = sum_loss / n_samples
                    sum_train_accuracies = {
                        k: v / n_samples for k, v in sum_train_accuracies.items()
                    }

                    # validation_loss, acc_dict = self.calculate_loss()
                    validation_loss, validation_accuracies = self.calculate_loss(
                        model, valid_loader, device
                    )
                    scheduler.step(validation_loss)

                    self.checkpoint.training_step(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        steps,
                        train_loss,
                        validation_loss,
                    )

                    if self.writer:
                        self.report_to_tensorboard(
                            steps,
                            train_loss,
                            validation_loss,
                            sum_train_accuracies,
                            validation_accuracies,
                        )
                    print(f"epoch {epoch} - step {steps}  loss: {train_loss}")
                    steps += 1
                    n_samples = sum_loss = 0
                    sum_train_accuracies = {k: 0 for k in sum_train_accuracies.keys()}

        if self.writer:
            self.writer.flush()
            self.writer.close()

    def report_to_tensorboard(
        self,
        step,
        avg_train_loss,
        avg_validation_loss,
        train_accuracies={},
        validation_accuracies={},
    ) -> None:
        """
        Write training progress to the tensorboard log file.

        :param step: Int. step number (x-axis in tensorboard)
        :param avg_train_loss: Float. Average training loss since last step
        :param avg_validation_loss: Float. Average validation loss since last step
        :param train_accuracies: contains optional additional training accuracy metric
         as {"Metric_name" : training_accuracy}
        :param validation_accuracies: contains optional additional validation accuracy
         metric as {"Metric_name" : validation_accuracy}
        """
        self.writer.add_scalar(
            "training_loss",
            avg_train_loss,
            step,
        )
        if self.loss_renorm_factor != 1:
            self.writer.add_scalar(
                "training_loss_renorm",
                avg_train_loss * self.loss_renorm_factor,
                step,
            )
        self.writer.add_scalar(
            "validation_loss",
            avg_validation_loss,
            step,
        )
        if self.loss_renorm_factor != 1:
            self.writer.add_scalar(
                "validation_loss_renorm",
                avg_validation_loss * self.loss_renorm_factor,
                step,
            )

        def write_metric_dict(d, label):
            for k, v in d.items():
                self.writer.add_scalar(label + "_" + k, v, step)
                if self.loss_renorm_factor != 1:
                    self.writer.add_scalar(
                        label + "_" + k + "_renorm", v * self.loss_renorm_factor, step
                    )

        write_metric_dict(train_accuracies, "training")
        write_metric_dict(validation_accuracies, "validation")

    def calculate_loss(self, model, data_loader, device) -> (float, dict):
        """calculate specified average loss and additional accuracy metrics as dict (if set)"""
        model.eval()
        model.to(device)
        sum_loss = 0
        metric_sum = {k: 0 for k in self.accuracy_metrics.keys()}
        n_samples = 0

        for sample in data_loader:
            with torch.set_grad_enabled(False):  # is needed independent of model.eval()
                y_hat = model(sample["x"].to(device), sample["male"].to(device))
            y = sample[self.y_key].to(device)
            sum_loss += self.loss_criterion(y_hat, y).item()
            if self.accuracy_metrics:
                for key, metric in self.accuracy_metrics.items():
                    metric_sum[key] += metric(y_hat, y).item()
            n_samples += y.shape[0]
        sum_loss = sum_loss / n_samples
        metric_sum = {k: v / n_samples for k, v in metric_sum.items()}

        model.train()
        return sum_loss, metric_sum

    def without_y_key(self, d):
        """return sample dictionary without the y key"""
        return {x: d[x] for x in d if x != self.y_key}
