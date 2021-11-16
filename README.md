# Bone Age

Now everything is integrated into [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/) which is basically a wrapper for pytorch code and fully compatible. 
Mainly, we are using the lightning Trainer instead of our own loop, so all the parameters for training (e.g. gpus, checkpoints files, etc.) are specified by the Trainer.

## Command line
On the inside the trainer is called and the trainer arguments are parsed from the commandline. 
Hence, all args can be directly specified on the commandline upon calling the `train_model.py`. 

Some useful args are
  * `precision` (e.g. 16 or 'bf16')
  * `ckpt_path` path to checkpoint to resume training from
  * `max_steps` maximum steps
  * `max_time` time after which the training ends (e.g. `00:01:00:00` for 1h )
  * `overfit_batches` option for reducing number of used batches for developing and debugging
  * `stochastic_wheight_avg`
  * `auto_select_gpus` use all gpus available, see [here](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#pytorch_lightning.trainer.Trainer.params.auto_select_gpus)
  * `resume_from_checkpoint` continue from existing checkpoint (Note: this creates a new run, so it will create a new checkpoint and tensorboard logs)

A whole list is available [here](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-class-api).

## Logging
As pre-defined, the run will create an output `run.log` in the main dir. 
For real runs the log should be moved to `bone_age/output/{run_name}/{version_number}/run.log`. 
The workaround to archive this is creating the according folder structure **before** running the script and setting an environment variable called `LOG_DIR`:

Hence, actual runs should be called like: 
``` bash
LOG_FILE="output/{run_name}/{version_number}_run.log" python train_model.py --model dbam_inceptionv3 --name {run_name} --gpus 1 ...
```
Make sure that the `--name` parameter matches the name in the `LOG_FILE` param.

## Code formatting
Code is formatted using the [black](https://black.readthedocs.io/en/stable/) formatter.