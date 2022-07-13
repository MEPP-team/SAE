import torch

from train import load_trainer

if __name__ == "__main__":
    with torch.no_grad():
        job_id = "SAE-LP-4096-16"

        trainer, opt = load_trainer(job_id, profiler=None)

        trainer.test(opt["model"], opt['dataloader_test'])
