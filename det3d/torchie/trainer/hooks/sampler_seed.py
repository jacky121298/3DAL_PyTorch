from .hook import Hook


class DistSamplerSeedHook(Hook):
    def before_epoch(self, trainer):
        trainer.data_loader.sampler.set_epoch(trainer.epoch)