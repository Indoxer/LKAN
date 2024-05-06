import torch
from omegaconf import OmegaConf

from lkan.datamodule import TestDataModule
from lkan.loggers import CustomLogger
from lkan.models import KAN
from lkan.trainers import BasicKANTrainer
from tmp.pykan.kan import KAN as KAN_real

# testing the KAN model

# if __name__ == "__main__":
#     layers_dims = [4, 3, 2]

#     real_model = KAN_real(layers_dims, grid=5, k=3, seed=0)
#     model = KAN(layers_dims)

#     for layer, real_layer in zip(model.layers, real_model.act_fun):
#         layer.coeff.data = real_layer.coef.data
#         layer.scale_base.data = real_layer.scale_base.data

#     x_update = torch.rand(3, 4)
#     real_model.update_grid_from_samples(x_update)
#     model.update_grid(x_update)

#     inp = torch.rand(3, 4)
#     out = model.forward(inp)
#     real_out = real_model.forward(inp)
#     print(f"my: {out.detach().numpy()}")
#     print(f"real: {real_out.detach().numpy()}")
#     print(f"max diff: {torch.max(torch.abs(out - real_out))}")


if __name__ == "__main__":

    model = KAN(layers_dims=[2, 5, 1])

    name = "basickan"
    version = "0.1"
    save_dir = f"./.experiments/{name}/{version}"

    datamodule = TestDataModule(batch_size=16, split_ratio=0.8)
    datamodule.setup()
    logger = CustomLogger(
        save_dir=save_dir, name=name, version=version, cfg=OmegaConf.create({})
    )

    trainer = BasicKANTrainer(
        model=model,
        lr=0.01,
        update_grid=False,
        grid_update_freq=63,
        stop_grid_update_step=10000,
        logger=logger,
        lr_scheduler=None,
        lr_scheduler_params={},
        lr_step=None,
        clip_grad_norm=0.5,
        accumulate_grad_batches=1,
        device="cuda",
    )

    trainer.fit(
        max_epochs=10,
        max_steps=1000,
        validation_every_n_steps=10,
        save_every_n_steps=10,
        datamodule=datamodule,
    )
