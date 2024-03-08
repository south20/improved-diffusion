# improved-diffusion
openai的开源项目，添加代码中文注释和学习历程[原地址](https://github.com/openai/improved-diffusion)

对应论文 [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672).

# 使用方法

本节介绍如何训练模型和从模型中采样。

## Installation

克隆此仓库并在终端中导航到它。然后运行：

```
pip install -e .
```

这应该会安装脚本所依赖的improved_diffusion Python包。

## Preparing Data


训练代码从一个包含图像文件的目录中读取图像。在[datasets](datasets)文件夹中，我们提供了为ImageNet、LSUN bedrooms和CIFAR-10准备这些目录的说明/脚本。

要创建自己的数据集，只需将所有图像放入一个扩展名为".jpg"、".jpeg"或".png"的目录中。如果您希望训练一个类条件模型，像这样命名文件："mylabel1_XXX.jpg"、"mylabel2_YYY.jpg"等，以便数据加载器知道"mylabel1"和"mylabel2"是标签。子目录也会被自动枚举，因此可以将图像组织成递归结构（尽管目录名称将被忽略，下划线前缀用作名称）。

数据加载管道会自动对图像进行缩放和中心裁剪。只需将 `--data_dir path/to/images` 传递给训练脚本，其余部分将由它处理。


## Training

要训练模型，您首先需要决定一些超参数。我们将超参数分为三组：模型架构、扩散过程和训练标志。以下是一些基线的合理默认值：

```
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
```

这里是我们实验中的一些变化，以及如何在标志中设置它们：

 * **Learned sigmas:** add `--learn_sigma True` to `MODEL_FLAGS`
 * **Cosine schedule:** change `--noise_schedule linear` to `--noise_schedule cosine`
 * **Importance-sampled VLB:** add `--use_kl True` to `DIFFUSION_FLAGS` and add `--schedule_sampler loss-second-moment` to  `TRAIN_FLAGS`.
 * **Class-conditional:** add `--class_cond True` to `MODEL_FLAGS`.

设置好超参数后，可以像这样运行实验：

```
python scripts/image_train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```
您可能还想以分布式方式训练。在这种情况下，使用`mpiexec`运行相同的命令：


```
mpiexec -n $NUM_GPUS python scripts/image_train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

在分布式训练时，您必须手动将 `--batch_size` 参数除以ranks。如果不进行分布式训练，您可以使用`--microbatch 16`（或在极端内存受限情况下使用`--microbatch 1`）来减少内存使用。

日志和保存的模型将写入由`OPENAI_LOGDIR`环境变量确定的日志目录。如果未设置，则将在`/tmp`中创建一个临时目录。

## Sampling


上述训练脚本将检查点保存为日志目录中的`.pt`文件。这些检查点将具有类似 `ema_0.9999_200000.pt`和 `model200000.pt`的名称。您可能希望从EMA模型中采样，因为它们产生更好的样本。

一旦有了模型的路径，就可以这样生成一大批样本：

```
python scripts/image_sample.py --model_path /path/to/model.pt $MODEL_FLAGS $DIFFUSION_FLAGS
```
同样，这将将结果保存到日志目录中。样本被保存为一个大型`npz`文件，其中文件中的`arr_0`是一大批样本。

就像训练一样，您可以通过MPI运行`image_sample.py`以使用多个GPU和机器。

您可以使用`--timestep_respacing`参数更改采样步数。例如，`--timestep_respacing 250`使用250步进行采样。传递`--timestep_respacing ddim250`类似，但使用来自DDIM论文的均匀步距，而不是我们的步距。

要使用[DDIM](https://arxiv.org/abs/2010.02502)进行采样，请传递`--use_ddim True`。


## 模型和超参数

本节包括论文中主要模型的模型检查点和运行标志。

Note that the batch sizes are specified for single-GPU training, even though most of these runs will not naturally fit on a single GPU. To address this, either set `--microbatch` to a small value (e.g. 4) to train on one GPU, or run with MPI and divide `--batch_size` by the number of GPUs.
请注意，即使大多数运行自然不会适合单个GPU，批量大小也是为单GPU训练指定的。为了解决这个问题，要么将`--microbatch`设置为一个小值（例如4）以在一个GPU上训练，要么使用MPI运行并将`--batch_size`除以GPU数量。

Unconditional ImageNet-64 with our `L_hybrid` objective and cosine noise schedule [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt)]:

```bash
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
```

Unconditional CIFAR-10 with our `L_hybrid` objective and cosine noise schedule [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt)]:

```bash
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
```

Class-conditional ImageNet-64 model (270M parameters, trained for 250K iterations) [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_cond_270M_250K.pt)]:

```bash
MODEL_FLAGS="--image_size 64 --num_channels 192 --num_res_blocks 3 --learn_sigma True --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 3e-4 --batch_size 2048"
```

Upsampling 256x256 model (280M parameters, trained for 500K iterations) [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/upsample_cond_500K.pt)]:

```bash
MODEL_FLAGS="--num_channels 192 --num_res_blocks 2 --learn_sigma True --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 3e-4 --batch_size 256"
```

LSUN bedroom model (lr=1e-4) [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/lsun_uncond_100M_1200K_bs128.pt)]:

```bash
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
```

LSUN bedroom model (lr=2e-5) [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/lsun_uncond_100M_2400K_bs64.pt)]:

```bash
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm False"
TRAIN_FLAGS="--lr 2e-5 --batch_size 128"
```

Unconditional ImageNet-64 with the `L_vlb` objective and cosine noise schedule [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_vlb_100M_1500K.pt)]:

```bash
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"
```

Unconditional CIFAR-10 with the `L_vlb` objective and cosine noise schedule [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_vlb_50M_500K.pt)]:

```bash
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"
```
