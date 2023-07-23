"""
Reference:
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/3.%20WGAN/train.py
"""
import random
import sys

sys.path.append('..')

import os
from functools import partial
import argparse
import yaml
# import uuid
# from datetime import datetime

import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.backends import cudnn

from ignite.engine import Events, Engine
import ignite.distributed as idist
import ignite.metrics
from ignite.contrib.handlers import ProgressBar

from kornia.geometry.transform import resize
from pytorch_fid.inception import InceptionV3
from torchsummary import summary

from domain_translator.models import models_wgan, models_sngan, models_encoder, models_wgan_large
import losses
import loaders
import utils
from transforms import Transformer


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the configuration file')

    # Datasets
    parser.add_argument('--dataset_tar', type=str, help="target dataset",
                        choices=['cifar10', 'cifar100', 'mnist', 'svhn', 'stl10', 'tinyimagenet',
                                 'objectron', 'imagenet', 'celeba', 'caltech256', 'illumination'])
    parser.add_argument('--dataset_src', type=str, help="source dataset",
                        choices=['cifar10', 'cifar100', 'mnist', 'svhn', 'stl10', 'tinyimagenet',
                                 'objectron', 'imagenet', 'celeba', 'caltech256', 'illumination'])
    parser.add_argument('--data_dir_tar', type=str, default='../work_dirs/datasets')
    parser.add_argument('--data_dir_src', type=str, default='../work_dirs/datasets')
    parser.add_argument('--concat_dataset_mode', type=str,
                        choices=['same_bs_trunc_head', 'same_bs_rand_trunc', 'same_bs_iter_all'],
                        default='same_bs_rand_trunc',
                        help="methods to align datasets when source dataset has more images "
                             "than the target. assume that target dataset has n images"
                             "same_bs_trunc_head: only use the first n images in source"
                             "same_bs_rand_trunc: use all souce images in a random pick manner"
                             "same_bs_iter_all: use all souce images in an ergodic manner")
    parser.add_argument('--image_size', type=int, default=32,
                        help="input images are resized to this size")
    parser.add_argument('--aug_src', action='store_true', default=False,
                        help="whether to augment the source dataset")
    parser.add_argument('--aug_tar', action='store_true', default=False,
                        help="whether to augment the target dataset")
    parser.add_argument('--src_size', type=int, default=0,
                        help="use how many source data, default 0 is to use all the data")

    # Data transformation
    parser.add_argument('--transform', type=str, required=True,
                        choices=['randaugment', 'rotate', '3d', 'illumination'],
                        help="data transformation to which we learn robustness")
    parser.add_argument('--rand_num_test', type=int, default=20,
                        help="number of random transformations per image to estimate the "
                             "expectation during testing")
    parser.add_argument('--rand_num_train', type=int, default=1,
                        help="number of random transformations per image to estimate the "
                             "expectation during training")

    # Model
    parser.add_argument('--framework', type=str,
                        choices=['wgan', 'wgangp', 'sngan', 'wgan_large', 'wgangp_large'],
                        default='wgan')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help="number of residual blocks used in generator")

    parser.add_argument('--n_disc_steps', type=int, default=5,
                        help="number of training steps for discriminator per iter")
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='enable to evaluate an existing model w/o training')
    parser.add_argument('--model_dir', type=str,
                        help='path to the trained model file for evaluation')

    # Settings for equivariance
    parser.add_argument("--w_eq", type=float, default=1.0,
                        help="weight of equivariance regularization")
    parser.add_argument('--eq_method', type=str, choices=['contrastive', 'groundtruth'],
                        default='contrastive', help="method for encouraging equivariance")
    parser.add_argument('--loss_eq', type=str, default='mse', choices=['mse', 'l1'])
    parser.add_argument('--eq_contrastive_encoder', type=str, default='default')

    # Settings for WGAN
    parser.add_argument('--clip_value', type=float, default=0.03,
                        help="lower and upper clip value for discriminator's weights."
                             "lower value seemingly improves equivariance but hurts FID")
    # Settings for WGAN-GP
    parser.add_argument('--lambda_gp', type=int, default=10,
                        help="weight of gradient penalty")

    # Basic training parameters
    parser.add_argument('--max_epochs', type=int, default=200, help="number of epochs of training")
    parser.add_argument('--batch_size', type=int, default=256, help="size of the batches")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--distributed', action='store_true', default=False)

    # Optimization
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="learning rate for both generator and discriminator")
    parser.add_argument("--lr_encoder", type=float, default=0.0002,
                        help="learning rate for both generator and discriminator")
    parser.add_argument('--decay_epoch', type=int, default=100,
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument("--b1", type=float, default=0.5,  # 0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,  # 0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument('--opt-type', choices=['RMSprop', 'Adam'], default='Adam',
                        help="type of optimizer")
    parser.add_argument('--decay_lr', action='store_true', default=False,
                        help='enable the lr scheduler')

    # Logs
    parser.add_argument('--log_dir', type=str, default='../work_dirs/domain_translator/logs')
    parser.add_argument('--log_id', type=str, default=None,
                        help="log folder name")
    parser.add_argument('--resume', type=int, default=None,
                        help="path to latest checkpoint (default: None)")
    parser.add_argument('--log_name', type=str, default='temp')
    parser.add_argument('--ckpt_freq', type=int, default=10,
                        help="epoch frequency for saving trained models")
    parser.add_argument('--print_freq', type=int, default=100,
                        help="iteration frequency for log printing")  # iters
    parser.add_argument('--plot_freq', type=int, default=10,
                        help="epoch frequency for saving example figures")  # epoch

    args = parser.parse_args()

    # If config file is provided, overwrite args with it
    if args.config is not None:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        for action in parser._actions:
            if action.dest in config:
                action.default = config[action.dest]
    # parse again to overwrite configs with manually passed argument
    args = parser.parse_args()

    # setup log folder name.
    # e.g., work_dirs/domain_translator/logs/cifar_svhn_randaugment_wgan_eq0.1_'log_id'_0658_af42
    # unique_name = f'{datetime.now().strftime("%y%m%d_%H%M")}_' \
    #               f'{str(uuid.uuid4())[:4]}'
    # if args.log_id is not None:
    #     unique_name = args.log_id + '_' + unique_name
    # log_folder_name = '_'.join([args.dataset_tar,
    #                          args.dataset_src,
    #                          args.transform,
    #                          args.framework,
    #                          f'eq{str(args.w_eq)}',
    #                          unique_name])
    log_folder_name = args.log_id
    args.log_dir = os.path.join(args.log_dir, log_folder_name)

    print(f"Working dir is {os.getcwd()}, log dir is {args.log_dir}")
    print('Args:', args)

    return args


def equivariance_loss_oracle(u, xi_u, transformer, netG, rand_num):
    loss_eq = 0.
    if args.transform not in ('3d', 'illumination'):  # does not evaluate 3d due to lack of ground-truth
        t_us, t_xi_us = [], []
        for i in range(rand_num):
            with torch.no_grad():
                t_xi_u, t_u = transformer.apply(xi_u, u, denorm_1='tar', denorm_2='src')
                t_xi_us.append(t_xi_u)
                t_us.append(t_u)

        t_u = torch.cat(t_us)
        xi_t_u = transformer.normalize_tar(netG(t_u))
        t_xi_u = torch.cat(t_xi_us).detach()

        if args.loss_eq == 'mse':
            loss_eq = F.mse_loss(xi_t_u, t_xi_u)
        elif args.loss_eq == 'l1':
            loss_eq = F.l1_loss(xi_t_u, t_xi_u)
        else:
            raise Exception(f'Unknown loss_eq {args.loss_eq}')

    return loss_eq


def equivariance_loss_contrastive(u, u_raw, transformer, netG, netTE, rand_num, epoch):
    t_us, us = [], []
    for i in range(rand_num):
        with torch.no_grad():
            t_us.append(transformer.apply(u_raw, denorm_1='src'))
            us.append(u)

    t_u = torch.cat(t_us)  # to device since 3d transformed data are still in cpu
    u = torch.cat(us)

    xi_u, xi_t_u = torch.tensor_split(
        transformer.normalize_tar(netG(torch.cat([u, t_u]))), 2)

    param_u_and_t_u, param_xi_u_and_xi_t_u = torch.tensor_split(
        netTE(torch.cat([u, xi_u]), torch.cat([t_u, xi_t_u])), 2)

    p_param_u_and_t_u, p_param_xi_u_and_xi_t_u = torch.tensor_split(
        netTE.predictor(torch.cat([param_u_and_t_u, param_xi_u_and_xi_t_u])), 2)

    loss_eq = F.cosine_similarity(p_param_xi_u_and_xi_t_u,
                                  param_u_and_t_u.detach(), dim=-1).mean().mul(-1)
    loss_eq += F.cosine_similarity(p_param_u_and_t_u,
                                   param_xi_u_and_xi_t_u.detach(), dim=-1).mean().mul(-1)
    loss_eq *= 0.5

    # trick to ensure coherent padding and filling
    if epoch < 10:  # soft start
        loss_eq += 1 * F.mse_loss(u - t_u, xi_u - xi_t_u)
    return loss_eq


def train(args, transformer, device):
    build_model = partial(idist.auto_model, sync_bn=True)

    if args.framework == 'sngan':
        netG = build_model(models_wgan.create_generator(image_size=args.image_size,
                                                        num_residual_blocks=args.num_res_blocks))
        netD = build_model(models_sngan.ResDiscriminator64())
        loss_fn = losses.Hinge()

    elif args.framework in ('wgan', 'wgangp'):
        netG = build_model(models_wgan.create_generator(image_size=args.image_size,
                                                        num_residual_blocks=args.num_res_blocks))
        netD = build_model(models_wgan.create_discriminator(image_size=args.image_size))
        loss_fn = losses.Wasserstein()

    elif args.framework in ('wgan_large', 'wgangp_large'):
        netG = build_model(models_wgan_large.create_generator(args.num_res_blocks))
        netD = build_model(
            models_wgan_large.Discriminator(input_shape=(3, args.image_size, args.image_size)))
        loss_fn = losses.Wasserstein()

    else:
        raise Exception(f'Unknown framework {args.framework}')

    netTE = build_model(models_encoder.Encoders[args.eq_contrastive_encoder])

    summary(netG, (3, args.image_size, args.image_size))
    summary(netD, (3, args.image_size, args.image_size))

    optimizer_G = idist.auto_optim(optim.Adam(netG.parameters(),
                                              lr=args.lr,
                                              betas=(args.b1, args.b2)))
    optimizer_D = idist.auto_optim(optim.Adam(netD.parameters(),
                                              lr=args.lr,
                                              betas=(args.b1, args.b2)))
    optimizer_TE = idist.auto_optim(optim.Adam(netTE.parameters(),
                                               lr=args.lr_encoder,
                                               betas=(args.b1, args.b2)))
    # does not decay optimizer_TE, following SimSiam
    schedulers_g = optim.lr_scheduler.LambdaLR(optimizer_G,
                                               lr_lambda=utils.LambdaLR(args.max_epochs, 0,
                                                                        args.decay_epoch).step)
    schedulers_d = optim.lr_scheduler.LambdaLR(optimizer_D,
                                               lr_lambda=utils.LambdaLR(args.max_epochs, 0,
                                                                        args.decay_epoch).step)
    schedulers = [schedulers_g, schedulers_d]

    def training_step(engine, batch):
        netG.train()
        netD.train()
        netTE.train()

        # train discriminator
        if isinstance(batch[0][0], torch.Tensor):  # handle args.n_disc_steps == 1
            batch = [batch]

        for (x, _), (u_raw, _) in batch:
            x = x.to(device)
            if isinstance(u_raw, dict):  # handle 3d dataset
                u = u_raw['anchor_view'].to(device)
            else:  # not 3d dataset, still keep u_raw since it later will be used in transform to keep compatibility with 3d transform
                u, u_raw = u_raw.to(device), u_raw.to(device)

            with torch.no_grad():
                xi_u = transformer.normalize_tar(netG(u).detach())

            yd_real = netD(x)
            yd_fake = netD(xi_u)
            loss_d = loss_fn(yd_real, yd_fake)

            if args.framework in ('wgangp', 'wgangp_large'):
                loss_gp = cacl_gradient_penalty(netD, x, xi_u.detach())
                loss_d += args.lambda_gp * loss_gp

            optimizer_D.zero_grad()
            loss_d.backward()
            optimizer_D.step()

            # clip critic weights between -0.01, 0.01
            if args.framework in ('wgan', 'wgan_large'):
                for p in netD.parameters():
                    p.data.clamp_(-args.clip_value, args.clip_value)

            acc_d = (yd_real.flatten(1).mean(1) > yd_fake.flatten(1).mean(1)
                     ).sum() / yd_real.shape[0]

        # train generator - encourage target-like translation
        xi_u = transformer.normalize_tar(netG(u))
        yd_fake = netD(xi_u)
        loss_gan = loss_fn(yd_fake)

        # train generator - encourage equivariance
        loss_eq = 0.
        if args.w_eq > 0 and args.eq_method == 'groundtruth':
            loss_eq = equivariance_loss_oracle(u, xi_u, transformer, netG, args.rand_num_train)

        elif args.w_eq > 0 and args.eq_method == 'contrastive':
            loss_eq = equivariance_loss_contrastive(u, u_raw, transformer, netG, netTE,
                                                    args.rand_num_train, engine.state.epoch)
        loss_g = loss_gan + args.w_eq * loss_eq

        # since we are using Adam w/o weight_decay,
        # args.w_eq doesn't affect the optimization of optimizer_TE
        optimizer_G.zero_grad()
        optimizer_TE.zero_grad()
        loss_g.backward()
        optimizer_G.step()
        optimizer_TE.step()

        outputs = dict(loss_d=loss_d, loss_gan=loss_gan, loss_eq=loss_eq, acc_d=acc_d)
        outputs = {k: v for k, v in outputs.items() if v is not None}

        return outputs

    return dict(netG=netG,
                netD=netD,
                optimizer_G=optimizer_G,
                optimizer_D=optimizer_D,
                schedulers=schedulers,
                trainer=Engine(training_step))


def create_evaluator(netG, transformer):
    """
    Compute FID score (of xi_u and xi_ut) and equivariance loss
    """
    device = idist.device()

    def evaluation_step(engine, batch):
        # evaluate equivariance loss
        netG.eval()
        with torch.no_grad():
            if not isinstance(batch[0][0], torch.Tensor):  # handle args.n_disc_steps > 1
                batch = batch[0]  # changes the batch in engine, too!
            (x, _), (u_raw, _) = batch
            x = x.to(device)
            if isinstance(u_raw, dict):  # 3d dataset
                u = u_raw['anchor_view'].to(device)
            else:
                u = u_raw.to(device)
            xi_u = transformer.normalize_tar(netG(u))  # netG outputs range [0, 1]
            loss_eq = equivariance_loss_oracle(u, xi_u, transformer, netG, args.rand_num_test)

            # both IS and FID metrics use the Inceptionv3 model for evaluation which requires images of minimum size 299 x 299 x 3
            # ref https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
            # resize function from https://kornia.readthedocs.io/en/latest/geometry.transform.html
            xi_u = resize(xi_u, (299, 299))
            x = resize(x, (299, 299))

        return dict(loss_eq=loss_eq,
                    y_pred=xi_u,  # meet the expected form of FID metric computation
                    y=x)

    engine = Engine(evaluation_step)

    # torchvision version InceptionV3
    # ref https://pytorch.org/ignite/metrics.html
    # ref https://pytorch.org/ignite/generated/ignite.metrics.FID.html
    # fid_metric = ignite.metrics.FID(device=idist.device())

    # pytorch_fid version InceptionV3
    # pytorch_fid model
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    # wrapper model to pytorch_fid model
    wrapper_model = WrapperInceptionV3(model)
    wrapper_model.eval()

    # comparable metric
    fid_metric = ignite.metrics.FID(num_features=dims, feature_extractor=wrapper_model,
                                    device=idist.device())  # the metrics will be on cpu device by default if no metric is passed.
    fid_metric.attach(engine, 'fid')

    return engine


# wrapper class as feature_extractor
# ref https://pytorch.org/ignite/generated/ignite.metrics.FID.html
class WrapperInceptionV3(nn.Module):

    def __init__(self, fid_incv3):
        super().__init__()
        self.fid_incv3 = fid_incv3

    @torch.no_grad()
    def forward(self, x):
        y = self.fid_incv3(x)
        y = y[0]
        y = y[:, :, 0, 0]
        return y


def cacl_gradient_penalty(net_D, real, fake):
    t = torch.rand(real.size(0), 1, 1, 1).to(real.device)
    t = t.expand(real.size())

    interpolates = t * real + (1 - t) * fake
    interpolates.requires_grad_(True)
    disc_interpolates = net_D(interpolates)
    grad = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True)[0]

    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), dim=1)
    loss_gp = torch.mean((grad_norm - 1) ** 2)
    return loss_gp


def main(args):
    cudnn.benchmark = True
    device = idist.device()
    # set name for log and model files: log_{prefix}.txt, ckpt-{prefix}-200.pth
    logger = utils.Logger(args.log_dir, prefix='translator', resume=args.resume)

    # load dataset
    datasets_combo = loaders.load_dataset_combo(dataset_name_tar=args.dataset_tar,
                                                dataset_name_src=args.dataset_src,
                                                data_dir_tar=args.data_dir_tar,
                                                data_dir_src=args.data_dir_src,
                                                mode=args.concat_dataset_mode,
                                                aug_src=args.aug_src,
                                                aug_tar=args.aug_tar,
                                                image_size=args.image_size,
                                                encore=args.n_disc_steps - 1,
                                                src_size=args.src_size)
    build_dataloader = partial(idist.auto_dataloader,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               pin_memory=True,
                               collate_fn=loaders.collate_fn_filter_out_none)
    args.num_classes = datasets_combo['num_classes']
    loader_train_combo = build_dataloader(datasets_combo['train_combo'], drop_last=True)

    # load data transformer
    transformer = Transformer(transform_name=args.transform,
                              dataset_src=args.dataset_src,
                              dataset_tar=args.dataset_tar, device=device)
    # build models
    models = train(args, transformer, device)
    trainer = models['trainer']
    evaluator = create_evaluator(models['netG'], transformer)

    if args.distributed:
        @trainer.on(Events.EPOCH_STARTED)
        def set_epoch(engine):  # from https://github.com/hankook/AugSelf/blob/main/pretrain.py
            # for _loader in [loader_train_combo, loader_val_tar, loader_test_tar, loader_test_src]:
            for _loader in [loader_train_combo]:
                _loader.sampler.set_epoch(engine.state.epoch)

    @trainer.on(Events.STARTED)
    def init_value(engine):
        pass  # save the model of the last epoch
    #     engine.state.best_acc = 0.
    #     # set up evaluation
    #     # best_loss_generator = 1000
    #     # best_loss_eq = 1000
    #     # best_epoch = 0

    @trainer.on(Events.EPOCH_STARTED)
    def reset_src_shuffle_map(engine):
        if args.concat_dataset_mode == 'same_bs_iter_all':
            datasets_combo['train_combo'].reset_src_shuffle_map()
            print('Reset src_shuffle_map')
            print(datasets_combo['train_combo']._src_shuffle_map[:10])

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        output = engine.state.output  # logging everything in the output
        logger.log(engine, engine.state.iteration,
                   print_msg=engine.state.iteration % args.print_freq == 0,
                   **output)

    @trainer.on(Events.COMPLETED)
    def log_training_results(engine):
        evaluator.run(loader_train_combo, max_epochs=1)
        fid_score = evaluator.state.metrics['fid']
        loss_eq = evaluator.state.output['loss_eq']
        logger.log(engine, engine.state.iteration,
                   print_msg=True,
                   loss_eq=loss_eq, fid_score=fid_score)

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_lr(engine):
        if args.decay_lr:
            for scheduler in models['schedulers']:
                scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED(every=args.plot_freq))
    def log_training_results_sampled(engine):
        evaluator.run(loader_train_combo, max_epochs=1, epoch_length=4)
        fid_score = evaluator.state.metrics['fid']
        loss_eq = evaluator.state.output['loss_eq']
        logger.log(engine, engine.state.iteration,
                   print_msg=True,
                   loss_eq=loss_eq, fid_score=fid_score)

    @trainer.on(Events.EPOCH_COMPLETED(every=args.plot_freq))
    def plot_images(engine):
        show_num = 10
        models['netG'].eval()
        # models['netD'].eval()
        with torch.no_grad():
            batch = engine.state.batch
            # do NOT need this cause the engine.state.batch has been changed in evaluator prior to this
            if not isinstance(batch[0][0], torch.Tensor):  # handle args.n_disc_steps > 1
                batch = batch[0]
            (x, _), (u_raw, _) = batch
            x = x.to(device)
            if isinstance(u_raw, dict):  # 3d dataset
                u = u_raw['anchor_view'].to(device)
            else:
                u = u_raw.to(device)
            xi_u = transformer.normalize_tar(models['netG'](u))

            if args.transform in ('3d', 'illumination'):
                if args.transform == 'illumination':
                    idx = random.randint(0, 23)
                else:
                    idx = 0
                t_u = transformer.apply(u_raw, denorm_1='src', idx=idx)
                xi_t_u = transformer.normalize_tar(models['netG'](t_u))
                imgs = torch.cat((
                    transformer.denormalize_tar(x)[:show_num],
                    transformer.denormalize_src(u)[:show_num],
                    transformer.denormalize_src(t_u)[:show_num],
                    transformer.denormalize_tar(xi_u)[:show_num],
                    transformer.denormalize_tar(xi_t_u)[:show_num]))

            else:
                t_xi_u, t_u = transformer.apply(xi_u, u, denorm_1='tar', denorm_2='src')
                xi_t_u = transformer.normalize_tar(models['netG'](t_u))
                imgs = torch.cat((
                    transformer.denormalize_tar(x)[:show_num],
                    transformer.denormalize_src(u)[:show_num],
                    transformer.denormalize_src(t_u)[:show_num],
                    transformer.denormalize_tar(xi_u)[:show_num],
                    transformer.denormalize_tar(xi_t_u)[:show_num],
                    transformer.denormalize_tar(t_xi_u)[:show_num]))

            img_grid = torchvision.utils.make_grid(imgs, nrow=show_num, normalize=True)
            logger.writer.add_image("six-row", img_grid, global_step=engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED(every=args.ckpt_freq))
    def save_ckpt(engine):
        logger.save(engine, 
                    record_epoch=False, 
                    netG=models['netG'],
                    epoch=engine.state.epoch,
                    arch=args.framework,
                    image_size=args.image_size,
                    num_res_blocks=args.num_res_blocks)

    # start training
    if args.eval_only:
        ProgressBar().attach(evaluator)
        ckpt = torch.load(args.model_dir, map_location='cpu')
        models['netG'].load_state_dict(ckpt['netG'])  # new model
        # if to eval direct FID, uncomment the line below and comment out the above two lines
        # evaluator = create_evaluator(transformer.denormalize_src, transformer)
        evaluator.run(loader_train_combo, max_epochs=1)
        fid_score = evaluator.state.metrics['fid']
        loss_eq = evaluator.state.output['loss_eq']
        logger.log(evaluator, evaluator.state.iteration,
                   print_msg=True,
                   loss_eq=loss_eq, fid_score=fid_score)

        print(f' [FID: {fid_score:.3f}]')
        print(f' [Equivariance loss: {loss_eq:.3f}]')
        return

    trainer.run(loader_train_combo, max_epochs=args.max_epochs)


if __name__ == '__main__':
    args = arg_parse()

    main(args)
