import os
from argparse import ArgumentParser
from functools import partial
import uuid
from datetime import datetime
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from ignite.engine import Events
import ignite.distributed as idist
from ignite.handlers import BasicTimeProfiler

import loaders
import models
import trainers
from utils import Logger
import transforms

from domain_translator.interface import DomainTranslator
from mbrdl.core.models.load import MUNITModelOfNatVar


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the configuration file')

    # run mode
    parser.add_argument('--calculate_src_mean_std', action='store_true', default=False,
                        help="calculate the mean and std of a given dataset")

    # datasets
    parser.add_argument('--dataset_tar', type=str, default='cifar10',
                        help="target dataset")
    parser.add_argument('--dataset_src', type=str,
                        help="source dataset")
    parser.add_argument('--data_dir_tar', type=str, default='work_dirs/datasets')
    parser.add_argument('--data_dir_src', type=str, default='work_dirs/datasets')
    parser.add_argument('--use_validation', type=str, default=False,
                        help="whether to do validation on separate val set for model selection")
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
    parser.add_argument('--eval_ood', action='store_true', default=False,
                        help="whether to evaluate ood generalization, only CIFAR10 has ood data")
    parser.add_argument('--src_size', type=int, default=0,
                        help="use how many source data, default 0 is to use all the data")

    # data transformation
    parser.add_argument('--transform', type=str, default='3d',
                        choices=['randaugment', 'rotate', '3d'],
                        help="data transformation to which we learn robustness")
    parser.add_argument('--rand_num_test', type=int, default=20,
                        help="number of random transformations per image to estimate the "
                             "expectation during testing")
    parser.add_argument('--rand_num_train', type=int, default=1,
                        help="number of random transformations per image to estimate the "
                             "expectation during training")

    # model
    parser.add_argument('--arch', type=str, default='resnet18small', choices=models.MODEL_NAMES,
                        help="classifier architecture (default: resnet18small)")
    parser.add_argument('--translator_dir', type=str, default='work_dirs/domain_translator/logs',
                        help="directory of the pre-trained domain translator")
    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help="number of residual blocks in domain translator")

    # basic training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--mixed_precision', action='store_true', default=False,
                        help="whether to use fp16 for faster training")

    # optimization
    parser.add_argument('--base_lr', type=float, default=0.1)
    parser.add_argument('--lr_scheduler_type', type=str, default='multistep',
                        choices=['multistep', 'cosine'])
    parser.add_argument('--lr_milestones', nargs='+', type=int, default=[30, 60, 90, 100])
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    # logs
    parser.add_argument('--log_dir', type=str, default='work_dirs/classifier/logs')
    parser.add_argument('--log_alias', type=str, default=None,
                        help="additional alias str for identifying log file, csv first column, "
                             "and saved model (default: None)")
    parser.add_argument('--resume', type=int, default=None,
                        help="path to latest checkpoint (default: None)")
    parser.add_argument('--ckpt_freq', type=int, default=9999,
                        help="epoch frequency for saving trained models")  # don't save checkpoints
    parser.add_argument('--eval_freq', type=int, default=5,
                        help="epoch frequency for evaluation on validation and test set")
    parser.add_argument('--eval_trainset_freq', type=int, default=9999,
                        help="epoch frequency for evaluation on training set")  # usually for larger transform num
    parser.add_argument('--print_freq', type=int, default=100,
                        help="iteration frequency for log printing")
    parser.add_argument('--save_best', action='store_true',
                        help="whether to save the best model based on some given metric")

    # baseline: model-based
    parser.add_argument('--mbrdl_model_path', type=str,
                        help="model-based robust deep learning baseline, "
                             "trained model for approximating transformation")
    parser.add_argument('--mbrdl_model_config', type=str,
                        help="model-based robust deep learning baseline, "
                             "trained model for approximating transformation")
    parser.add_argument('--style_dim', type=int, default=8,
                        help="model-based robust deep learning baseline, style dimension used in model")

    # transrobust settings
    parser.add_argument('--framework', type=str, default='erm',
                        choices=['erm', 'uda', 'transrobust', 'oracle', 'mbrdl', 'simclr'],
                        help="training method. transrobust: robustness domain adaptation")
    parser.add_argument('--translator_name', type=str, default='none',
                        help="folder name containing the translator. "
                             "default is none and ignores translator during training")
    parser.add_argument('--translator_arch', type=str, default='wgan',
                        help="translator's architecture")
    parser.add_argument('--w_src', type=float, default=0.,
                        help="weight for robust training on the source dataset")
    parser.add_argument('--w_xi', type=float, default=0.,
                        help="weight for robust training on the domain translated images")
    # parser.add_argument('--w_equivariant', type=float, default=0,
    #                     help="weight for regularizing classifier to minimize equivariant loss"
    #                          "(default: 0)")
    parser.add_argument('--inv_loss_type', type=str, default='kl',
                        help="loss type for regularizing robustness on source "
                             "and domain-translated source")

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

    # args.batch_size = args.batch_size * idist.get_world_size()
    # args.lr = args.base_lr * args.batch_size / 256
    args.lr = args.base_lr

    # set-up log path
    log_folder_name = args.framework
    if args.framework == 'transrobust':
        log_folder_name = '_'.join([log_folder_name, args.translator_name])
    args.log_dir = os.path.join(args.log_dir, log_folder_name)

    # setup log file name
    unique_name = f'{datetime.now().strftime("%y%m%d_%H%M")}_' \
                  f'{str(uuid.uuid4())[:4]}'
    if args.log_alias is not None:
        unique_name = args.log_alias + '_' + unique_name

    # name str for identifying log file, csv first column, and saved model
    args.log_id = '_'.join([args.dataset_tar,
                            args.dataset_src,
                            args.transform,
                            args.framework,
                            args.translator_name,
                            str(args.w_src),
                            str(args.w_xi),
                            unique_name])

    print(f"Working dir is {os.getcwd()}, log dir is {args.log_dir}, log id is {args.log_id}")
    print('Args:', args)

    return args


def erm(args, device):
    """ Empirical risk minimization """

    build_model = partial(idist.auto_model, sync_bn=True)
    classifier = build_model(models.Classifier(args))

    SGD = partial(optim.SGD, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    build_optim = lambda x: idist.auto_optim(SGD(x))
    optimizers = [build_optim(list(classifier.parameters()))]
    if args.lr_scheduler_type == 'cosine':
        schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizers[0], args.max_epochs)]
    elif args.lr_scheduler_type == 'multistep':
        schedulers = [optim.lr_scheduler.MultiStepLR(optimizers[0], args.lr_milestones)]
    else:
        raise Exception(f'Unknown lr scheduler {args.lr_scheduler_type}')

    trainer = trainers.erm(classifier=classifier,
                           optimizers=optimizers,
                           device=device)

    return dict(classifier=classifier,
                optimizers=optimizers,
                schedulers=schedulers,
                trainer=trainer)


def mbrdl(args, transformer, device):
    """ Empirical risk minimization """
    build_model = partial(idist.auto_model, sync_bn=True)
    classifier = build_model(models.Classifier(args))

    SGD = partial(optim.SGD, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    build_optim = lambda x: idist.auto_optim(SGD(x))
    optimizers = [build_optim(list(classifier.parameters()))]
    if args.lr_scheduler_type == 'cosine':
        schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizers[0], args.max_epochs)]
    elif args.lr_scheduler_type == 'multistep':
        schedulers = [optim.lr_scheduler.MultiStepLR(optimizers[0], args.lr_milestones)]
    else:
        raise Exception(f'Unknown lr scheduler {args.lr_scheduler_type}')
    # schedulers = [optim.lr_scheduler.StepLR(optimizers[0], args.step_lr)]

    G = MUNITModelOfNatVar(args.mbrdl_model_path, reverse=False,
                           config=args.mbrdl_model_config).cuda()

    trainer = trainers.mbrdl(classifier=classifier,
                             optimizers=optimizers,
                             device=device,
                             transformer=transformer,
                             w_src=args.w_src,
                             G=G,
                             style_dim=args.style_dim)

    return dict(classifier=classifier,
                optimizers=optimizers,
                schedulers=schedulers,
                trainer=trainer)


def simclr(args, transformer, device):
    """ Auxiliary SimCLR loss on the source """
    classifier = idist.auto_model(models.Classifier(args), sync_bn=True)

    SGD = partial(optim.SGD, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    build_optim = lambda x: idist.auto_optim(SGD(x))
    optimizers = [build_optim(list(classifier.parameters()))]
    if args.lr_scheduler_type == 'cosine':
        schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizers[0], args.max_epochs)]
    elif args.lr_scheduler_type == 'multistep':
        schedulers = [optim.lr_scheduler.MultiStepLR(optimizers[0], args.lr_milestones)]
    else:
        raise Exception(f'Unknown lr scheduler {args.lr_scheduler_type}')
    projector = models.load_mlp(classifier.num_backbone_features,
                                classifier.num_backbone_features,
                                128,
                                num_layers=2,
                                last_bn=False)
    projector = idist.auto_model(projector, sync_bn=True)
    trainer = trainers.simclr(classifier=classifier,
                              optimizers=optimizers,
                              device=device,
                              transformer=transformer,
                              projector=projector,
                              w_src=args.w_src,
                              inv_loss_type=args.inv_loss_type)

    return dict(classifier=classifier,
                optimizers=optimizers,
                schedulers=schedulers,
                trainer=trainer)


def transrobust(args, transformer, translator, device):
    """ Empirical risk minimization """
    classifier = idist.auto_model(models.Classifier(args), sync_bn=True)

    SGD = partial(optim.SGD, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    build_optim = lambda x: idist.auto_optim(SGD(x))
    optimizers = [build_optim(list(classifier.parameters()))]
    if args.lr_scheduler_type == 'cosine':
        schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizers[0], args.max_epochs)]
    elif args.lr_scheduler_type == 'multistep':
        schedulers = [optim.lr_scheduler.MultiStepLR(optimizers[0], args.lr_milestones)]
    else:
        raise Exception(f'Unknown lr scheduler {args.lr_scheduler_type}')

    trainer = trainers.transrobust(classifier=classifier,
                                   optimizers=optimizers,
                                   device=device,
                                   transformer=transformer,
                                   translator=translator,
                                   w_src=args.w_src,
                                   w_xi=args.w_xi,
                                   inv_loss_type=args.inv_loss_type)

    return dict(classifier=classifier,
                optimizers=optimizers,
                schedulers=schedulers,
                trainer=trainer)


def main(local_rank, args):
    cudnn.benchmark = True
    device = idist.device()
    logger = Logger(args.log_dir, prefix=args.log_id, resume=args.resume)

    # load dataset
    datasets_combo = loaders.load_dataset_combo(dataset_name_tar=args.dataset_tar,
                                                dataset_name_src=args.dataset_src,
                                                data_dir_tar=args.data_dir_tar,
                                                data_dir_src=args.data_dir_src,
                                                mode=args.concat_dataset_mode,
                                                aug_src=args.aug_src,
                                                aug_tar=args.aug_tar,
                                                image_size=args.image_size,
                                                rand_num_train=args.rand_num_train,
                                                src_size=args.src_size)
    if args.eval_ood:
        dataset_ood_1 = loaders.load_ood_dataset(dataset='cifar10_1', datadir=args.data_dir_tar)
        dataset_ood_2 = loaders.load_ood_dataset(dataset='cifar10_2', datadir=args.data_dir_tar)
    build_dataloader = partial(idist.auto_dataloader,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               pin_memory=True,
                               collate_fn=loaders.collate_fn_filter_out_none)
    args.num_classes = datasets_combo['num_classes']
    loader_train_combo = build_dataloader(datasets_combo['train_combo'], drop_last=True)
    loader_train_tar = build_dataloader(datasets_combo['train_tar'], drop_last=False)
    loader_test_tar = build_dataloader(datasets_combo['test_tar'], drop_last=False)
    loader_test_src = build_dataloader(datasets_combo['test_src'], drop_last=False)
    if args.eval_ood:
        loader_ood_1 = build_dataloader(dataset_ood_1, drop_last=False)
        loader_ood_2 = build_dataloader(dataset_ood_2, drop_last=False)
    if args.use_validation:
        loader_val_tar = build_dataloader(datasets_combo['val_tar'], drop_last=False)
    else:
        loader_val_tar = None
    print(f'Target training set has {len(loader_train_tar)} batches')

    # load data transformer
    transformer = transforms.Transformer(args.transform, args.dataset_tar,
                                         args.dataset_src, device=device)
    # transformer.transform = idist.auto_model(transformer.transform)
    # transformer = idist.auto_model(transformer)

    # load domain translator
    if args.translator_name != 'none':
        translator = DomainTranslator(translator_dir=args.translator_dir,
                                      translator_name=args.translator_name,
                                      # generator_output_scale=transformer.scale_tar,
                                      normalizer_tar=transformer.normalize_tar)
        translator.translator = idist.auto_model(translator.translator)
    else:
        translator = None

    # build models
    if args.framework == 'erm':
        models = erm(args, device)
    elif args.framework == 'transrobust':
        models = transrobust(args, transformer, translator, device)
    elif args.framework == 'uda':
        models = transrobust(args, transformer, translator, device)
    elif args.framework == 'oracle':
        models = transrobust(args, transformer, translator, device)
    elif args.framework == 'mbrdl':
        models = mbrdl(args, transformer, device)
    elif args.framework == 'simclr':
        models = simclr(args, transformer, device)
    else:
        raise Exception(f'Unknown framework {args.framework}')

    trainer = models['trainer']
    build_evaluator = partial(trainers.create_evaluator,
                              classifier=models['classifier'],
                              transformer=transformer,
                              device=device,
                              rand_num_test=args.rand_num_test)

    if args.use_validation:
        evaluator_tar_val = build_evaluator(loader=loader_val_tar)
    evaluator_tar_train = build_evaluator(loader=loader_train_tar)
    evaluator_tar_test = build_evaluator(loader=loader_test_tar)
    evaluator_src_test = build_evaluator(loader=loader_test_src, eval_correct=False)
    if args.eval_ood:
        evaluator_ood_1 = build_evaluator(loader=loader_ood_1, rand_num_test=0)
        evaluator_ood_2 = build_evaluator(loader=loader_ood_2, rand_num_test=0)

    if args.distributed:
        @trainer.on(Events.EPOCH_STARTED)
        def set_epoch(engine):
            if args.use_validation:
                _loaders = [loader_train_combo, loader_val_tar, loader_test_tar, loader_test_src]
            else:
                _loaders = [loader_train_combo, loader_test_tar, loader_test_src]
            for _loader in _loaders:
                _loader.sampler.set_epoch(engine.state.epoch)

    @trainer.on(Events.EPOCH_STARTED)
    def reset_src_shuffle_map(engine):
        if args.concat_dataset_mode == 'same_bs_iter_all':
            datasets_combo['train_combo'].reset_src_shuffle_map()
            print('Reset src_shuffle_map')
            print(datasets_combo['train_combo']._src_shuffle_map[:10])

    @trainer.on(Events.STARTED)
    def init_value(engine):
        engine.state_dict_user_keys.append('best_acc')
        engine.state.best_acc = 0.

    @trainer.on(Events.ITERATION_STARTED)
    def log_lr(engine):
        lrs = {}
        for i, optimizer in enumerate(models['optimizers']):
            for j, pg in enumerate(optimizer.param_groups):
                lrs[f'lr/{i}-{j}'] = pg['lr']
        logger.log(engine, engine.state.iteration, print_msg=False, **lrs)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log(engine):
        logger.log(engine, engine.state.iteration,
                   print_msg=engine.state.iteration % args.print_freq == 0,
                   **engine.state.output)

        logger.log(engine, engine.state.iteration,
                   print_msg=False,
                   **engine.state.output)

    @trainer.on(Events.EPOCH_COMPLETED(every=args.eval_freq))
    def evaluate(engine):
        stats_tar_test = evaluator_tar_test()
        stats_src_test = evaluator_src_test()
        if args.eval_ood:
            stats_ood_1 = evaluator_ood_1()
            stats_ood_2 = evaluator_ood_2()
        else:
            stats_ood_1 = {'std': 0}
            stats_ood_2 = {'std': 0}
        if args.use_validation:
            stats_tar_val = evaluator_tar_val()
            new_best_acc = stats_tar_val['rob']
            msg = dict(std_val=stats_tar_val['std'],
                       rob_val=stats_tar_val['rob'],
                       inv_val=stats_tar_val['inv'],
                       std_test=stats_tar_test['std'],
                       rob_test=stats_tar_test['rob'],
                       inv_test=stats_tar_test['inv'],
                       inv_test_src=stats_src_test['inv'],
                       ood_1=stats_ood_1['std'],
                       ood_2=stats_ood_2['std'])
        else:
            new_best_acc = stats_tar_test['rob']
            msg = dict(std_test=stats_tar_test['std'],
                       rob_test=stats_tar_test['rob'],
                       inv_test=stats_tar_test['inv'],
                       inv_test_src=stats_src_test['inv'],
                       ood_1=stats_ood_1['std'],
                       ood_2=stats_ood_2['std'])
        logger.log(engine, engine.state.epoch, **msg)

        if engine.state.best_acc < new_best_acc:
            logger.log_msg('New best test accuracy found')
            engine.state.best_acc = new_best_acc
            logger.log_csv([stats_tar_test['std'],
                            stats_tar_test['rob'],
                            stats_tar_test['inv'],
                            stats_src_test['inv'],
                            stats_ood_1['std'],
                            stats_ood_2['std'],
                            engine.state.epoch])
            if args.save_best:
                logger.save(engine, record_epoch=False, **models)  # save best model

    @trainer.on(Events.EPOCH_COMPLETED(every=args.eval_trainset_freq))
    def evaluate_train(engine):
        stats_tar_train = evaluator_tar_train()
        logger.log(engine, engine.state.epoch,
                   std_tar=stats_tar_train['std'],
                   rob_tar=stats_tar_train['rob'],
                   inv_tar=stats_tar_train['inv'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_lr(engine):
        for scheduler in models['schedulers']:
            scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED(every=args.ckpt_freq))
    def save_ckpt(engine):
        logger.save(engine, **models)

    @trainer.on(Events.COMPLETED)
    def report(engine):
        logger.save_csv(args.log_id)

    if args.resume is not None:
        @trainer.on(Events.STARTED)
        def load_state(engine):
            ckpt = torch.load(os.path.join(args.log_dir, f'ckpt-{args.resume}.pth'),
                              map_location='cpu')
            for k, v in models.items():
                if isinstance(v, nn.parallel.DistributedDataParallel):
                    v = v.module

                if hasattr(v, 'state_dict'):
                    v.load_state_dict(ckpt[k])

                if type(v) is list and hasattr(v[0], 'state_dict'):
                    for i, x in enumerate(v):
                        x.load_state_dict(ckpt[k][i])

                if type(v) is dict and k == 'ss_predictor':
                    for y, x in v.items():
                        x.load_state_dict(ckpt[k][y])

    # profiler = BasicTimeProfiler()
    # profiler.attach(trainer)
    # @trainer.on(Events.ITERATION_COMPLETED)
    # def log_intermediate_results():
    #     profiler.print_results(profiler.get_results())

    if args.calculate_src_mean_std:
        loader_train_src = build_dataloader(datasets_combo['train_src'], drop_last=False)
        mean_std_calculator = trainers.create_mean_std_calculator(transformer, device)
        mean_std_calculator.run(loader_train_src, max_epochs=1)
        print(args.dataset_src, mean_std_calculator.state.metrics['mean'])
        print(args.dataset_src, mean_std_calculator.state.metrics['std'])
        return

    trainer.run(loader_train_combo, max_epochs=args.max_epochs)


if __name__ == '__main__':
    args = arg_parse()

    if not args.distributed:
        with idist.Parallel() as parallel:
            parallel.run(main, args)
    else:
        assert torch.cuda.is_available(), torch.cuda.is_available()
        assert torch.backends.cudnn.enabled, "Nvidia/Amp requires cudnn backend to be enabled."

        print('Using distributed training')
        with idist.Parallel('nccl', nproc_per_node=torch.cuda.device_count()) as parallel:
            parallel.run(main, args)
