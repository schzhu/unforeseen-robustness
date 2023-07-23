import torch
import torch.nn.functional as F

import ignite
from ignite.engine import Engine
import ignite.distributed as idist
from ignite.contrib.handlers import ProgressBar


def prepare_training_batch(batch, transformer, translator, device):
    """
    function: unpack input; apply transform; send to device
    """
    (x, label), (u_raw, _) = batch
    x, label = x.to(device), label.to(device)

    # get source data (u)
    # input/transform cases: (3d data, std transform), (3d data, 3d transform)
    if isinstance(u_raw, dict):  # 3d dataset
        u = u_raw['anchor_view'].to(device)  # u is ((img, meta), (img, meta), (img, meta), last_visited=-1)
    else:
        u_raw = u_raw.to(device)
        u = u_raw

    if translator is None:
        translate = lambda x: x.detach()
    else:
        translate = translator.translate

    with torch.no_grad():
        # get domain translated source data (xi_u)
        # generator returns normalized images
        xi_u = translate(u)

        # get transformed source data (ut), and transformed domain translated source data (t_xi_u)
        ut = transformer.apply(u_raw, denorm_1='src')

        # get domain translated source data (xi_ut)
        # generator returns normalized images
        xi_ut = translate(ut)

    # x: target data, label: target label, u: source data, ut: transformed u,
    # xi_u: domain translated u, xi_ut: domain translated ut, t_xi_u: transformed xi_u
    return x, label, u, ut, xi_u.detach(), xi_ut.detach()


def prepare_training_batch_mbrdl(batch, transformer, translator, device, G, style_dim):
    """
    function: unpack input; apply transform; send to device
    """
    (x, label), (u_raw, _) = batch
    x, label = x.to(device), label.to(device)
    u_raw = u_raw.to(device)

    # get source data (u)
    # input/transform cases: (3d data, std transform), (3d data, 3d transform)
    if isinstance(u_raw, dict):  # 3d dataset
        u = u_raw['anchor_view'].to(
            device)  # u is ((img, meta), (img, meta), (img, meta), last_visited=-1)
        if not transformer.transform_names[0].startswith(
                '3d'):  # non-3d transform, extract the anchor image
            u_raw = u  # extract
    else:
        u = u_raw.to(device)

    if translator is None:
        translate = lambda x: x.detach()
    else:
        translate = translator.translate

    with torch.no_grad():
        # get domain translated source data (xi_u)
        # generator returns normalized images
        xi_u = translate(u)

        # get transformed source data (ut), and transformed domain translated source data (t_xi_u)
        u_raw = transformer.denormalize_src(u_raw)
        delta = torch.randn(u_raw.size(0), style_dim, 1, 1).cuda()
        ut = G(u_raw, delta)
        ut = transformer.normalize_src(ut)

        xi_u = transformer.denormalize_src(xi_u)
        delta = torch.randn(xi_u.size(0), style_dim, 1, 1).cuda()
        t_xi_u = G(xi_u, delta)
        t_xi_u = transformer.normalize_src(t_xi_u)

        # get domain translated source data (xi_ut)
        # generator returns normalized images
        xi_ut = translate(ut)

    # x: target data, label: target label, u: source data, ut: transformed u,
    # xi_u: domain translated u, xi_ut: domain translated ut, t_xi_u: transformed xi_u
    return x, label, u, ut, xi_u.detach(), xi_ut.detach(), t_xi_u.detach()


def prepare_test_batch(batch, transformer, device, transform=True):
    x, label = batch  # don't send label to gpu
    if isinstance(x, dict):  # 3d dataset
        x = x['anchor_view'].to(device)
    else:  # other datasets
        x = x.to(device)

    if transform:
        with torch.no_grad():
            xt = transformer.apply(x, denorm_1='tar')
            xt = xt.to(device)
    else:
        xt = x

    return xt, label


def erm(classifier,
        optimizers,
        device):
    def training_step(engine, batch):
        classifier.train()
        for o in optimizers:
            o.zero_grad()

        (x, label), (_, _) = batch
        x = x.to(device)
        label = label.to(device)

        yx = classifier(x)

        loss = F.cross_entropy(yx, label)

        outputs = dict(loss=loss)

        loss.backward()

        for o in optimizers:
            o.step()

        return outputs

    return Engine(training_step)


def transrobust(classifier,
        optimizers,
        device,
        transformer,
        translator,
        w_src,
        w_xi,
        inv_loss_type):
    def training_step(engine, batch):
        if translator is None:
            assert w_xi == 0
        classifier.train()
        for o in optimizers:
            o.zero_grad()

        x, label, u, ut, xi_u, xi_ut = prepare_training_batch(
            batch, transformer, translator, device)

        if w_xi > 0:
            y_combo = classifier(torch.cat([x, u, ut, xi_u, xi_ut], dim=0))
            yx, yu, yut, yxi_u, yxi_ut = torch.tensor_split(y_combo, 5)
        else:
            y_combo = classifier(torch.cat([x, u, ut], dim=0))
            yx, yu, yut = torch.tensor_split(y_combo, 3)

        loss_x = F.cross_entropy(yx, label)

        # robustness loss on the source
        if inv_loss_type == 'kl':
            loss_u = F.kl_div(F.log_softmax(yut, dim=1),
                              F.log_softmax(yu.detach(), dim=1),
                              reduction='batchmean', log_target=True)
        elif inv_loss_type == 'js':
            loss_u = F.kl_div(F.log_softmax(yut, dim=1),
                              F.log_softmax(yu.detach(), dim=1),
                              reduction='batchmean', log_target=True)
            loss_u += F.kl_div(F.log_softmax(yu, dim=1),
                               F.log_softmax(yut.detach(), dim=1),
                               reduction='batchmean', log_target=True)
            loss_u *= 0.5
        else:
            raise Exception(f'Unknown invariant loss type {inv_loss_type}')

        # robustness loss on the domain-translated source
        if w_xi > 0:
            if inv_loss_type == 'kl':
                loss_xi = F.kl_div(F.log_softmax(yxi_ut, dim=1),
                                   F.log_softmax(yxi_u.detach(), dim=1),
                                   reduction='batchmean', log_target=True)
            elif inv_loss_type == 'js':
                loss_xi = F.kl_div(F.log_softmax(yxi_ut, dim=1),
                                   F.log_softmax(yxi_u.detach(), dim=1),
                                   reduction='batchmean', log_target=True)
                loss_xi += F.kl_div(F.log_softmax(yxi_u, dim=1),
                                    F.log_softmax(yxi_ut.detach(), dim=1),
                                    reduction='batchmean', log_target=True)
                loss_xi *= 0.5
            else:
                raise Exception(f'Unknown invariant loss type {inv_loss_type}')
        else:
            loss_xi = 0.

        loss = (loss_x +
                loss_u * w_src +
                loss_xi * w_xi)

        outputs = dict(loss=loss,
                       loss_x=loss_x,
                       loss_u=loss_u,
                       loss_xi=loss_xi)
        loss.backward()
        for o in optimizers:
            o.step()

        return outputs

    return Engine(training_step)


def simclr(classifier,
           optimizers,
           device,
           transformer,
           projector,
           w_src,
           inv_loss_type,
           T=0.2):
    def training_step(engine, batch):
        classifier.train()
        for o in optimizers:
            o.zero_grad()

        x, label, u, ut, _, _ = prepare_training_batch(
            batch, transformer, translator=None, device=device)

        y_combo, z_combo = classifier(torch.cat([x, u, ut], dim=0), with_latent=True)
        yx, _, _ = torch.tensor_split(y_combo, 3)
        _, zu, zut = torch.tensor_split(z_combo, 3)
        hu = F.normalize(projector(zu))
        hut = F.normalize(projector(zut))

        loss_x = F.cross_entropy(yx, label)

        # simclr loss on the source
        z = torch.cat([hu, hut], 0)
        scores = torch.einsum('ik, jk -> ij', z, z).div(T)
        n = hu.shape[0]
        labels = torch.tensor(list(range(n, 2*n)) + list(range(0, n)), device=scores.device)
        masks = torch.zeros_like(scores, dtype=torch.bool)
        for i in range(2*n):
            masks[i, i] = True
        scores = scores.masked_fill(masks, float('-inf'))
        loss_u = F.cross_entropy(scores, labels)

        loss = (loss_x +
                loss_u * w_src)

        outputs = dict(loss=loss,
                       loss_x=loss_x,
                       loss_u=loss_u)
        loss.backward()
        for o in optimizers:
            o.step()

        return outputs

    return Engine(training_step)


def mbrdl(classifier,
          optimizers,
          device,
          transformer,
          w_src,
          G,
          style_dim):
    def training_step(engine, batch):

        classifier.train()
        for o in optimizers:
            o.zero_grad()

        x, label, u, ut, xi_u, xi_ut, t_xi_u = prepare_training_batch_mbrdl(
            batch, transformer, None, device, G, style_dim)

        y_combo = classifier(torch.cat([x, u, ut], dim=0))
        yx, yu, yut = torch.tensor_split(y_combo, 3)

        loss_x = F.cross_entropy(yx, label)

        # robustness loss on the source
        loss_u = F.kl_div(F.log_softmax(yut, dim=1),
                          F.log_softmax(yu.detach(), dim=1),
                          reduction='batchmean', log_target=True)

        loss = (loss_x +
                loss_u * w_src)

        outputs = dict(loss=loss,
                       loss_x=loss_x,
                       loss_u=loss_u)
        loss.backward()
        for o in optimizers:
            o.step()

        return outputs

    return Engine(training_step)


def create_evaluator(classifier,
                     loader,
                     transformer,
                     rand_num_test,
                     device,
                     eval_correct=True):
    """
    Evaluate std acc, rob acc, and inv acc
    """

    def evaluator():
        classifier.eval()
        with torch.no_grad():
            corrects, total = 0, 0
            corrects_rob, invariants_rob, total_rob = 0, 0, 0
            for batch in loader:
                x, labels = prepare_test_batch(batch, transformer, device, transform=False)
                yx = classifier(x)
                if eval_correct:
                    corrects += (yx.argmax(1).cpu() == labels).long().sum().item()
                else:
                    corrects = 0
                total += labels.shape[0]

                for i in range(rand_num_test):
                    xt, labels = prepare_test_batch(batch, transformer, device)
                    yxt = classifier(xt)

                    invariants_rob += (
                                yxt.argmax(1).cpu() == yx.argmax(1).cpu()).long().sum().item()
                    if eval_correct:
                        corrects_rob += (yxt.argmax(1).cpu() == labels).long().sum().item()
                    else:
                        corrects_rob = 0
                    total_rob += yx.shape[0]  # avoid division by zero

            corrects = idist.utils.all_reduce(corrects)
            total = idist.utils.all_reduce(total)
            corrects_rob = idist.utils.all_reduce(corrects_rob)
            invariants_rob = idist.utils.all_reduce(invariants_rob)
            total_rob = idist.utils.all_reduce(total_rob)

        return dict(std=corrects / total,
                    rob=corrects_rob / (total_rob + 1e-12),
                    inv=invariants_rob / (total_rob + 1e-12))

    return evaluator


def create_mean_std_calculator(transformer,
                               device):
    """
    Evaluate mean and std of a source dataset
    """

    def calculation_step(engine, batch):
        with torch.no_grad():
            u, _ = batch
            if isinstance(u, dict):  # 3d dataset
                u = u['anchor_view']
        return dict(mean=transformer.denormalize_src(u).mean(dim=(0, 2, 3)),
                    std=transformer.denormalize_src(u).std(dim=(0, 2, 3)))

    mean = ignite.metrics.Average(output_transform=lambda x: x['mean'])
    std = ignite.metrics.Average(output_transform=lambda x: x['std'])

    engine = Engine(calculation_step)
    ProgressBar().attach(engine)
    mean.attach(engine, 'mean')
    std.attach(engine, 'std')

    return engine
