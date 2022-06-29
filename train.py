from imports import (os, np, torch, tqdm, image_histogram2d, transforms)
from utils import (available_ram, set_seed, AverageMeter, ToTensor, depthnorm, fetch_params,
                   set_prefix, logprogress, get_params, prepare_dataset, prepare_training, RandomRotate,
                   RandomHorizontalFlip, RandomHorizontalTranslate, RandomVerticalTranslate, RandomCrop,
                   Resize)

available_ram()

_SUPPORTED_DATASETS = ['balser_v1', 'nyu_v2', 'diode', 'balser_v2']
_SUPPORTED_MODELS = ['dgunet']
_SUPPORTED_LOSSES = ['l1', 'l2', 'hist']
_TRAIN_TRANSFORM = train_transform = transforms.Compose([
            RandomRotate(),
            RandomHorizontalTranslate(),
            RandomVerticalTranslate(),
            RandomHorizontalFlip(),
            RandomCrop(),
            Resize((480, 640)),
            ToTensor()])
_LOAD_CHECKPOINT = True
_LOAD_OPTIMIZER = True
_BEST_VAL_ERR = 1.0e20
_SEED_VALUE = 123456789
_START_EPOCH = 0
_EPOCHS = 1500
_DEVICE = set_seed(_SEED_VALUE)
_CONFIG_FILE = os.path.join(os.getcwd(), 'config.json')
parameters = fetch_params(get_params(_CONFIG_FILE), supported_dataset=_SUPPORTED_DATASETS, supported_models=_SUPPORTED_MODELS)
_PREFIX = set_prefix(parameters)
if parameters['dataset_name'] == 'balser_v1':
    _MAX_DEPTH = 1300
elif parameters['dataset_name'] == 'nyu_v2':
    _MAX_DEPTH = 10
elif parameters['dataset_name'] == 'diode':
    _MAX_DEPTH = 10
else:
    _MAX_DEPTH = 1500


def print_constant_values():
    print("Supported datasets names: {}\n".format(_SUPPORTED_DATASETS),
          "Supported models names: {}\n".format(_SUPPORTED_MODELS),
          "Supported losses names: {}\n".format(_SUPPORTED_LOSSES),
          "Load checkpoint: {}\n".format(_LOAD_CHECKPOINT),
          "Load optimizer: {}\n".format(_LOAD_OPTIMIZER),
          "Best_value error ")


def print_supported_elements():
    print("Supported datasets names: {}\n".format(_SUPPORTED_DATASETS),
          "Supported models names: {}\n".format(_SUPPORTED_MODELS),
          "Supported losses names: {}\n".format(_SUPPORTED_LOSSES))


data_train, data_val, example_sample = prepare_dataset(parameters, True, _TRAIN_TRANSFORM)
start_epoch, writer, loss_func, optimizer, scheduler, model_ckpt_file, model = prepare_training(
    parameters, _PREFIX, _START_EPOCH, _DEVICE, load_checkpoint=_LOAD_CHECKPOINT,
    load_optimizer=_LOAD_OPTIMIZER)

if __name__ == '__main__':
    #
    print(print_constant_values())
    print(print_supported_elements())
    print("Begininning training")
    #
    n_bins = 2
    start_epoch = _START_EPOCH

    for epoch in range(_EPOCHS):
        np.random.seed(_SEED_VALUE + epoch)
        log_epoch = epoch + start_epoch
        t_losses = AverageMeter()
        v_losses = AverageMeter()
        t_N = len(data_train)
        v_N = len(data_val)
        trange = tqdm(data_train)
        vrange = tqdm(data_val)
        # Switch to train mode
        model.train()
        trange.set_description(desc='Training Epoch {}/{}'.format(log_epoch + 1, _EPOCHS + start_epoch))
        vrange.set_description(desc='Testing Epoch {}/{}'.format(log_epoch + 1, _EPOCHS + start_epoch))
        # TRAIN LOOP
        for i, sample_batched in enumerate(trange):
            optimizer.zero_grad()
            #
            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
            with torch.no_grad():
                mask = torch.gt(sample_batched['mask'], 0).cuda()
            depth = depthnorm(depth, _MAX_DEPTH)
            assert ~image.isnan().any()
            assert ~image.isinf().any()
            assert ~depth.isnan().any()
            assert ~depth.isinf().any()
            assert ~mask.isnan().any()
            assert ~mask.isinf().any()
            # Predict
            output = model(image.cuda())
            assert ~output.isnan().any()
            assert ~output.isinf().any()
            # Compute loss
            if parameters['loss_type'] == 'hist':
                if log_epoch % 50:
                    n_bins *= 2
                    print("Number of bins : {}".format(n_bins))
                _, depth_hist = image_histogram2d(depth, min=0.0, max=1.0,
                                                  n_bins=n_bins, return_pdf=True)
                _, output_hist = image_histogram2d(output, min=0.0, max=1.0,
                                                   n_bins=n_bins, return_pdf=True)
                curr_loss = loss_func(depth_hist, output_hist)
                curr_loss = torch.mean(curr_loss)
            elif parameters['loss_type'] == 'l1':
                curr_loss = loss_func(output[mask], depth[mask])
            elif parameters['loss_type'] == 'l2':
                curr_loss = loss_func(output[mask], depth[mask])
            else:
                print("Loss type not supported")
                exit()
            assert ~curr_loss.isnan().any()
            assert ~curr_loss.isinf().any()
            # Update step
            t_losses.update(curr_loss.data.item(), image.size(0))
            curr_loss.backward()
            optimizer.step()
            # Log progress
            niter = log_epoch * t_N + i
            trange.set_postfix_str("Loss :{}/ Average :{}".format(t_losses.val, t_losses.avg))
            # Log to tensorboard
            writer.add_scalar('Train/Loss Step', t_losses.val, niter)
            logprogress(model, writer, example_sample, niter, maxdepth=_MAX_DEPTH )
            # Save Model
            if t_losses.avg < _BEST_VAL_ERR:
                best_val_err = t_losses.avg
                torch.save({
                    'epoch': log_epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, model_ckpt_file)
            # Record epoch's intermediate results
            del image, depth, mask, curr_loss, depth_hist, output_hist, output
        writer.add_scalar('Train/Loss Epoch', t_losses.avg, log_epoch)
        if log_epoch % 10 == 0:
            print("Last learning rate : {}".format(scheduler.get_last_lr()))
        scheduler.step()
        trange.close()
        # VALIDATION LOOP
        for i, sample_batched in enumerate(vrange):
            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
            with torch.no_grad():
                mask = torch.gt(sample_batched['mask'], 0).cuda()
            depth = depthnorm(depth, _MAX_DEPTH)
            assert ~image.isnan().any()
            assert ~image.isinf().any()
            assert ~depth.isnan().any()
            assert ~depth.isinf().any()
            assert ~mask.isnan().any()
            assert ~mask.isinf().any()
            # Predict
            output = model(image.cuda())
            assert ~output.isnan().any()
            assert ~output.isinf().any()
            # Compute loss
            if parameters['loss_type'] == 'hist':
                if log_epoch % 50:
                    n_bins *= 2
                _, depth_hist = image_histogram2d(depth, min=0.0, max=1.0, n_bins=n_bins,
                                                  return_pdf=True)
                _, output_hist = image_histogram2d(output, min=0.0, max=1.0, n_bins=n_bins,
                                                   return_pdf=True)
                curr_loss = loss_func(depth_hist, output_hist)
                curr_loss = torch.mean(curr_loss)
            elif parameters['loss_type'] == 'l1':
                curr_loss = loss_func(output[mask], depth[mask])
            elif parameters['loss_type'] == 'l2':
                curr_loss = loss_func(output[mask], depth[mask])
            else:
                print("Loss type not defined")
            assert ~curr_loss.isnan().any()
            assert ~curr_loss.isinf().any()
            # Log progress
            niter = log_epoch * v_N + i
            vrange.set_postfix_str("Validation Loss :{}/ Average :{}".format(v_losses.val, v_losses.avg))
            # Log to tensorboard
            writer.add_scalar('Validation/Loss Step', v_losses.val, niter)
            # Record epoch's intermediate results
            del image, depth, mask, curr_loss, depth_hist, output_hist, output
        vrange.close()
        writer.add_scalar('Validation/Loss Epoch', v_losses.avg, log_epoch)
