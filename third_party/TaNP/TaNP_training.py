import os
import torch
import pickle
import random
import numpy as np
from eval import testing


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_training_checkpoint(trainer, checkpoint_filename, epoch, sample_order):
    """Atomically save everything required to continue at the next epoch."""
    checkpoint = {
        'format_version': 1,
        'epoch': epoch,
        'model_state': trainer.state_dict(),
        'optimizer_state': trainer.optimizer.state_dict(),
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'sample_order': list(sample_order),
    }
    if torch.cuda.is_available():
        checkpoint['cuda_random_state_all'] = torch.cuda.get_rng_state_all()

    temporary_filename = checkpoint_filename + '.tmp'
    torch.save(checkpoint, temporary_filename)
    os.replace(temporary_filename, checkpoint_filename)


def load_training_checkpoint(trainer, checkpoint_filename):
    """Restore model, optimizer and RNG state; return next epoch and order."""
    device = next(trainer.parameters()).device
    checkpoint = _torch_load(checkpoint_filename, map_location=device)
    trainer.load_state_dict(checkpoint['model_state'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
    random.setstate(checkpoint['python_random_state'])
    np.random.set_state(checkpoint['numpy_random_state'])
    torch.set_rng_state(checkpoint['torch_random_state'].cpu())
    if torch.cuda.is_available() and 'cuda_random_state_all' in checkpoint:
        cuda_states = [
            state.cpu() for state in checkpoint['cuda_random_state_all']
        ]
        torch.cuda.set_rng_state_all(cuda_states)
    return checkpoint['epoch'] + 1, checkpoint.get('sample_order')


def training(
    trainer,
    opt,
    train_dataset,
    test_dataset,
    batch_size,
    num_epoch,
    model_save=True,
    model_filename=None,
    logger=None,
    start_epoch=0,
    sample_order=None,
    checkpoint_filename=None,
    checkpoint_every=0,
):
    training_set_size = len(train_dataset)
    if sample_order is None:
        sample_order = list(range(training_set_size))
    if len(sample_order) != training_set_size:
        raise ValueError('Checkpoint sample order does not match the training set.')

    for epoch in range(start_epoch, num_epoch):
        random.shuffle(sample_order)
        epoch_dataset = [train_dataset[index] for index in sample_order]
        num_batch = int(training_set_size / batch_size)
        a, b, c, d = zip(*epoch_dataset)
        trainer.train()
        all_C_distribs = []
        train_loss = float('nan')
        for i in range(num_batch):
            try:
                supp_xs = list(a[batch_size*i:batch_size*(i+1)])
                supp_ys = list(b[batch_size*i:batch_size*(i+1)])
                query_xs = list(c[batch_size*i:batch_size*(i+1)])
                query_ys = list(d[batch_size*i:batch_size*(i+1)])
            except IndexError:
                continue
            train_loss, batch_C_distribs = trainer.global_update(supp_xs, supp_ys, query_xs, query_ys)
            all_C_distribs.append(batch_C_distribs)

        P5, NDCG5, MAP5, P7, NDCG7, MAP7, P10, NDCG10, MAP10 = testing(trainer, opt, test_dataset)
        logger.log(
            "{}\t{:.6f}\t TOP-5 {:.4f}\t{:.4f}\t{:.4f}\t TOP-7: {:.4f}\t{:.4f}\t{:.4f}"
            "\t TOP-10: {:.4f}\t{:.4f}\t{:.4f}".
                format(epoch, train_loss, P5, NDCG5, MAP5, P7, NDCG7, MAP7, P10, NDCG10, MAP10))
        if epoch == (num_epoch-1):
            attention_filename = os.path.join(
                os.path.dirname(model_filename), 'output_att.pkl'
            )
            with open(attention_filename, 'wb') as fp:
                pickle.dump(all_C_distribs, fp)

        should_checkpoint = (
            checkpoint_filename
            and checkpoint_every > 0
            and ((epoch + 1) % checkpoint_every == 0 or epoch == num_epoch - 1)
        )
        if should_checkpoint:
            save_training_checkpoint(
                trainer, checkpoint_filename, epoch, sample_order
            )

    if model_save:
        torch.save(trainer.state_dict(), model_filename)
