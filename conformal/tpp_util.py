import torch

from tpps.utils.events import get_events, get_window


def get_history_and_target(batch, args):
    device = args.device
    times, labels = batch['times'].to(device), batch['labels'].to(device)
    mask = (times != args.padding_id).type(times.dtype).to(device)
    times = times * args.time_scale
    window_start, window_end = get_window(times=times, window=args.window)
    past_events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end, remove_last_event=True)
    target_time = times[torch.arange(times.shape[0]), past_events.times_final_idx + 1]
    target_label = labels[torch.arange(times.shape[0]), past_events.times_final_idx + 1]
    return past_events, target_time, target_label


def get_dist(model, batch):
    features = model.get_features(batch) #returns the batch of event sequences in torch format.
    context = model.get_context(features) #returns the context vectors (as computed by the rnn) at each position of the sequence.
    return model.get_inter_time_dist(context)

def get_seq_len(batch):
    return batch.mask.sum(dim=1).long()

def at_last_event(obj, final_idx):
    return obj[torch.arange(final_idx.shape[0]), final_idx]

def until_last_event(obj, seq_len, min_event_index=0):
    # TODO How to vectorize this?
    sliced = []
    for i, seq_len_i in enumerate(seq_len):
        sliced.append(obj[i, min_event_index:seq_len_i])
    return torch.cat(sliced)
