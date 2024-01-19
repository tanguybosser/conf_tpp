import torch

from tpps.utils.events import get_events, get_window


def get_history_and_target(batch, args):
    device = args.device
    times, labels = batch['times'].to(device), batch['labels'].to(device)
    mask = (times != args.padding_id).type(times.dtype).to(device)
    times = times * args.time_scale
    window_start, window_end = get_window(times=times, window=args.window)
    past_events = get_events(
        times=times,
        mask=mask,
        labels=labels,
        window_start=window_start,
        window_end=window_end,
        remove_last_event=True,
    )
    target_time = times[torch.arange(times.shape[0]), past_events.times_final_idx + 1]
    target_label = labels[torch.arange(times.shape[0]), past_events.times_final_idx + 1]
    return past_events, target_time, target_label
