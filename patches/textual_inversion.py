import csv
import os

import modules.textual_inversion.textual_inversion
from modules import shared

delayed_values = {}


def write_loss(log_directory, filename, step, epoch_len, values):
    if shared.opts.training_write_csv_every == 0:
        return

    if (step + 1) % shared.opts.training_write_csv_every != 0:
        return
    write_csv_header = False if os.path.exists(os.path.join(log_directory, filename)) else True
    try:
        with open(os.path.join(log_directory, filename), "a+", newline='') as fout:
            csv_writer = csv.DictWriter(fout, fieldnames=["step", "epoch", "epoch_step", *(values.keys())])

            if write_csv_header:
                csv_writer.writeheader()
            if log_directory + filename in delayed_values:
                delayed = delayed_values[log_directory + filename]
                for step, epoch, epoch_step, values in delayed:
                    csv_writer.writerow({
                        "step": step,
                        "epoch": epoch,
                        "epoch_step": epoch_step + 1,
                        **values,
                    })
                delayed.clear()
            epoch = step // epoch_len
            epoch_step = step % epoch_len
            csv_writer.writerow({
                "step": step + 1,
                "epoch": epoch,
                "epoch_step": epoch_step + 1,
                **values,
            })
    except OSError:
        epoch, epoch_step = divmod(step, epoch_len)
        if log_directory + filename in delayed_values:
            delayed_values[log_directory + filename].append((step + 1, epoch, epoch_step, values))
        else:
            delayed_values[log_directory + filename] = [(step+1, epoch, epoch_step, values)]

modules.textual_inversion.textual_inversion.write_loss = write_loss
