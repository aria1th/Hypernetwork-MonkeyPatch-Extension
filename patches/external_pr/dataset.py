# source:https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4886/files

import os
import sys

import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

from ..hnutil import get_closest
import random
from collections import defaultdict
from random import Random
import tqdm
from modules import devices, shared
import re

from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

re_numbers_at_start = re.compile(r"^[-\d]+\s*")

random_state_manager = Random(None)
shuffle = random_state_manager.shuffle
choice = random_state_manager.choice
choices = random_state_manager.choices


def set_rng(seed=None):
    random_state_manager.seed(seed)


class DatasetEntry:
    def __init__(self, filename=None, filename_text=None, latent_dist=None, latent_sample=None, cond=None,
                 cond_text=None, pixel_values=None):
        self.filename = filename
        self.filename_text = filename_text
        self.latent_dist = latent_dist
        self.latent_sample = latent_sample
        self.cond = cond
        self.cond_text = cond_text
        self.pixel_values = pixel_values


class PersonalizedBase(Dataset):
    def __init__(self, data_root, width, height, repeats, flip_p=0.5, placeholder_token="*", model=None,
                 cond_model=None, device=None, template_file=None, include_cond=False, batch_size=1, gradient_step=1,
                 shuffle_tags=False, tag_drop_out=0, latent_sampling_method='once', latent_sampling_std=-1):
        re_word = re.compile(shared.opts.dataset_filename_word_regex) if len(
            shared.opts.dataset_filename_word_regex) > 0 else None
        seed = random.randrange(sys.maxsize)
        set_rng(seed) # reset forked RNG state when we create dataset.
        print(f"Dataset seed was set to f{seed}")
        self.placeholder_token = placeholder_token

        self.width = width
        self.height = height
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        self.dataset = []

        with open(template_file, "r") as file:
            lines = [x.strip() for x in file.readlines()]

        self.lines = lines

        assert data_root, 'dataset directory not specified'
        assert os.path.isdir(data_root), "Dataset directory doesn't exist"
        assert os.listdir(data_root), "Dataset directory is empty"

        self.image_paths = [os.path.join(data_root, file_path) for file_path in os.listdir(data_root)] # We assert batch size > 1 can work, by having multiple same-size images
        # But note that we can't stack tensors with other size. so its not working now.
        self.shuffle_tags = shuffle_tags
        self.tag_drop_out = tag_drop_out
        groups = defaultdict(list)

        print("Preparing dataset...")
        _i = 0
        for path in tqdm.tqdm(self.image_paths):
            if shared.state.interrupted:
                raise Exception("inturrupted")
            try: # apply variable size here
                image = Image.open(path).convert('RGB')
                w, h = image.size
                r = max(1, w / self.width, h / self.height) # divide by this
                amp = min(self.width / w, self.height / h) # if amp < 1, then ignore, else, multiply.
                if amp > 1:
                    w, h = w * amp, h * amp
                w, h = int(w/r), int(h/r)
                w, h = get_closest(w), get_closest(h)
                image = image.resize((w,h), PIL.Image.LANCZOS)
            except Exception:
                continue

            text_filename = os.path.splitext(path)[0] + ".txt"
            filename = os.path.basename(path)

            if os.path.exists(text_filename):
                with open(text_filename, "r", encoding="utf8") as file:
                    filename_text = file.read()
            else:
                filename_text = os.path.splitext(filename)[0]
                filename_text = re.sub(re_numbers_at_start, '', filename_text)
                if re_word:
                    tokens = re_word.findall(filename_text)
                    filename_text = (shared.opts.dataset_filename_join_string or "").join(tokens)

            npimage = np.array(image).astype(np.uint8)
            npimage = (npimage / 127.5 - 1.0).astype(np.float32)

            torchdata = torch.from_numpy(npimage).permute(2, 0, 1).to(device=device, dtype=torch.float32)
            latent_sample = None

            with torch.autocast("cuda"):
                latent_dist = model.encode_first_stage(torchdata.unsqueeze(dim=0))

            if latent_sampling_method == "once" or (
                    latent_sampling_method == "deterministic" and not isinstance(latent_dist,
                                                                                 DiagonalGaussianDistribution)):
                latent_sample = model.get_first_stage_encoding(latent_dist).squeeze().to(devices.cpu)
                latent_sampling_method = "once"
                entry = DatasetEntry(filename=path, filename_text=filename_text, latent_sample=latent_sample)
            elif latent_sampling_method == "deterministic":
                # Works only for DiagonalGaussianDistribution
                latent_dist.std = 0
                latent_sample = model.get_first_stage_encoding(latent_dist).squeeze().to(devices.cpu)
                entry = DatasetEntry(filename=path, filename_text=filename_text, latent_sample=latent_sample)
            elif latent_sampling_method == "random":
                if latent_sampling_std != -1:
                    assert latent_sampling_std > 0, f"Cannnot apply negative standard deviation {latent_sampling_std}"
                    print(f"Applying patch, clipping std from {torch.max(latent_dist.std).item()} to {latent_sampling_std}...")
                    latent_dist.std.clip_(latent_sampling_std)
                entry = DatasetEntry(filename=path, filename_text=filename_text, latent_dist=latent_dist)

            if not (self.tag_drop_out != 0 or self.shuffle_tags):
                entry.cond_text = self.create_text(filename_text)

            if include_cond and not (self.tag_drop_out != 0 or self.shuffle_tags):
                with torch.autocast("cuda"):
                    entry.cond = cond_model([entry.cond_text]).to(devices.cpu).squeeze(0)
            groups[image.size].append(_i)  #record indexes of images in dataset into group. When we pull batch, try using single group to make torch.stack work.
            _i += 1
            self.dataset.append(entry)
            del torchdata
            del latent_dist
            del latent_sample
        self.groups = list(groups.values())
        self.length = len(self.dataset)
        assert self.length > 0, "No images have been found in the dataset."
        self.batch_size = min(batch_size, self.length)
        self.gradient_step = min(gradient_step, self.length // self.batch_size)
        self.latent_sampling_method = latent_sampling_method

    def create_text(self, filename_text):
        text = choice(self.lines)
        text = text.replace("[name]", self.placeholder_token)
        tags = filename_text.split(',')
        if self.tag_drop_out != 0:
            tags = [t for t in tags if random.random() > self.tag_drop_out]
        if self.shuffle_tags:
            shuffle(tags)
        text = text.replace("[filewords]", ','.join(tags))
        return text

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        entry = self.dataset[i]
        if self.tag_drop_out != 0 or self.shuffle_tags:
            entry.cond_text = self.create_text(entry.filename_text)
        if self.latent_sampling_method == "random":
            entry.latent_sample = shared.sd_model.get_first_stage_encoding(entry.latent_dist).to(devices.cpu)
        return entry

class GroupedBatchSampler(Sampler):
    # See https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/6620
    def __init__(self, data_source: PersonalizedBase, batch_size: int):
        n = len(data_source)
        self.groups = data_source.groups
        self.len = n_batch = n // batch_size
        expected = [len(g) / n * n_batch * batch_size for g in data_source.groups]
        self.base = [int(e) // batch_size for e in expected]
        self.n_rand_batches = n_batch - sum(self.base)
        self.probs = [e % batch_size/self.n_rand_batches/batch_size if self.n_rand_batches > 0 else 0 for e in expected]
        self.batch_size = batch_size

    def __len__(self):
        return self.len

    def __iter__(self):
        b = self.batch_size
        batches = []
        for g in self.groups:
            shuffle(g)
            batches.extend(g[i*b:(i+1)*b] for i in range(len(g) // b))
        for _ in range(self.n_rand_batches):
            rand_group = choices(self.groups, self.probs)[0]
            batches.append(choices(rand_group, k=b))
        shuffle(batches)
        yield from batches

class PersonalizedDataLoader(DataLoader):
    def __init__(self, dataset, latent_sampling_method="once", batch_size=1, pin_memory=False):
        super(PersonalizedDataLoader, self).__init__(dataset, batch_sampler=GroupedBatchSampler(dataset, batch_size), pin_memory=pin_memory)
        if latent_sampling_method == "random":
            self.collate_fn = collate_wrapper_random
        else:
            self.collate_fn = collate_wrapper


class BatchLoader:
    def __init__(self, data):
        self.cond_text = [entry.cond_text for entry in data]
        self.cond = [entry.cond for entry in data]
        self.latent_sample = torch.stack([entry.latent_sample for entry in data]).squeeze(1)
        self.filename = [entry.filename for entry in data]
        # self.emb_index = [entry.emb_index for entry in data]
        # print(self.latent_sample.device)

    def pin_memory(self):
        self.latent_sample = self.latent_sample.pin_memory()
        return self


def collate_wrapper(batch):
    return BatchLoader(batch)


class BatchLoaderRandom(BatchLoader):
    def __init__(self, data):
        super().__init__(data)

    def pin_memory(self):
        return self


def collate_wrapper_random(batch):
    return BatchLoaderRandom(batch)
