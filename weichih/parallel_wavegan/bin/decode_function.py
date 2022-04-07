#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode with trained Parallel WaveGAN Generator."""

import argparse
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
import yaml

from tqdm import tqdm

from parallel_wavegan.datasets import MelDataset
from parallel_wavegan.datasets import MelSCPDataset
from parallel_wavegan.utils import load_model
from parallel_wavegan.utils import read_hdf5
import pdb

# inference
# python bin/decode.py --dumpdir ../taco2/exp/pwg/mel --outdir ../taco2/exp/pwg/soundfile

def decode():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description="Decode dumped features with trained Parallel WaveGAN Generator "
                    "(See detail in parallel_wavegan/bin/decode.py).")
    parser.add_argument("--feats-scp", "--scp", default=None, type=str,
                        help="kaldi-style feats.scp file. "
                             "you need to specify either feats-scp or dumpdir.")
    parser.add_argument("--dumpdir", default="../taco2/exp/pwg/mel", type=str,
                        help="directory including feature files. "
                             "you need to specify either feats-scp or dumpdir.")
    parser.add_argument("--outdir", type=str, required=False, default="../taco2/exp/pwg/soundfile",
                        help="directory to save generated speech.")
    parser.add_argument("--checkpoint", 
                        default="../pwg_f1.pkl", 
                        type=str, help="checkpoint file to be loaded.")
    parser.add_argument("--config", 
                        default="config.yml", type=str,
                        help="yaml format configuration file. if not explicitly provided, "
                             "it will be searched in the checkpoint directory. (default=None)")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load config
    if args.config is None:
        dirname = os.path.dirname(args.checkpoint)
        args.config = os.path.join(dirname, "config.yml")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # check arguments
    if (args.feats_scp is not None and args.dumpdir is not None) or \
            (args.feats_scp is None and args.dumpdir is None):
        raise ValueError("Please specify either --dumpdir or --feats-scp.")

    # get dataset
    if args.dumpdir is not None:
        if config["format"] == "hdf5":
            mel_query = "*.h5"
            mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA
        elif config["format"] == "npy":
            mel_query = "*-feats.npy"
            mel_load_fn = np.load
        else:
            raise ValueError("Support only hdf5 or npy format.")
        dataset = MelDataset(
            args.dumpdir,
            mel_query=mel_query,
            mel_load_fn=mel_load_fn,
            return_utt_id=True,
        )
    else:
        dataset = MelSCPDataset(
            feats_scp=args.feats_scp,
            return_utt_id=True,
        )
    logging.info(f"The number of features to be decoded = {len(dataset)}.")

    # re-order dataset
    sorted_dataset = sorted(dataset, key=lambda tup: int(tup[0].split('_')[-1]))
    
    # setup model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = load_model(args.checkpoint, config)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")
    model.remove_weight_norm()
    model = model.eval().to(device)

    # start generation
    total_rtf = 0.0
    with torch.no_grad(), tqdm(sorted_dataset, desc="[decode]") as pbar:
        count = 0
        for idx, (utt_id, c) in enumerate(pbar, 1):
            # generate
            c = torch.tensor(c, dtype=torch.float).to(device)
            start = time.time()
            y = model.inference(c).view(-1)
            rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

            # save as PCM 16 bit wav file
            max_sample = abs(y.cpu().numpy()).max()
            audio = y.cpu().numpy()/max_sample

            sf.write(os.path.join(config["outdir"], f"{utt_id}_gen.wav"),
                     audio, config["sampling_rate"], "PCM_16")

            # concatenate
            if count == 0:
                X = audio
            else:
                X = np.concatenate((X, audio), axis=0)

            count += 1
            
        sf.write(os.path.join(config["outdir"], "pwg_birthday.wav"),
                 X, config["sampling_rate"], "PCM_16")

    # report average RTF
    logging.info(f"Finished generation of {idx} utterances (RTF = {total_rtf / count:.03f}).")


if __name__ == "__main__":
    decode()
