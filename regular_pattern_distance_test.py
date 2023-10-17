# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:36:19 2023

@author: Mieszko Ferens

Script to run simulation for evaluating distance metrics of Binary-coded with
Padding (BP) CRP subsets.
"""

import argparse
import pandas as pd
from pathlib import Path

import numpy as np
from pypuf.simulation import XORArbiterPUF

import scipy as sp
from math import floor
import time

class ChallengeResponseSet():
    def __init__(self, n, challenges, responses):
        self.challenge_length = n
        self.challenges = challenges
        self.responses = np.expand_dims(
            np.expand_dims(responses,axis=1),axis=1)

def create_binary_code_challenges(n, N):
    
    n_bits = 16
    lsb = np.arange(2**n_bits, dtype=np.uint8).reshape(-1,1)
    msb = lsb.copy()
    msb.sort(axis=0)

    lsb = np.unpackbits(lsb, axis=1)[:,-8:].copy()
    msb = np.unpackbits(msb, axis=1)[:,-8:].copy()

    challenges = 2*np.concatenate((msb,lsb), axis=1, dtype=np.int8) - 1
    for i in range(int(np.sqrt(n/n_bits))):
        challenges = np.insert(
            challenges, range(1,((2**i)*n_bits)+1), -1, axis=1)
    
    shift = challenges.copy()
    for i in range(1, int(n/n_bits)):
        challenges = np.append(challenges, np.roll(shift, i, axis=1), axis=0)
    
    _ , idx = np.unique(challenges, return_index=True, axis=0)
    challenges = challenges[np.sort(idx)]
    
    assert N <= len(challenges), (
        "Not enough CRPs have been generated. The limit is 2^18 - 3 CRPs.")
    challenges = challenges[:N]
    
    return challenges

def main():
    
    # Set-up logging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./Results/")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--n-bits", type=int, default=64,
                        help="Challenge length in bits.")
    parser.add_argument("--k", type=int, default=1,
                        help="The number of parallel APUF in the XOR PUF.")
    parser.add_argument("--n-CRPs", type=int, default=2**16,
                        help="Number of CRPs to be generated.")
    parser.add_argument("--metric", type=str, default="hamming",
                        help="The distance metric to be calculated (entropy " +
                        "or distance metrics from scipy).")
    args = parser.parse_args()
    
    # Generate the PUF
    puf = XORArbiterPUF(args.n_bits, args.k, args.seed)
    
    # Generate the challenges
    challenges = create_binary_code_challenges(n=args.n_bits, N=args.n_CRPs)
    
    # Get responses
    responses = puf.eval(challenges)
    
    # Convert challenges to contigous array for speed
    challenges = np.ascontiguousarray(challenges)
    
    # Get only relevant segment of BP challenges
    i = floor((args.n_CRPs - 1) / (2**16))
    start = (2**16) * i
    end = (2**16) * (i+1)
    
    t0 = time.time()
    if(args.metric == "entropy"):
        # Calculate the entropy of the challenges
        challenges = (challenges + 1) != 0
        prob = challenges[start:end]/challenges[start:end].sum(
            axis=1, keepdims=True)
        distance = (sp.special.entr(prob).sum(axis=1)/np.log(2)).mean()
    else:
        # Calculate the fractional Hamming Distance of challenge subset
        distance = sp.spatial.distance.pdist(
            challenges[start:end], args.metric).mean()
    calc_time = time.time() - t0
    
    # Calculate uniformity of responses
    uniformity = responses.mean()
    
    # Log data into csv format
    data = pd.DataFrame({"seed": [args.seed],
                         "n_bits": [args.n_bits],
                         "k": [args.k],
                         "n_CRPs": [args.n_CRPs],
                         "metric": [args.metric],
                         "distance": [distance],
                         "uniformity": [uniformity],
                         "calc_time": [calc_time]})
    filepath = Path(args.outdir + "out_dist_regular_pattern_" + str(args.k) +
                    "XOR.csv")
    if(filepath.is_file()):
        data.to_csv(filepath, header=False, index=False, mode='a')
    else:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath, header=True, index=False, mode='a')
    

if(__name__ == "__main__"):
    main()

