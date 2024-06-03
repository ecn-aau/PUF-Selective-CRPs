# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:52:12 2024

@author: Mieszko Ferens

Script to run simulation for evaluating uniqueness of Binary-coded with
Padding (BP) CRP subsets.
"""

import argparse
import pandas as pd
from pathlib import Path

import numpy as np
from pypuf.simulation import XORArbiterPUF

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
    parser.add_argument("--n-PUFs", type=int, default=20,
                        help="Number of PUFs to evaluate for uniqueness")
    parser.add_argument("--n-bits", type=int, default=64,
                        help="Challenge length in bits.")
    parser.add_argument("--k", type=int, default=1,
                        help="The number of parallel APUF in the XOR PUF.")
    parser.add_argument("--n-CRPs", type=int, default=2**16,
                        help="Number of CRPs to be generated.")
    args = parser.parse_args()
    
    # Generate the PUF
    pufs = []
    for i in range(args.n_PUFs):
        pufs.append(XORArbiterPUF(args.n_bits, args.k, args.seed + i))
    
    # Generate the challenges
    challenges = create_binary_code_challenges(n=args.n_bits, N=args.n_CRPs)
    
    # Get responses
    responses = []
    for i in range(args.n_PUFs):
        responses.append((pufs[i].eval(challenges) + 1) / 2)
    
    # Calculate uniqueness of responses
    distances = []
    for i in range(args.n_PUFs):
        for j in range(i, args.n_PUFs):
            if(i == j):
                continue
            distances.append(np.count_nonzero(responses[i] != responses[j]))
    uniq_mean = (sum(distances)/len(distances)) / args.n_CRPs
    uniq_std = np.sqrt(
        sum([((x/args.n_CRPs)-uniq_mean)**2 for x in distances])/len(distances))
    
    # Log data into csv format
    data = pd.DataFrame({"seed": [args.seed],
                         "n_PUFs": [args.n_PUFs],
                         "n_bits": [args.n_bits],
                         "k": [args.k],
                         "n_CRPs": [args.n_CRPs],
                         "distances": [distances],
                         "uniq_mean": [uniq_mean],
                         "uniq_std": [uniq_std]})
    filepath = Path(args.outdir + "out_uniq_regular_pattern_" + str(args.k) +
                    "XOR.csv")
    if(filepath.is_file()):
        data.to_csv(filepath, header=False, index=False, mode='a')
    else:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath, header=True, index=False, mode='a')
    

if(__name__ == "__main__"):
    main()
