# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:38:50 2024

@author: Mieszko Ferens

Script to run simulation for evaluating uniqueness of traditional and
Random Shifted Pattern (RSP) CRP subsets.
"""

import argparse
import pandas as pd
from pathlib import Path

import numpy as np
from pypuf.simulation import XORArbiterPUF
from pypuf.io import random_inputs

class ChallengeResponseSet():
    def __init__(self, n, challenges, responses):
        self.challenge_length = n
        self.challenges = challenges
        self.responses = np.expand_dims(
            np.expand_dims(responses,axis=1),axis=1)

def create_shifted_pattern_challenges(n, N, n_patterns, pattern_len, seed=0):
    
    patterns = random_inputs(pattern_len, n_patterns, seed=seed)

    challenges = -np.ones(((n-pattern_len+1)*n_patterns,n), dtype=np.int8)
    for i in range(n_patterns):
        for j in range(n-pattern_len+1):
            challenges[i*(n-pattern_len+1)+j, j:j+pattern_len] = patterns[i]
    
    _ , idx = np.unique(challenges, return_index=True, axis=0)
    challenges = challenges[np.sort(idx)]
    
    assert N <= len(challenges), (
        "Not enough unique CRPs exist due to duplicates. " +
        "Tip: You might need to increase the number of patterns")
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
    parser.add_argument("--n-CRPs", type=int, default=10000,
                        help="Number of CRPs to be generated.")
    parser.add_argument("--n-patterns", type=int, default=10000,
                        help="Number of random patterns to be created.")
    parser.add_argument("--pattern-len", type=int, default=64,
                        help="Random pattern length in bits.")
    args = parser.parse_args()
    
    assert args.n_CRPs <= args.n_patterns*(args.n_bits-args.pattern_len+1), (
        "Not enough patterns. Tip: Increase the number of patterns or " +
        "decrease the number of CRPs/pattern lenght.")
    
    # Generate the PUFs
    pufs = []
    for i in range(args.n_PUFs):
        pufs.append(XORArbiterPUF(args.n_bits, args.k, args.seed + i))
    
    # Generate the challenges
    challenges = create_shifted_pattern_challenges(
        n=args.n_bits, N=args.n_CRPs, n_patterns=args.n_patterns,
        pattern_len=args.pattern_len, seed=args.seed)
    
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
                         "n_patterns": [args.n_patterns],
                         "pattern_len": [args.pattern_len],
                         "distances": [distances],
                         "uniq_mean": [uniq_mean],
                         "uniq_std": [uniq_std]})
    filepath = Path(args.outdir + "out_uniq_random_pattern_" + str(args.k) +
                    "XOR.csv")
    if(filepath.is_file()):
        data.to_csv(filepath, header=False, index=False, mode='a')
    else:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath, header=True, index=False, mode='a')
    

if(__name__ == "__main__"):
    main()
