# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:59:01 2024

@author: Mieszko Ferens

Script to run simulation for evaluating reliability of Binary Shifted Pattern
(BSP) CRP subsets.
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

def create_binary_code_challenges(n, N, pattern_len):
    
    assert ((2**(pattern_len))*(n-pattern_len+1) > N), (
        "Pattern length is too low for the number of patterns")
    
    max_bits = int(np.ceil(np.log2(N)))
    bits = min(max_bits, pattern_len)
    
    lsb = np.arange(2**bits, dtype=np.uint8).reshape(-1,1)
    
    extra = 0
    if(bits % 8):
        extra = 1
    
    msb = []
    for i in range(1, int(bits/8) + extra):
        msb.append(lsb.copy())
        for j in range(2**8):
            msb[i-1][(2**(8*(i+1)))*j:(2**(8*(i+1)))*(j+1)].sort(axis=0)

    lsb = np.unpackbits(lsb, axis=1)[:,-8:].copy()
    for i in range(len(msb)):
        msb[i] = np.unpackbits(msb[i], axis=1)[:,-8:].copy()

    msb.insert(0, lsb)
    patterns = 2*np.concatenate(msb[::-1], axis=1, dtype=np.int8) - 1
    if(bits % 8):
        patterns = np.delete(
            patterns[:N], slice(8 - (bits % 8)), axis=1)
    else:
        patterns = patterns[:N]
    
    if(bits < pattern_len):
        patterns = np.concatenate(
            (-np.ones((N, pattern_len - bits), dtype=np.int8), patterns),
            axis=1, dtype=np.int8)
    
    challenges = -np.ones(((n-pattern_len+1)*len(patterns), n), dtype=np.int8)
    for i in range(len(patterns)):
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
    parser.add_argument("--n-bits", type=int, default=64,
                        help="Challenge length in bits.")
    parser.add_argument("--k", type=int, default=1,
                        help="The number of parallel APUF in the XOR PUF.")
    parser.add_argument("--n-CRPs", type=int, default=100000,
                        help="Minimum number of CRPs to be created.")
    parser.add_argument("--n-evals", type=int, default=10,
                        help="Number of repeated noisy measurements.")
    parser.add_argument("--pattern-len", type=int, default=25,
                        help="Binary coded pattern length in bits.")
    parser.add_argument("--noise", type=float, default=0.1,
                        help="Noise factor added to PUF parameters.")
    args = parser.parse_args()
    
    # Generate the PUF
    puf0 = XORArbiterPUF(args.n_bits, args.k, args.seed)
    pufN = XORArbiterPUF(args.n_bits, args.k, args.seed, noisiness=args.noise)
    
    # Generate the challenge
    challenges = create_binary_code_challenges(
        n=args.n_bits, N=args.n_CRPs, pattern_len=args.pattern_len)
    
    # Get responses
    responses = puf0.eval(challenges)
    
    # Calculate reliability of responses
    errors = []
    for i in range(args.n_evals):
        errors.append(np.count_nonzero(responses - pufN.eval(challenges)))
    reliability = sum(errors)/(args.n_CRPs*args.n_evals)
    
    # Log data into csv format
    data = pd.DataFrame({"seed": [args.seed],
                         "n_bits": [args.n_bits],
                         "k": [args.k],
                         "n_CRPs": [args.n_CRPs],
                         "pattern_len": [args.pattern_len],
                         "noise": [args.noise],
                         "reliability": [reliability]})
    filepath = Path(args.outdir + "out_rel_binary_pattern_" + str(args.k) +
                    "XOR.csv")
    if(filepath.is_file()):
        data.to_csv(filepath, header=False, index=False, mode='a')
    else:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath, header=True, index=False, mode='a')
    

if(__name__ == "__main__"):
    main()
