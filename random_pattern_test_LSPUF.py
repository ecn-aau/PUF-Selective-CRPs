# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:23:58 2024

@author: Mieszko Ferens

Script to run an experiment for modelling a Lightweight Secure PUF that uses
shifted random patterns as CRPs (RSP or traditional CRPs) during authentication
with the server.
"""

import argparse
import pandas as pd
from pathlib import Path

import numpy as np
from pypuf.simulation import LightweightSecurePUF
from pypuf.io import random_inputs
import pypuf.attack

class ChallengeResponseSet():
    def __init__(self, n, challenges, responses):
        self.challenge_length = n
        self.challenges = challenges
        self.responses = np.expand_dims(
            np.expand_dims(responses,axis=1),axis=1)

def create_shifted_pattern_challenges(n, N, n_patterns, pattern_len, seed=0):
    
    assert n_patterns <= 2**pattern_len, (
        "The number of random patterns to be generated exceeds the possible " +
        "total. Tip: The max is 2^(pattern length)")
    
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
    parser.add_argument("--n-bits", type=int, default=64,
                        help="Challenge length in bits.")
    parser.add_argument("--k", type=int, default=2,
                        help="The number of parallel APUF in the LSPUF.")
    parser.add_argument("--n-patterns", type=int, default=2**16,
                        help="Number of random patterns to be created.")
    parser.add_argument("--pattern-len", type=int, default=16,
                        help="Random pattern length in bits.")
    parser.add_argument("--train-data", type=int, default=100000,
                        help="Number of training data samples for the model.")
    parser.add_argument("--test-data", type=int, default=10000,
                        help="Number of testing data samples for the model.")
    args = parser.parse_args()
    
    # Generate the PUF
    puf = LightweightSecurePUF(args.n_bits, args.k, args.seed)
    
    # Generate the challenges
    challenges = create_shifted_pattern_challenges(
        n=args.n_bits, N=(args.train_data + args.test_data),
        n_patterns=args.n_patterns, pattern_len=args.pattern_len,
        seed=args.seed)
    
    # Get responses
    responses = puf.eval(challenges)
    
    # Split the data into training and testing
    train = args.train_data
    test = train + args.test_data
    
    # Prepare the data for training and testing
    train_crps = ChallengeResponseSet(
        args.n_bits, np.array(challenges[:train], dtype=np.int8),
        np.array(responses[:train], dtype=np.float64))
    test_x = challenges[train:test]
    test_y = np.expand_dims(0.5 - 0.5*responses[train:test], axis=1)
    
    # Use MLP as a predictor
    if(args.k <= 4): # If the LSPUF is small don't reduce the NN too much
        network = [8, 16, 8] 
    else: # As defined in the literature: [2^(k-1), 2^k, 2^(k-1)]
        network = [2**(args.k-1), 2**args.k, 2**(args.k-1)]
    model = pypuf.attack.MLPAttack2021(
        train_crps, seed=args.seed, net=network, epochs=30, lr=.001,
        bs=1000, early_stop=.08)
    
    # Train the model
    model.fit()

    # Test the model
    pred_y = model._model.eval(test_x)
    pred_y = pred_y.reshape(len(pred_y), 1)
    
    # Calculate accuracy
    accuracy = np.count_nonzero(((pred_y<0.5) + test_y)-1)/len(test_y)
    print("---\n" +
          "Accuracy in the testing data: " + str(accuracy*100) + "%")
    
    # Log data into csv format
    data = pd.DataFrame({"seed": [args.seed],
                         "n_bits": [args.n_bits],
                         "k": [args.k],
                         "n_patterns": [args.n_patterns],
                         "pattern_len": [args.pattern_len],
                         "train_data": [args.train_data],
                         "test_data": [args.test_data],
                         "ML_algorithm": ["MLP"],
                         "accuracy": [accuracy]})
    filepath = Path(args.outdir + "out_random_pattern_" + str(args.k) +
                    "LS.csv")
    if(filepath.is_file()):
        data.to_csv(filepath, header=False, index=False, mode='a')
    else:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath, header=True, index=False, mode='a')
    

if(__name__ == "__main__"):
    main()

