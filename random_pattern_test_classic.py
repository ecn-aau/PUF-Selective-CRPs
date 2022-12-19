# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 11:55:28 2022

@author: Mieszko Ferens

Script to run an experiment for modelling an Arbiter PUF that uses shifted
random patterns as CRPs during authentication with the server
"""

import argparse
import pandas as pd
from pathlib import Path

import numpy as np
from pypuf.simulation import XORArbiterPUF
from pypuf.io import random_inputs
import pypuf.attack

class ChallengeResponseSet():
    def __init__(self, n, challenges, responses):
        self.challenge_length = n
        self.challenges = challenges
        self.responses = np.expand_dims(
            np.expand_dims(responses,axis=1),axis=1)

def create_shifted_pattern_challenges(n, N, pattern_len, seed=0):
    
    patterns = random_inputs(pattern_len, N, seed=seed)

    challenges = -np.ones(((n-pattern_len+1)*N,n), dtype=np.int8)
    for i in range(N):
        for j in range(n-pattern_len+1):
            challenges[i*(n-pattern_len+1)+j, j:j+pattern_len] = patterns[i]
    
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
    parser.add_argument("--n-patterns", type=int, default=180000,
                        help="Number of random patterns to be created.")
    parser.add_argument("--pattern-len", type=int, default=16,
                        help="Random pattern length in bits.")
    parser.add_argument("--train-data", type=int, default=160000,
                        help="Number of training data samples for the model.")
    parser.add_argument("--test-data", type=int, default=10000,
                        help="Number of testing data samples for the model.")
    args = parser.parse_args()
    
    # Generate the PUF
    k = 5
    puf = XORArbiterPUF(args.n_bits, k, args.seed)
    
    # Generate the challenges
    challenges = create_shifted_pattern_challenges(
        n=args.n_bits, N=args.n_patterns, pattern_len=args.pattern_len,
        seed=args.seed)
    
    # Remove duplicate challenges
    _ , idx = np.unique(challenges, return_index=True, axis=0)
    challenges = challenges[np.sort(idx)]
    
    # Check if split the data into training and testing is possible
    assert args.train_data + args.test_data <= len(challenges), (
        "Not enough training data. Tip: Increase the number of patterns")
    train = args.train_data
    test = train + args.test_data
    
    # Get responses
    responses = puf.eval(challenges)
    
    # Prepare the data for training and testing
    train_crps = ChallengeResponseSet(
        args.n_bits, np.array(challenges[:train], dtype=np.int8),
        np.array(responses[:train], dtype=np.float64))
    test_x = challenges[train:test]
    test_y = np.expand_dims(0.5 - 0.5*responses[train:test], axis=1)
    
    # Use an MLP as a predictor
    network = [16, 32, 16] # As defined in the literature: [2^(k-1), 2^k, 2^(k-1)]
    model = pypuf.attack.MLPAttack2021(
        train_crps, seed=args.seed, net=network, epochs=30, lr=.001, bs=1000,
        early_stop=.08)
    # model = pypuf.attack.LRAttack2021(
    #     train_crps, seed=args.seed, k=k, epochs=100, lr=.001, bs=1000,
    #     stop_validation_accuracy=.97)
    
    # Train the model
    model.fit()

    # Test the model
    pred_y = model._model.eval(test_x)
    
    # pred_y = pred_y.reshape(len(pred_y), 1) # TODO (for LR output only)
    
    # Calculate accuracy
    accuracy = np.count_nonzero(((pred_y<0.5) + test_y)-1)/len(test_y)
    print("---\n" +
          "Accuracy in the testing data: " + str(accuracy*100) + "%")
    
    # Log data into csv format
    filepath = Path(args.outdir + "out_random.csv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame({"seed": [args.seed],
                         "pattern_len": [args.pattern_len],
                         "train_data": [args.train_data],
                         "accuracy": [accuracy]})
    data.to_csv(filepath, header=False, index=False, mode='a')
    

if(__name__ == "__main__"):
    main()

