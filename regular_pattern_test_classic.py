# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 09:58:06 2022

@author: Mieszko Ferens

Script to run an experiment for modelling an Arbiter PUF that uses regular
patterns (BP CRPs) as CRPs during authentication with the server.
"""

import argparse
import pandas as pd
from pathlib import Path

import numpy as np
from pypuf.simulation import XORArbiterPUF
import pypuf.attack

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
        "Not enough CRPs have been generated. The limit is (2^18 - 3) CRPs.")
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
    parser.add_argument("--train-data", type=int, default=80000,
                        help="Number of training data samples for the model.")
    parser.add_argument("--test-data", type=int, default=10000,
                        help="Number of testing data samples for the model.")
    parser.add_argument("--ML-algorithm", type=str, default="LR",
                        help="ML algorithm to model the PUF with (LR or MLP).")
    args = parser.parse_args()
    
    # Generate the PUF
    puf = XORArbiterPUF(args.n_bits, args.k, args.seed)
    
    # Generate the challenges
    challenges = create_binary_code_challenges(
        n=args.n_bits, N=(args.train_data + args.test_data))
    
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

    if(args.ML_algorithm == "LR"): # Use LR as a predictor
        model = pypuf.attack.LRAttack2021(
            train_crps, seed=args.seed, k=args.k, epochs=100, lr=.001, bs=1000,
            stop_validation_accuracy=.97)
    elif(args.ML_algorithm == "MLP"): # Use MLP as a predictor
        if(args.k <= 4): # If the XOR PUF is small don't reduce the NN too much
            network = [8, 16, 8] 
        else: # As defined in the literature: [2^(k-1), 2^k, 2^(k-1)]
            network = [2**(args.k-1), 2**args.k, 2**(args.k-1)]
        model = pypuf.attack.MLPAttack2021(
            train_crps, seed=args.seed, net=network, epochs=30, lr=.001,
            bs=1000, early_stop=.08)
    else:
        raise NotImplementedError("Only LR and MLP are supported.")
    
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
                         "train_data": [args.train_data],
                         "test_data": [args.test_data],
                         "ML_algorithm": [args.ML_algorithm],
                         "accuracy": [accuracy]})
    filepath = Path(args.outdir + "out_regular_pattern_" + str(args.k) +
                    "XOR.csv")
    if(filepath.is_file()):
        data.to_csv(filepath, header=False, index=False, mode='a')
    else:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath, header=True, index=False, mode='a')
    

if(__name__ == "__main__"):
    main()

