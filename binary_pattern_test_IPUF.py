# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:44:04 2023

@author: Mieszko Ferens

Script to run an experiment for modelling an (1,k)-Interpose PUF that uses
shifted binary patterns as CRPs (BSP CRPs) during authentication with the
server.
"""

import argparse
import pandas as pd
from pathlib import Path

import numpy as np
from pypuf.simulation import InterposePUF
import pypuf.attack

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

def create_upper_layer_train_set(n, model_down, challenges, responses):
    
    # Create lower layer challenge sets with both possible interpose bit values
    challenges_neg = np.insert(challenges, n//2, -1, axis=1)
    challenges_pos = np.insert(challenges, n//2, +1, axis=1)
    
    # Predict responses to both challenge sets
    pred_y_neg = np.array(
        model_down._model.eval(challenges_neg).flatten(), dtype=np.int8)
    pred_y_pos = np.array(
        model_down._model.eval(challenges_pos).flatten(), dtype=np.int8)
    
    # Get the challenges where the interpose bit affects the final output
    indices = np.where(pred_y_neg - pred_y_pos != 0)
    
    # Get the correct interpose bit for those challenges
    interpose_bits = np.array([], dtype=np.int8)
    for index, out in enumerate(responses[indices]):
        if(out == pred_y_neg[indices][index]):
            interpose_bits = np.append(
                interpose_bits, -1)
        else:
            interpose_bits = np.append(
                interpose_bits, +1)
    
    return interpose_bits, indices

# Log data into csv format
def log_data_to_csv(args, iterations, accuracy):
    data = pd.DataFrame({"seed": [args.seed],
                         "n_bits": [args.n_bits],
                         "k_down": [args.k_down],
                         "pattern_len": [args.pattern_len],
                         "train_data": [args.train_data],
                         "test_data": [args.test_data],
                         "threshold": [args.threshold],
                         "iterations": [iterations],
                         "ML_algorithm": [args.ML_algorithm],
                         "accuracy": [accuracy]})
    filepath = Path(args.outdir + "out_binary_pattern_1" + str(args.k_down) +
                    "IPUF.csv")
    if(filepath.is_file()):
        data.to_csv(filepath, header=False, index=False, mode='a')
    else:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath, header=True, index=False, mode='a')

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
    parser.add_argument("--k-down", type=int, default=1,
                        help="Number of parallel APUFs in the XOR PUF of the" +
                        " lower layer of the IPUF.")
    parser.add_argument("--n-CRPs", type=int, default=40000,
                        help="Minimum number of CRPs to be created.")
    parser.add_argument("--pattern-len", type=int, default=25,
                        help="Binary coded pattern length in bits.")
    parser.add_argument("--train-data", type=int, default=30000,
                        help="Number of training data samples for the model.")
    parser.add_argument("--test-data", type=int, default=10000,
                        help="Number of testing data samples for the model.")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Max number of splitting attack algorithm" +
                        "iterations.")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Accuracy threshold to obtain.")
    parser.add_argument("--ML-algorithm", type=str, default="LR",
                        help="ML algorithm to model the lower layer of the " +
                        "IPUF with (LR or MLP).")
    args = parser.parse_args()
    
    # Split the data into training and testing
    N = args.train_data + args.test_data
    train = args.train_data
    test = train + args.test_data
    
    # Generate the PUF
    k_up = 1
    puf = InterposePUF(
        args.n_bits, args.k_down, k_up, interpose_pos=33, seed=args.seed)
    
    # Generate the challenges
    challenges_up = create_binary_code_challenges(
        n=args.n_bits, N=args.n_CRPs, pattern_len=args.pattern_len)
    
    # Initially assume random values for the interpose bits
    interpose_bits = np.random.randint(2, size=N)
    challenges_down = np.insert(
        challenges_up, args.n_bits//2, interpose_bits, axis=1)
    
    # Get responses
    responses = puf.eval(challenges_up)
    
    count = 0
    while(1):
        # Prepare the data for training of lower layer model
        train_crps_down = ChallengeResponseSet(
            args.n_bits+1, np.array(challenges_down[:train], dtype=np.int8),
            np.array(responses[:train], dtype=np.float64))
        
        if(args.ML_algorithm == "LR"): # Use LR as a predictor
            model_down = pypuf.attack.LRAttack2021(
                train_crps_down, seed=args.seed, k=args.k_down, epochs=100,
                lr=.001, bs=1000, stop_validation_accuracy=.97)
        elif(args.ML_algorithm == "MLP"): # Use MLP as a predictor
            if(args.k_down <= 4): # If the XOR PUF is small don't reduce the NN too much
                network = [8, 16, 8] 
            else: # As defined in the literature: [2^(k-1), 2^k, 2^(k-1)]
                network = [2**(args.k_down-1), 2**args.k_down, 2**(args.k_down-1)]
            model_down = pypuf.attack.MLPAttack2021(
                train_crps_down, seed=args.seed, net=network, epochs=30,
                lr=.001, bs=1000, early_stop=.08)
        else:
            raise NotImplementedError("Only LR and MLP are supported.")
        
        # Train the model for the lower layer
        model_down.fit()
        
        # Test the lower layer model
        pred_y = model_down._model.eval(challenges_down[:train])
        pred_y = pred_y.reshape(len(pred_y), 1)
        pred_y = np.array(pred_y.flatten(), dtype=np.int8) # Convert array
        
        # Flip interpose bits where the final output didn't match
        interpose_bits, indices = create_upper_layer_train_set(
            args.n_bits, model_down, challenges_up[:train], responses[:train])
        
        # Create training set for upper layer
        train_crps_up = ChallengeResponseSet(
            args.n_bits, np.array(challenges_up[indices], dtype=np.int8),
            np.array(interpose_bits, dtype=np.float64))
        
        # Use an MLP as a predictor for upper layer
        model_up = pypuf.attack.LRAttack2021(
            train_crps_up, args.seed, k=k_up, epochs=30, lr=.001, bs=1000,
            stop_validation_accuracy=.97)
        
        # Train the model for the upper layer
        model_up.fit()
        
        # Test the final model
        interpose_bits = np.array(
            model_up._model.eval(
                challenges_up[train:test]).reshape(args.test_data, 1),
            dtype=np.int8).flatten()
        challenges_down[train:test] = np.insert(
            challenges_up[train:test], args.n_bits//2, interpose_bits, axis=1)
        pred_y = model_down._model.eval(challenges_down[train:test])
        pred_y = pred_y.reshape(len(pred_y), 1)
    
        # Calculate accuracy
        test_y = np.expand_dims(0.5 - 0.5*responses[train:test], axis=1)
        accuracy = np.count_nonzero(((pred_y<0.5) + test_y)-1)/len(test_y)
        print("---\n" +
              "Accuracy in the testing data: " + str(accuracy*100) + "%")
        
        count += 1
        if(accuracy > args.threshold or count >= args.iterations):
            print("Ending after " + str(count) + " iterations")
            log_data_to_csv(args, count, accuracy)
            break
        
        # Predict interpose bits
        interpose_bits = np.array(
            model_up._model.eval(challenges_up[:train]), dtype=np.int8).flatten()
        
        # Update training set for lower layer model
        challenges_down[:train] = np.insert(
            challenges_up[:train], args.n_bits//2, interpose_bits, axis=1)

if(__name__ == "__main__"):
    main()

