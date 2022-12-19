# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 13:13:01 2022

@author: Mieszko Ferens
"""

import os
import subprocess
from datetime import date
from pathlib import Path

outdir = "./Results/" + str(date.today()) + "/"
i = 0
while(1):
    if(not os.path.exists(outdir + str(i) + "/")):
        outdir += str(i) + "/"
        break
    i += 1

Path(outdir).mkdir(parents=True, exist_ok=True)

# Random shifted patterns CRPs (pattern length test)
# script = "random_pattern_test_classic.py"
# seeds = range(20)
# pattern_len = range(13,65)
# # pattern_len = range(12,65)
# for i in seeds:
#     print("Seed " + str(i) + "/" + str(seeds[-1]))
#     for j in pattern_len:
#         print("- Pattern length " + str(j) + "/" + str(pattern_len[-1]) +
#               "...", end='')
#         output_file = outdir + "debug.txt"
#         command = ["python", script, "--outdir", outdir, "--pattern-len",
#                     str(j), "--seed", str(i)]
#         with open(output_file, 'w') as out:
#             subprocess.call(command, stdout=out, stderr=subprocess.STDOUT)
#         print("Done")

# Random shifted pattern CRPs (training CRPs test)
# script = "random_pattern_test_classic.py"
# seeds = range(20)
# train_data = range(100000, 250000, 20000)
# # train_data = range(10000, 101000, 10000)
# for i in seeds:
#     print("Seed " + str(i) + "/" + str(seeds[-1]))
#     count = 1
#     for j in train_data:
#         print("- Training data " + str(j) + " (" + str(count) + "/" +
#               str(len(train_data)) + ")...", end='')
#         output_file = outdir + "debug.txt"
#         command = ["python", script, "--outdir", outdir, "--train-data",
#                     str(j), "--seed", str(i)]
#         with open(output_file, 'w') as out:
#             subprocess.call(command, stdout=out, stderr=subprocess.STDOUT)
#         print("Done")
#         count += 1

# Traditional CRPs (fully random)
# script = "random_pattern_test_classic.py"
# seeds = range(20)
# train_data = range(100000, 250000, 20000)
# #train_data = range(10000, 101000, 10000)
# for i in seeds:
#     print("Seed " + str(i) + "/" + str(seeds[-1]))
#     count = 1
#     for j in train_data:
#         print("- Training data " + str(j) + " (" + str(count) + "/" +
#               str(len(train_data)) + ")...", end='')
#         output_file = outdir + "debug.txt"
#         command = ["python", script, "--outdir", outdir, "--train-data",
#                     str(j), "--seed", str(i), "--pattern-len", "64",
#                     "--n-patterns", "320000"]
#         with open(output_file, 'w') as out:
#             subprocess.call(command, stdout=out, stderr=subprocess.STDOUT)
#         print("Done")
#         count += 1

# Regular pattern CRPs (Binary code with padding)
# script = "regular_pattern_test_classic.py"
# seeds = range(20)
# train_data = range(100000, 250000, 20000)
# # train_data = range(10000, 101000, 10000)
# for i in seeds:
#     print("Seed " + str(i) + "/" + str(seeds[-1]))
#     count = 1
#     for j in train_data:
#         print("- Training data " + str(j) + " (" + str(count) + "/" +
#               str(len(train_data)) + ")...", end='')
#         output_file = outdir + "debug.txt"
#         command = ["python", script, "--outdir", outdir, "--train-data",
#                     str(j), "--seed", str(i)]
#         with open(output_file, 'w') as out:
#             subprocess.call(command, stdout=out, stderr=subprocess.STDOUT)
#         print("Done")
#         count += 1

# Binary coded shifted patterns CRPs (pattern length test)
script = "binary_pattern_test_classic.py"
seeds = range(20)
pattern_len = range(63,64)
for i in seeds:
    print("Seed " + str(i) + "/" + str(seeds[-1]))
    for j in pattern_len:
        print("- Pattern length " + str(j) + "/" + str(pattern_len[-1]) +
              "...", end='')
        output_file = outdir + "debug.txt"
        command = ["python", script, "--outdir", outdir, "--pattern-len",
                    str(j), "--seed", str(i)]
        with open(output_file, 'w') as out:
            subprocess.call(command, stdout=out, stderr=subprocess.STDOUT)
        print("Done")

# Binary coded shifted pattern CRPs (training CRPs test)
# script = "binary_pattern_test_classic.py"
# seeds = range(20)
# train_data = range(100000, 250000, 20000)
# # train_data = range(10000, 101000, 10000)
# for i in seeds:
#     print("Seed " + str(i) + "/" + str(seeds[-1]))
#     count = 1
#     for j in train_data:
#         print("- Training data " + str(j) + " (" + str(count) + "/" +
#               str(len(train_data)) + ")...", end='')
#         output_file = outdir + "debug.txt"
#         command = ["python", script, "--outdir", outdir, "--train-data",
#                     str(j), "--seed", str(i)]
#         with open(output_file, 'w') as out:
#             subprocess.call(command, stdout=out, stderr=subprocess.STDOUT)
#         print("Done")
#         count += 1

