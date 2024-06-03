# PUF-Selective-CRPs

Code to test Machine Learning (ML) modelling attacks on Arbiter-based PUFs when Selective CRPs are used.

The supported PUF types are:
 - Arbiter PUF
 - k-XOR PUF
 - Interpose PUF (IPUF)

The supported ML algorithms are:
 - Logistic Regression (LR)
 - Multi-Layer Perceptron (MLP)
 - Splitting attack (using LR or MLP) for IPUF

## How to cite

M. Ferens, E. Dushku, and S. Kosta, "Securing PUFs Against ML Modeling Attacks via an Efficient Challenge-Response Approach," IEEE INFOCOM 2023 - IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS), Hoboken, NJ, USA, 2023, pp. 1-6, doi: 10.1109/INFOCOMWKSHPS57453.2023.10226062.

## Dependencies

The code was tested with the following:
- Python 3.8
- pypuf 3.2.1
- tensorflow 2.4.4
- numpy 1.23.4
- pandas 1.5.1

## How to use

Keywords:
 - "random": For Random Shifted Pattern (RSP) CRPs
 - "regular": For Binary-coded with Padding (BP) CRPs
 - "binary": For Binary Shifted Pattern (BSP) CRPs

For traditional CRPs use the scripts designed for RSP CRPs, and set the pattern length equal to the challenge length (e.g., `--n-bits 64 --pattern-len 64`).

To run single experiments for ML modelling attacks on an k-XOR PUF (`--k 1` is an Arbiter PUF) use one of the following (use `--help` for a list of arguments):
```
python3 random_pattern_test_classic.py
python3 regular_pattern_test_classic.py
python3 binary_pattern_test_classic.py
```
To do so for an IPUF use one of the following (use `--help` for a list of arguments):
```
python3 random_pattern_test_IPUF.py
python3 regular_pattern_test_IPUF.py
python3 binary_pattern_test_IPUF.py
```
To run single experiments for evaluating a distance metric on a k-XOR PUF (`--k 1` is an Arbiter PUF) use one of the following (use `--help` for a list of arguments):
```
python3 random_pattern_distance_test.py
python3 regular_pattern_distance_test.py
python3 binary_pattern_distance_test.py
```
This also calculates the uniformity of the PUF when the respective CRPs are used. However, is computationally costly. To obtain only the uniformity value much faster use on of the following (use `--help` for a list of arguments):
```
python3 random_pattern_unif_test.py
python3 regular_pattern_unif_test.py
python3 binary_pattern_unif_test.py
```
You may also evaluate the uniqueness or reliability of the PUF when the respective CRPs are used (use `--help` for a list of arguments):
```
python3 random_pattern_uniq_test.py
python3 regular_pattern_uniq_test.py
python3 binary_pattern_uniq_test.py
python3 random_pattern_rel_test.py
python3 regular_pattern_rel_test.py
python3 binary_pattern_rel_test.py
```
Note that reliability is calculated based on a "noisiness" factor defined as `--noise`. A realistic value according to [3] is `--noise 0.1`. The reliability simulation scripts are intended to evaluate the effect of Selective CRPs compared to traditional ones, not to evaluate the real-life reliability of any given PUF.

## References

1. N. Wisiol, B. Thapaliya, K. T. Mursi, J. P. Seifert, and Y. Zhuang, "Neural network modeling attacks on Arbiter-PUF-based designs," in *IEEE Transactions on Information Forensics and Security*, vol. 17, pp. 2719-2731, 2022.
2. K. T. Mursi, B. Thapaliya, Y. Zhuang, A. O. Aseeri, and M. S. Alkatheiri, "A fast deep learning method for security vulnerability study of XOR PUFs," *Electronics*, vol. 9, no. 10, 2020. [Online]. Available: https://www.mdpi.com/2079-9292/9/10/1715
3. N. Wisiol, C. Gräbnitz, C. Mühl, B. Zengin, T. Soroceanu, N. Pirnay, K. T. Mursi, and A. Baliuka, "pypuf: Cryptanalysis of Physically Unclonable Functions," 2021, version v2. [Online]. Available: https://doi.org/10.5281/zenodo.3901410
