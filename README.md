# PUF-Selective-CRPs

Code to test Improved Logistic Regression (LR) and Improved Multi-Layer Perceptron (MLP) modelling attacks
on 4-XOR and 5-XOR PUFs respectively.

## How to cite

M. Ferens, E. Dushku, and S. Kosta, "Securing PUFs Against ML Modeling Attacks via an Efficient Challenge-Response Approach," IEEE INFOCOM 2023 - IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS), Hoboken, NJ, USA, 2023, pp. 1-6, doi: 10.1109/INFOCOMWKSHPS57453.2023.10226062.

## Dependencies

The code was tested with the following:
- Python 3.8
- pypuf 3.2.1
- numpy 1.23.4
- pandas 1.5.1

## How to use

To run single experiments use one of the following (use `--help` for a list of arguments):
```
python3 binary_padded_test_classic.py
python3 random_pattern_test_classic.py
python3 binary_pattern_test_classic.py
```
For parametric simulations some examples are available with:
```
python3 main.py
```

## References

1. N. Wisiol, B. Thapaliya, K. T. Mursi, J. P. Seifert, and Y. Zhuang, "Neural network modeling attacks on Arbiter-PUF-based designs," in *IEEE Transactions on Information Forensics and Security*, vol. 17, pp. 2719-2731, 2022.
2. K. T. Mursi, B. Thapaliya, Y. Zhuang, A. O. Aseeri, and M. S. Alkatheiri, "A fast deep learning method for security vulnerability study of XOR PUFs," *Electronics*, vol. 9, no. 10, 2020. [Online]. Available: https://www.mdpi.com/2079-9292/9/10/1715
3. N. Wisiol, C. Gräbnitz, C. Mühl, B. Zengin, T. Soroceanu, N. Pirnay, K. T. Mursi, and A. Baliuka, "pypuf: Cryptanalysis of Physically Unclonable Functions," 2021, version v2. [Online]. Available: https://doi.org/10.5281/zenodo.3901410
