# pyoptDMD

Exploratory notebooks for understanding the BOP-DMD and methods for ensemble guidance.

Both the optDMD and BOP-DMD were previously only matlab implementations. The focus of this
repository was the implementation and testing of the python translations. With the 
debut of the optDMD/BOP-DMD methods on PyDMD, this is no longer necessary.

1) [optDMD](https://github.com/klapo/pyoptDMD/blob/main/examples/ex_optDMD.ipynb) 
   provides an optimized framework for solving the DMD regressions that may come from 
   unevenly spaced time snapshots. Additionally, the optDMD is less biased than 
   standard DMD algorithms.
2) [BOP-DMD](https://github.com/klapo/pyoptDMD/blob/main/examples/ex-BOP-DMD.ipynb) 
   takes advantage of this property and solves the DMD using statistical
   bagging (i.e., randomly selected ensembles) for constructing the DMD.

The advantage of the combined BOP-DMD is: (a) the additional ability to provide
uncertainty estimates in the DMD solutions, especially the uncertainty in the spatial
modes, (b) the ability to better represent the time dynamics for more complex systems such
as those commonly found in geophysics, and (c) robustly solving the DMD for noisy data.

# Current status:

See PyDMD.

Additionally, the effect of noise on the BOP-DMD fits was explored and two attempts at 
ensemble guidance for the BOP-DMD were attempted.

The tutorials from PyDMD were additionally reproduced and explored. I am keeping a 
copy here for my own reference.

# Citations:

Askham, T., & Kutz, J. N. (2018). Variable projection methods for an optimized
dynamic mode decomposition. SIAM Journal on Applied Dynamical Systems, 17(1), 380–416.
https://doi.org/10.1137/M1124176

Sashidhar, D., & Kutz, J. N.
(2022). Bagging, optimized dynamic mode decomposition for robust, stable forecasting
with spatial and temporal uncertainty quantification. Philosophical Transactions of
the Royal Society A: Mathematical, Physical and Engineering Sciences, 380(2229).
https://doi.org/10.1098/rsta.2021.0199
