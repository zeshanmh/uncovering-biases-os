# Uncovering Bias Mechanisms in Observational Studies

This repo contains code to recreate the results in the paper: "Uncovering Bias Mechanisms in Observational Studies via Predictive Performance". 

## Setup 
1. Clone the repository:\
  ```git@github.com:clinicalml/benchmarking-os.git```\
  ```cd benchmarking-os```
2. Create and activate the conda environment:\
  ```conda env create -f environment.yml```\
  ```conda activate benchmarking-os```
4. Run the tests in ```whi/``` to verify the installation:\
  ```pytest test_main.py```\
  ```pytest test_replication.py```


## Synthetic Experiments
To run a synthetic experiment, run the following command:\
```python main.py --bias_S --bias_Y1 --bias_type "selection_bias" --d 5 6 7 8 --n_rct 2000 --n_val 2000```

To plot a figure similar to the ones in the paper, run:\
```python plot.py --bias_type "selection_bias" --d 5 6 7 8 --n_rct 2000 --n_val 2000```

Ensure that the ```bias_type``` flag as well as the ```n_rct``` and ```n_val``` values are the same in both commands. 

## WHI Experiments
Run the following to conduct a WHI experiment:\
```python main.py --selection_flag biased --censored --outcome_name CHD --model_type LR```

Adjust plot-whi.r accordingly based on the saved results files to generate the final plot.
