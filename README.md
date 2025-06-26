# EarnMore
```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
git clone https://github.com/microsoft/qlib.git && cd qlib
python setup.py install
cd ..
pip install -r requirements.txt
```

# RUN
```
1. make scripts
sh tools/pipeline_mask_sac_dj30_example.sh

2. make pipeline
python tools/pipeline.py

3. run pipeline
sh tools/pipeline.sh
```

# Instructions for Forked version (tailored for me)
```
# Create the necessary directories
mkdir -p datasets/mcad/raw
mkdir -p datasets/mcad/features
mkdir -p datasets/mcad/aux_stocks_files
# Copy raw data to the raw directory
# Make sure the file name is uppercase & the first column is "Date" and not "date"

# Create stocks.txt with your assets (uppercase)
echo -e "SPY\nQQQ\nGLD\nSHY\nTLT\nIEF\nDBC" > datasets/mcad/stocks.txt

# Create asset class files (IMPORTANT: avoid underscores in the category name)
echo -e "SPY\nQQQ" > datasets/mcad/aux_stocks_files/01_Equity.txt
echo -e "SHY\nTLT\nIEF" > datasets/mcad/aux_stocks_files/02_FixedIncome.txt
echo -e "GLD\nDBC" > datasets/mcad/aux_stocks_files/03_Commodity.txt

# run preprocess just as given in the EarnMore codebase
python tools/preprocess.py --dataset mcad

# Create pipeline scripts for different algorithms
sh tools/pipeline_mask_sac_mcad.sh
sh tools/pipeline_ppo_mcad.sh
sh tools/pipeline_ddpg_mcad.sh
sh tools/pipeline_mask_dqn_mcad.sh

# Generate the pipeline script
python tools/make_pipeline.py

# Run a specific algorithm
export PYTHONPATH=$PWD:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/mask_sac/mask_sac_*_mcad_*.py

# References

ElegantRL: https://github.com/AI4Finance-Foundation/ElegantRL

RL-Adventure: https://github.com/higgsfield/RL-Adventure

Qlib: https://github.com/microsoft/qlib

# Citing EarnMore

```bibtex
@inproceedings{zhang2024reinforcement,
    title={Reinforcement Learning with Maskable Stock Representation for Portfolio Management in Customizable Stock Pools}, 
    author={Wentao Zhang and Yilei Zhao and Shuo Sun and Jie Ying and Yonggang Xie and Zitao Song and Xinrun Wang and Bo An},
    booktitle={The Web Conference 2024},
    year={2024},
}
```