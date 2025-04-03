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

# Create stocks.txt with your assets (uppercase)
echo -e "SPY\nQQQ\nGLD\nSHY\nTLT\nIEF\nDBC" > datasets/mcad/stocks.txt

# run preprocess just as given in the EarnMore codebase
python tools/preprocess.py

# Create a run script for multi-asset-class dataset
# mcad = Multi-Class Asset Dataset
cat > tools/pipeline_{algorithm}_mcad.sh << 'EOF'
python tools/make_scripts.py \
 --config configs/mask_sac_portfolio_management.py \
 --mask \
 --action_wrapper_method softmax \
 --num_episodes 2000 \
 --dataset mcad \
 --num_stocks 7 \
 --buffer_size 10000 \
 --repeat_times 128 \
 --gpu_id 0 \
 --days 10 \
 --lr 1e-5 \
 --act_lr 1e-5 \
 --cri_lr 1e-5 \
 --rep_lr 1e-5 \
 --beta_lr 1e-5 \
 --seed 42 \
 --T 0.1
EOF

chmod +x tools/pipeline_{algorithm}_mcad.sh

# Add export functionality to train.py
sed -i '1s/^/from pm.utils.export import export_allocation_history\n/' tools/train.py
sed -i '/def parse_args/a \    parser.add_argument("--export_allocations", action="store_true", help="Export allocation history")' tools/train.py
sed -i '/return agent, train_env, val_env, test_env/i \    if args.export_allocations:\n        export_path = os.path.join(cfg.workdir, cfg.tag, "allocation_history.csv")\n        export_allocation_history(agent, test_env, export_path)' tools/train.py

# Generate the training script
sh tools/pipeline_{algorithm}_mcad.sh

# Run the generated script with allocation export
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/ppo/ppo_portfolio_management.py --export_allocations
```

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