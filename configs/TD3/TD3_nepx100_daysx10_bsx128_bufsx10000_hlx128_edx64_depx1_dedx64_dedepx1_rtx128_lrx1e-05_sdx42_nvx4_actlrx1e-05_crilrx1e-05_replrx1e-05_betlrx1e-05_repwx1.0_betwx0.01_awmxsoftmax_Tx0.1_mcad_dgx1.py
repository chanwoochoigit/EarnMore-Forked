root = None
workdir = 'workdir'
tag = 'TD3_nepx100_daysx10_bsx128_bufsx10000_hlx128_edx64_depx1_dedx64_dedepx1_rtx128_lrx1e-05_sdx42_nvx4_actlrx1e-05_crilrx1e-05_replrx1e-05_betlrx1e-05_repwx1.0_betwx0.01_awmxsoftmax_Tx0.1_mcad_dgx1'
num_stocks = 7
num_envs = 4
discretized_low = 0
discretized_high = 10
num_features = 102
temporal_dim = 3
train_start_date = '2007-02-06'
val_start_date = '2019-10-25'
test_start_date = '2022-07-19'
test_end_date = '2025-04-14'
if_use_per = False
if_norm = True
if_norm_temporal = False
save_freq = 20
num_episodes = 100
days = 10
batch_size = 128
buffer_size = 10000
horizon_len = 128
embed_dim = 64
depth = 1
lr = 1e-05
explore_noise_std = 0.05
policy_noise_std = 0.1
seed = 42
transition = ['state', 'action', 'reward', 'done', 'next_state']
transition_shape = dict(
    state=dict(shape=(4, 7, 10, 102), type='float32'),
    action=dict(shape=(4, 8), type='float32'),
    reward=dict(shape=(4, ), type='float32'),
    done=dict(shape=(4, ), type='float32'),
    next_state=dict(shape=(4, 7, 10, 102), type='float32'))
dataset = dict(
    type='PortfolioManagementDataset',
    root=None,
    data_path='datasets/mcad/features',
    stocks_path='datasets/mcad/stocks.txt',
    features_name=[
        'open', 'high', 'low', 'close', 'kmid2', 'kup2', 'klow', 'klow2',
        'ksft2', 'roc_5', 'roc_10', 'roc_20', 'roc_30', 'roc_60', 'ma_5',
        'ma_10', 'ma_20', 'ma_30', 'ma_60', 'std_5', 'std_10', 'std_20',
        'std_30', 'std_60', 'beta_5', 'beta_10', 'beta_20', 'beta_30',
        'beta_60', 'max_5', 'max_10', 'max_20', 'max_30', 'max_60', 'min_5',
        'min_10', 'min_20', 'min_30', 'min_60', 'qtlu_5', 'qtlu_10', 'qtlu_20',
        'qtlu_30', 'qtlu_60', 'qtld_5', 'qtld_10', 'qtld_20', 'qtld_30',
        'qtld_60', 'rank_5', 'rank_10', 'rank_20', 'rank_30', 'rank_60',
        'imax_5', 'imax_10', 'imax_20', 'imax_30', 'imax_60', 'imin_5',
        'imin_10', 'imin_20', 'imin_30', 'imin_60', 'imxd_5', 'imxd_10',
        'imxd_20', 'imxd_30', 'imxd_60', 'cntp_5', 'cntp_10', 'cntp_20',
        'cntp_30', 'cntp_60', 'cntn_5', 'cntn_10', 'cntn_20', 'cntn_30',
        'cntn_60', 'cntd_5', 'cntd_10', 'cntd_20', 'cntd_30', 'cntd_60',
        'sump_5', 'sump_10', 'sump_20', 'sump_30', 'sump_60', 'sumn_5',
        'sumn_10', 'sumn_20', 'sumn_30', 'sumn_60', 'sumd_5', 'sumd_10',
        'sumd_20', 'sumd_30', 'sumd_60'
    ],
    temporals_name=['weekday', 'day', 'month'],
    labels_name=['ret1', 'mov1'])
environment = dict(
    type='Environment',
    dataset=None,
    mode='train',
    if_norm=True,
    if_norm_temporal=False,
    scaler=None,
    days=10,
    start_date=None,
    end_date=None,
    initial_amount=1000.0,
    transaction_cost_pct=0.001)
act_net = dict(
    type='ActorTD3',
    embed_type='TimesEmbed',
    input_dim=102,
    temporal_dim=3,
    in_chans=1,
    embed_dim=64,
    depth=1,
    cls_embed=True,
    explore_noise_std=0.05)
cri_net = dict(
    type='CriticTD3',
    embed_type='TimesEmbed',
    input_dim=102,
    temporal_dim=3,
    in_chans=1,
    embed_dim=64,
    depth=1,
    cls_embed=True)
criterion = dict(type='MSELoss', reduction='none')
optimizer = dict(type='AdamW', params=None, lr=1e-05)
agent = dict(
    type='AgentTD3',
    act_net=dict(
        type='ActorTD3',
        embed_type='TimesEmbed',
        input_dim=102,
        temporal_dim=3,
        in_chans=1,
        embed_dim=64,
        depth=1,
        cls_embed=True,
        explore_noise_std=0.05),
    cri_net=dict(
        type='CriticTD3',
        embed_type='TimesEmbed',
        input_dim=102,
        temporal_dim=3,
        in_chans=1,
        embed_dim=64,
        depth=1,
        cls_embed=True),
    criterion=dict(type='MSELoss', reduction='none'),
    optimizer=dict(type='AdamW', params=None, lr=1e-05),
    if_use_per=False,
    num_envs=4,
    transition_shape=dict(
        state=dict(shape=(4, 7, 10, 102), type='float32'),
        action=dict(shape=(4, 8), type='float32'),
        reward=dict(shape=(4, ), type='float32'),
        done=dict(shape=(4, ), type='float32'),
        next_state=dict(shape=(4, 7, 10, 102), type='float32')),
    max_step=10000.0,
    gamma=0.99,
    reward_scale=1,
    repeat_times=128,
    batch_size=128,
    clip_grad_norm=3.0,
    soft_update_tau=0,
    state_value_tau=0.005,
    explore_noise_std=0.05,
    policy_noise_std=0.1,
    device=None)
decoder_embed_dim = 64
decoder_depth = 1
repeat_times = 128
act_lr = 1e-05
cri_lr = 1e-05
rep_lr = 1e-05
beta_lr = 1e-05
rep_loss_weight = 1.0
beta_loss_weight = 0.01
action_wrapper_method = 'softmax'
T = 0.1
data_path = 'datasets/mcad/features'
stocks_path = 'datasets/mcad/stocks.txt'
aux_stocks_path = 'datasets/mcad/aux_stocks_files'
pred_num_stocks = 7
n_steps_per_episode = 1280
feature_size = (10, 102)
patch_size = (10, 102)
