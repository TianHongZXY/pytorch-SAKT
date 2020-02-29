class DefaultConfig(object):
    model = 'SAKT'
    train_data = ""  # train_data_path
    test_data = ""
    batch_size = 256
    state_size = 200
    num_heads = 5
    max_len = 50
    dropout = 0.1
    max_epoch = 10
    lr = 3e-3
    lr_decay = 0.9
    max_grad_norm = 1.0
    weight_decay = 0  # l2正则化因子

opt = DefaultConfig()