from src import *


# Original Setup Paper

## First Experiment With Shuffling Data
exp_config_random_baes = { 
    'split_type' : SPLIT_TYPE,
    'shuffle_loader' : True,
    'epochs' : EPOCHS,
    'criterion_type' : LOSS_TYPE,
    'batch_norm' : False,
    'hidden_sizes' : (1000, 500),
    'seed' : RANDOM_STATE,
    'emp_dims': EMB_DIMS,
    'learning_rate': LEARNING_RATE
}

exp_config_time_based = { 
    'split_type' : 'time',
    'shuffle_loader' : False,
    'epochs' : EPOCHS,
    'criterion_type' : LOSS_TYPE,
    'batch_norm' : False,
    'hidden_sizes' : (1000, 500),
    'seed' : RANDOM_STATE,
    'emp_dims': EMB_DIMS,
    'learning_rate': LEARNING_RATE
}

# Ablation 
## Change Embedding SIZE

EMB_DIMS_NEW = {
"Store": 20,
"DayOfWeek": 12,
"day": 20,
"month": 12,
"year": 4,
"Promo": 2,
"State": 12,
}

exp_config_embedding = { 
    'split_type' : SPLIT_TYPE,
    'shuffle_loader' : True,
    'epochs' : EPOCHS,
    'criterion_type' : LOSS_TYPE,
    'batch_norm' : False,
    'hidden_sizes' : (1000, 500),
    'seed' : RANDOM_STATE,
    'emp_dims': EMB_DIMS_NEW,
    'learning_rate': LEARNING_RATE
}

## Batch Normaliztion
exp_config_bn= { 
    'split_type' : SPLIT_TYPE,
    'shuffle_loader' : True,
    'epochs' : EPOCHS,
    'criterion_type' : LOSS_TYPE,
    'batch_norm' : True,
    'hidden_sizes' : (1000, 500),
    'seed' : RANDOM_STATE,
    'emp_dims': EMB_DIMS,
    'learning_rate': LEARNING_RATE
}

## Batch Learning Rate
exp_config_dec_lr= { 
    'split_type' : SPLIT_TYPE,
    'shuffle_loader' : True,
    'epochs' : EPOCHS,
    'criterion_type' : LOSS_TYPE,
    'batch_norm' : False,
    'hidden_sizes' : (1000, 500),
    'seed' : RANDOM_STATE,
    'emp_dims': EMB_DIMS,
    'learning_rate': 0.0005
}

exp_config_inc_lr= { 
    'split_type' : SPLIT_TYPE,
    'shuffle_loader' : True,
    'epochs' : EPOCHS,
    'criterion_type' : LOSS_TYPE,
    'batch_norm' : False,
    'hidden_sizes' : (1000, 500),
    'seed' : RANDOM_STATE,
    'emp_dims': EMB_DIMS,
    'learning_rate': 0.005
}


all_setups = {
    'Original_Model_Random_Based': exp_config_random_baes,
    'Original_Model_Time_Based': exp_config_time_based,
    'Ablation_With_Embedding_Dim': exp_config_embedding,
    'Ablation_With_Batch_Normalization': exp_config_bn,
    'Ablation_With_Learning_Rate_Increasing': exp_config_inc_lr,
    'Ablation_With_Learning_Rate_Decreasing': exp_config_dec_lr
}
