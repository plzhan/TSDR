import os
import copy
import argparse
import math
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW
import yaml
from data_loaders import (
    MostRecentQuestionSkillDataset,
    MostEarlyQuestionSkillDataset,
    CounterDatasetWrapper,
)
from torch.optim.lr_scheduler import LambdaLR
from models.dkt import DKT
from models.simplekt import simpleKT
from models.diskt import DisKT
from models.akt import AKT
from models.folibikt import folibiKT
from models.sparsekt import sparseKT
from models.drkt import DRKT
from train import model_train
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
from utils.config import ConfigNode as CN
from utils.file_io import PathManager
import logging
#from tabpfn_extensions import TabPFNClassifier
from tqdm import tqdm

SUPPORTED_MODELS = ("dkt", "akt", "simplekt", "folibikt", "sparsekt", "diskt")


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        # Warmup 阶段：线性增加
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Decay 阶段：余弦衰减
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)
    
def setup_logger(log_dir="../result/logs"):
    """设置日志记录器，同时输出到控制台和文件"""
    # 创建日志目录（如果不存在）
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 日志文件名包含当前时间，避免重复
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{current_time}.log")

    # 创建日志记录器
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)  # 设置日志级别

    # 避免重复设置处理器
    if logger.handlers:
        return logger

    # 格式器：包含时间、日志级别、消息
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 文件处理器：输出到文件
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    # 控制台处理器：输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def build_model(model_name, config, num_skills, num_questions, seq_len):
    if model_name == 'dkt':
        model_config = config.dkt_config
        model = DKT(num_skills, **model_config)
    elif model_name == 'simplekt':
        model_config = config.simplekt_config
        model = simpleKT(num_skills, num_questions, seq_len, **model_config)
    elif model_name == 'diskt':
        model_config = config.diskt_config
        model = DisKT(num_skills, num_questions, seq_len, **model_config)
    elif model_name == "akt":
        model_config = config.akt_config
        model = AKT(num_skills, num_questions, **model_config)
    elif model_name == 'folibikt':
        model_config = config.folibikt_config
        model = folibiKT(num_skills, num_questions, seq_len, **model_config)
    elif model_name == "sparsekt":
        model_config = config.sparsekt_config
        model = sparseKT(num_skills, num_questions, seq_len, **model_config)
    else:
        raise ValueError(
            f"Unsupported model_name={model_name!r}. "
            f"Supported models: {', '.join(SUPPORTED_MODELS)}"
        )
    return model, model_config


def assign_probabilities(train_df):
    # 1. 统计item_id的频次并按降序排序
    freq = train_df['skill_id'].value_counts().sort_values(ascending=True)
    
    # 2. 计算后35%的item数量（向上取整）
    total = len(freq)
    n = int(np.ceil(total * 0.9999))
    
    # 3. 选取后35%的item（频次较低的部分）
    low_freq_items = freq.tail(n)
    
    # 4. 计算反向概率：频次低的概率高，频次高的概率低
    # 先将频次标准化到0-1范围（反转后）
    normalized = 1 - (low_freq_items - low_freq_items.min()) / (low_freq_items.max() - low_freq_items.min())
    
    # 5. 归一化，使概率总和为1
    probabilities = normalized / normalized.sum()
    
    return probabilities-probabilities + 1./total
    
def sample_items(probabilities, n_samples=1, replace=True):
    """
    根据概率分布采样item
    
    参数:
        probabilities: 包含item_id和对应概率的Series
        n_samples: 采样数量
        replace: 是否允许重复采样（True表示有放回，False表示无放回）
    
    返回:
        采样的item_id列表
    """
    # 获取item列表和对应的概率
    items = probabilities.index.values
    probs = probabilities.values
    
    # 执行采样
    sampled = np.random.choice(
        items,
        size=n_samples,
        p=probs,
        replace=replace
    )
    
    # 如果只采一个样本，返回标量而不是数组
    return sampled[0] if n_samples == 1 else sampled
    
def generate_0_data(train_df, max_users, seq_len):
    train_user = train_df.user_id.unique()
    p_ = assign_probabilities(train_df)
    r_ = np.random.choice(train_user, size=130, replace=True)
    clf = TabPFNClassifier(n_estimators=6)
    for i in tqdm(r_):
        train_ = copy.deepcopy(train_df[train_df.user_id==i].iloc[:seq_len, :])
        train_.item_id = 0
        test_ = copy.deepcopy(train_)
        max_users += 1
        test_.user_id = max_users
        test_.skill_id = sample_items(p_, seq_len, True)
        test_.correct = np.nan
        clf.fit(train_.values[:, [1,2,4]], train_.values[:, 3])
        test_.iloc[:, 3] = clf.predict(test_.values[:, [1,2,4]])
        train_df = pd.concat([train_df, test_], axis=0)
    return train_df
    
def main(config):
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    model_name = config.model_name
    dataset_path = config.dataset_path
    data_name = config.data_name
    seed = config.seed
    test_name = config.test_name

    np.random.seed(seed)
    torch.manual_seed(seed)

    train_config = config.train_config
    checkpoint_dir = config.checkpoint_dir

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    ckpt_path = os.path.join(checkpoint_dir, model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, data_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    batch_size = train_config.batch_size
    eval_batch_size = train_config.eval_batch_size
    learning_rate = train_config.learning_rate
    optimizer = train_config.optimizer
    seq_len = train_config.seq_len
    dr_training = train_config.dr_training
    ipw = train_config.ipw
    imput_training = getattr(train_config, "imput_training", False)
    baseline = getattr(train_config, "baseline", False)
    use_ips = ipw or dr_training
    use_imputation = dr_training or imput_training
    use_drkt = use_ips or use_imputation
    mode = train_config.mode

    if train_config.sequence_option == "recent":  # the most recent N interactions
        dataset = MostRecentQuestionSkillDataset
    elif train_config.sequence_option == "early":  # the most early N interactions
        dataset = MostEarlyQuestionSkillDataset
    else:
        raise NotImplementedError("sequence option is not valid")

    test_aucs, test_accs, test_rmses = [], [], []

    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

    df_path = os.path.join(os.path.join(dataset_path, data_name), "preprocessed_df.csv")
    df = pd.read_csv(df_path, sep="\t")

    print("skill_min", df["skill_id"].min())
    users = df["user_id"].unique()
    df["skill_id"] += 1  # zero for padding
    df["item_id"] += 1  # zero for padding
    max_users = df["user_id"].nunique()
    num_skills = df["skill_id"].max() + 1
    num_questions = df["item_id"].max() + 1

    np.random.shuffle(users)

    logger = setup_logger(log_dir=f"./result/logs/{model_name}/{mode}/{data_name}/{seed}")
    logger.info(f"MODEL:{model_name}", )
    logger.info(f"IMP_MODE:{mode}", )
    logger.info(f"Data: {data_name}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Baseline enabled: {baseline}")
    logger.info(f"DR enabled: {dr_training}")
    logger.info(f"IPW enabled: {ipw}")
    logger.info(f"Imputation enabled: {use_imputation}")
    # print("MODEL", model_name)

    print(dataset)
    if data_name in ["statics", "assistments15"]:
        num_questions = 0

    for fold, (train_ids, test_ids) in enumerate(kfold.split(users)):
        model, model_config = build_model(
            model_name,
            config,
            num_skills,
            num_questions,
            seq_len,
        )

        if use_drkt:
            base_model = copy.deepcopy(model)
            model = DRKT(
                device,
                num_skills,
                embedding_size=model_config['embedding_size'],
                state_fetcher=base_model,
                dropout=model_config['dropout'],
                model_name=model_name,
                lambda_=config["lambda"],
                use_ips=use_ips,
            )

        if use_imputation:
            impute = copy.deepcopy(build_model(
                model_name,
                config,
                num_skills,
                num_questions,
                seq_len,
            )[0])
            impute_model = DRKT(
                device,
                num_skills,
                embedding_size=model_config['embedding_size'],
                state_fetcher=impute,
                dropout=0.5,
                model_name=model_name,
                imp_model=True,
                lambda_=config["lambda"],
                use_ips=False,
            )


        train_users = users[train_ids]
        np.random.shuffle(train_users)
        offset = int(len(train_ids) * 0.9)

        valid_users = train_users[offset:]
        train_users = train_users[:offset]

        test_users = users[test_ids]

        train_df = df[df["user_id"].isin(train_users)]
        valid_df = df[df["user_id"].isin(valid_users)]
        test_df = df[df["user_id"].isin(test_users)]
        train_dataset = dataset(train_df, seq_len, num_skills, num_questions)
        valid_dataset = dataset(valid_df, seq_len, num_skills, num_questions)
        test_dataset = dataset(test_df, seq_len, num_skills, num_questions)

        logger.info(f"train_ids:{len(train_users)}")
        logger.info(f"valid_ids:{len(valid_users)}")
        logger.info(f"test_ids:{len(test_users)}")
        if "dis" in model_name:  # diskt
            train_loader = accelerator.prepare(
                DataLoader(
                    CounterDatasetWrapper(
                        train_dataset,
                        seq_len,
                    ),
                    batch_size=batch_size) # )#
            )

            valid_loader = accelerator.prepare(
                DataLoader(
                    CounterDatasetWrapper(
                        valid_dataset,
                        seq_len,
                    ),
                    batch_size=eval_batch_size,
                )
            )

            test_loader = accelerator.prepare(
                DataLoader(
                    CounterDatasetWrapper(
                        test_dataset,
                        seq_len,
                    ),
                    batch_size=eval_batch_size,
                )
            )

        else:

            train_loader = accelerator.prepare(
                DataLoader(train_dataset, batch_size=batch_size)# )#
            )

            valid_loader = accelerator.prepare(
                DataLoader(valid_dataset, batch_size=eval_batch_size)
            )

            test_loader = accelerator.prepare(
                DataLoader(test_dataset, batch_size=eval_batch_size)
            )

        n_gpu = torch.cuda.device_count()
        model = model.to(device)
        if use_imputation:
            impute_model = impute_model.to(device)
        ips_paras = []
        ori_paras = []
        if use_ips:
            for name, para in model.named_parameters():
                if 'propensity' in name:
                    print(name)
                    ips_paras.append(para)
                else:
                    ori_paras.append(para)

        if optimizer == "sgd":
            if use_ips:
                opt = [SGD(ori_paras, learning_rate, momentum=0.9), SGD(ips_paras, learning_rate, momentum=0.9)]
            else:
                opt = SGD(model.parameters(), learning_rate, momentum=0.9)
            if use_imputation:
                impute_opt = SGD(impute_model.parameters(), learning_rate, momentum=0.9)

        elif optimizer == "adam":
            if use_ips:
                opt = [Adam(ori_paras, learning_rate, weight_decay=train_config.wl), Adam(ips_paras, learning_rate, weight_decay=0.001)] #weight_decay=0.0001)]
            else:
                opt = Adam(model.parameters(), learning_rate, weight_decay=train_config.wl)
            if use_imputation:
                impute_opt = Adam(impute_model.parameters(), learning_rate, weight_decay=0.0001)
                
        elif optimizer == "adamw":
            if use_ips:
                opt = [AdamW(ori_paras, learning_rate, weight_decay=train_config.wl), AdamW(ips_paras, learning_rate, weight_decay=0.01)]
            else:
                opt = AdamW(model.parameters(), learning_rate, weight_decay=train_config.wl)
            if use_imputation:
                impute_opt = AdamW(impute_model.parameters(), learning_rate, weight_decay=0.0001)

        # 采用warmup
#        num_epochs = 200
#        num_update_steps_per_epoch = len(train_loader)
#        max_train_steps = num_epochs * num_update_steps_per_epoch
#        num_warmup_steps = int(max_train_steps * 0.1) # 推荐：10% 的步数用于 Warmup

#        schedulers = [
#            get_cosine_schedule_with_warmup(opt[0], num_warmup_steps, max_train_steps),
#            get_cosine_schedule_with_warmup(opt[1], num_warmup_steps, max_train_steps)
#        ]

#        impute_scheduler = get_cosine_schedule_with_warmup(impute_opt, num_warmup_steps, max_train_steps)


        if use_ips:
            model, opt[0], opt[1] = accelerator.prepare(model, opt[0], opt[1])
        else:
            model, opt = accelerator.prepare(model, opt)
        
#        model, opt[0], opt[1], schedulers[0], schedulers[1] = accelerator.prepare(model, opt[0], opt[1], schedulers[0], schedulers[1])

#        if dr_training:
#            impute_model, impute_opt, impute_scheduler = accelerator.prepare(
#                impute_model, impute_opt, impute_scheduler
#            )
        
        
        if use_imputation:
            impute_model, impute_opt = accelerator.prepare(impute_model, impute_opt)

        test_auc, test_acc, test_rmse = model_train(
            fold,
            (model, impute_model) if use_imputation else model,
            accelerator,
            (opt, impute_opt) if use_imputation else opt,
            None, # <--- 新增：传入 Schedulers
            train_loader,
            valid_loader,
            test_loader,
            config,
            n_gpu,
            logger,
            dr=dr_training,
            use_ips=use_ips,
            use_imputation=use_imputation,
        )

        logger.info(f"===== Finished Fold {fold + 1}/{5} =====\n")
        test_aucs.append(test_auc)
        test_accs.append(test_acc)
        test_rmses.append(test_rmse)
    test_auc = np.mean(test_aucs)
    test_auc_std = np.std(test_aucs)
    test_acc = np.mean(test_accs)
    test_acc_std = np.std(test_accs)
    test_rmse = np.mean(test_rmses)
    test_rmse_std = np.std(test_rmses)

    now = (datetime.now() + timedelta(hours=9)).strftime("%Y%m%d-%H%M%S")  # KST time

    log_out_path = os.path.join(
        os.path.join("logs", "5-fold-cv", "{}".format(data_name))
    )
    os.makedirs(log_out_path, exist_ok=True)
    with open(os.path.join(log_out_path, "{}-{}".format(model_name, now)), "w") as f:
        f.write("AUC\tACC\tRMSE\tAUC_std\tACC_std\tRMSE_std\n")
        f.write("{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(test_auc, test_acc, test_rmse, test_auc_std,
                                                                          test_acc_std, test_rmse_std))
        f.write("AUC_ALL\n")
        f.write(",".join([str(auc) for auc in test_aucs]) + "\n")
        f.write("ACC_ALL\n")
        f.write(",".join([str(auc) for auc in test_accs]) + "\n")
        f.write("RMSE_ALL\n")
        f.write(",".join([str(auc) for auc in test_rmses]) + "\n")

    logger.info("\n5-fold CV Result")
    logger.info("AUC\tACC\tRMSE\tAUC_std\tACC_std\tRMSE_std\n")
    logger.info("{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(test_auc, test_acc, test_rmse, test_auc_std,
                                                                          test_acc_std, test_rmse_std))
    logger.info("AUC_ALL\n")
    logger.info(",".join([str(auc) for auc in test_aucs]) + "\n")
    logger.info("ACC_ALL\n")
    logger.info(",".join([str(auc) for auc in test_accs]) + "\n")
    logger.info("RMSE_ALL\n")
    logger.info(",".join([str(auc) for auc in test_rmses]) + "\n")
    # print("\n5-fold CV Result")
    # print("AUC\tACC\tRMSE")
    # print("{:.5f}\t{:.5f}\t{:.5f}".format(test_auc, test_acc, test_rmse))


if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str,
                        default=None,
                        choices=[None, '00', '01', '03', '05', '07', '1', '2', '5', '10'],
                        help="experiment mode",
                        )

    train_mode = parser.add_mutually_exclusive_group()
    train_mode.add_argument('--baseline', action='store_true', help='train the baseline model only')
    train_mode.add_argument('--ipw', action='store_true', help='utilize ipw')
    train_mode.add_argument('--imput', action='store_true', help='train prediction model + imputation model without IPW')
    train_mode.add_argument('--dr', action='store_true', help='utilize Doubly Robust Learning')

    parser.add_argument(
        "--model_name",
        type=str,
        default="akt",
        choices=SUPPORTED_MODELS,
        help="The name of the model to train. \
            The possible models are in [dkt, akt, simplekt, folibikt, sparsekt, diskt]. \
            The default model is akt.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="spanish",
        help="The name of the dataset to use in training.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout probability"
    )
    parser.add_argument(
        "--batch_size", type=float, default=64, help="train batch size"
    )
    parser.add_argument(
        "--embedding_size", type=int, default=64, help="embedding size"
    )
    parser.add_argument(
        "--test_name", type=str, default='low',
        help="the possible testsets are in [ednet-low, ednet-medium, ednet-high]"
    )
    parser.add_argument("--l2", type=float, default=1e-5, help="l2 regularization param")
    parser.add_argument("--wl", type=float, default=1e-4, help="wl regularization param")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.1, help="temporal smooth strength")
    args = parser.parse_args()

    base_cfg_file = PathManager.open("configs/example.yaml", "r")
    base_cfg = yaml.safe_load(base_cfg_file)
    cfg = CN(base_cfg)
    cfg.set_new_allowed(True)
    cfg.test_name = args.test_name
    cfg.model_name = args.model_name
    cfg.data_name = args.data_name
    cfg.train_config.batch_size = int(args.batch_size)
    cfg.train_config.eval_batch_size = int(args.batch_size)
    cfg.train_config.learning_rate = args.lr
    cfg.train_config.optimizer = args.optimizer
    cfg.train_config.baseline = args.baseline or not (args.ipw or args.imput or args.dr)
    cfg.train_config.dr_training = args.dr
    cfg.train_config.ipw = args.ipw
    cfg.train_config.imput_training = args.imput
    cfg.train_config.mode = args.mode
    cfg.train_config.l2 = args.l2
    cfg.train_config.wl = args.wl
    cfg["lambda"] = args.lambda_

    if args.model_name == 'dkt':  # dkt
        cfg.dkt_config.dropout = args.dropout
        cfg.dkt_config.embedding_size = args.embedding_size
    elif args.model_name == 'simplekt':  # simplekt
        cfg.simplekt_config.dropout = args.dropout
        cfg.simplekt_config.embedding_size = args.embedding_size
    elif args.model_name == 'diskt':  # dikt
        cfg.diskt_config.dropout = args.dropout
    elif args.model_name == 'akt':  # akt
        cfg.akt_config.l2 = args.l2
        cfg.akt_config.dropout = args.dropout
        cfg.akt_config.embedding_size = args.embedding_size
    elif args.model_name == 'folibikt':  # folibikt
        cfg.folibikt_config.l2 = args.l2
        cfg.folibikt_config.dropout = args.dropout
        cfg.folibikt_config.embedding_size = args.embedding_size
    elif args.model_name == 'sparsekt':  # sparsekt
        cfg.sparsekt_config.dropout = args.dropout
        cfg.sparsekt_config.embedding_size = args.embedding_size

    cfg.freeze()

    print(cfg)
    main(cfg)
