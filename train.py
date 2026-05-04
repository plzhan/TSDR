# import pandas as pd
# import numpy as np
# import torch
# import os
# import glob
# import logging
# from datetime import datetime, timedelta
# from tqdm import tqdm
# from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
# import shutil
# if torch.cuda.is_available():
#    torch.set_default_tensor_type(torch.cuda.FloatTensor)


# class InfiniteDataLoader:
#    def __init__(self, dataloader):
#        self.dataloader = dataloader
#        self.iterator = iter(dataloader)

#    def __next__(self):
#        # 1. 临时将默认 Tensor 类型切回 CPU
#        # 这样 RandomSampler 生成随机索引时就会在 CPU 上运行，不会报错
#        if torch.cuda.is_available():
#            torch.set_default_tensor_type('torch.FloatTensor')
       
#        try:
#            try:
#                batch = next(self.iterator)
#            except StopIteration:
#                self.iterator = iter(self.dataloader)
#                batch = next(self.iterator)
#        finally:
#            # 2. 无论成功与否，都要立即切回 GPU
#            # 保证原本的代码逻辑不受影响
#            if torch.cuda.is_available():
#                torch.set_default_tensor_type('torch.cuda.FloatTensor')
           
#        return batch

#    def __iter__(self):
#        return self 

# def model_train(
#        fold,
#        model,
#        accelerator,
#        opt,
#        scheduler,
#        train_loader,
#        valid_loader,
#        test_loader,
#        config,
#        n_gpu,
#        logger,
#        early_stop=True,
#        dr=False,
#        use_ips=True,
#        use_imputation=False,
# ):
#    train_losses = []
#    direct_losses = []
#    ips_max_list = []
#    ips_min_list = []
#    train_aucs = []
#    ips_model_losses = []
#    imputation_losses = []
#    best_valid_auc = float("-inf")
#    best_epoch = 0
#    best_loss = 1e8
#    imputation_loss = 0
#    logger.info(f"===== Start Fold {fold + 1}/5 =====")
#    num_epochs = config["train_config"]["num_epochs"]
#    model_name = config["model_name"]
#    data_name = config["data_name"]
#    train_config = config["train_config"]
#    log_path = train_config["log_path"]
#    dr_training = train_config["dr_training"]
#    mode = train_config["mode"]

#    now = (datetime.now() + timedelta(hours=9)).strftime("%Y%m%d-%H%M%S")  # KST time
#    if use_imputation:
#        model, impute_model = model
#        opt, impute_opt = opt
#    if use_ips:
#        opt1, opt2 = opt
#    else:
#        opt1, opt2 = opt, None

#    train_iterator = InfiniteDataLoader(train_loader)
# #    if dr:
# #        scheduler_list, impute_scheduler = scheduler
# #        scheduler_ori, scheduler_ips = scheduler_list
#    steps_per_epoch = int(len(train_loader) * 0.3333333333333) # 举例：只跑 10% 的数据
#    if steps_per_epoch < 1: 
#        steps_per_epoch = 1
   
   
   
   
#    for i in range(1, num_epochs + 1):
#        if use_imputation:
#            ### train imputation_model
#            for _ in range(steps_per_epoch):
#                batch = next(train_iterator) # 手动获取 batch
               
#                impute_opt.zero_grad()
#                model.eval()
#                impute_model.train()
#                with torch.no_grad():
#                    out_dict = model(batch)
#                impute_out_dict = impute_model(batch)
#                loss2 = impute_model.loss(batch, out_dict, pred_or_impt="impt", dr=True,
#                                          imputation_out_dict=impute_out_dict, inv_prop=model.inv_prop)
#                accelerator.backward(loss2)
#                impute_opt.step()
#                imputation_losses.append(loss2.item())
#            imputation_loss = np.average(imputation_losses)

#        if use_ips:
#            for _ in range(steps_per_epoch):
#                """
#                train IPS model
#                """
#                batch = next(train_iterator) # 获取新的一批 batch
               
#                opt2.zero_grad()
#                model.train()
#                out_dict = model(batch, get_state=True)
#                ips_model_loss = model.loss_ips_model(out_dict)
#                accelerator.backward(ips_model_loss)
#                ips_model_losses.append(ips_model_loss.item())
#                opt2.step()
           
#        for _ in range(steps_per_epoch):
#            batch = next(train_iterator) # 获取新的一批 batch
           
#            opt1.zero_grad()
#            model.train()
#            if hasattr(model, "imp_model"):
#                out_dict = model(batch, get_state=False)
#                if use_imputation:
#                    with torch.no_grad():
#                        impute_out_dict = impute_model(batch)

#                    loss, xent_loss, direct_loss, train_auc, ips_max, ips_min = model.loss(
#                        batch, out_dict, pred_or_impt="pred",
#                        imputation_out_dict=impute_out_dict, dr=True
#                    )
#                else:
#                    loss, xent_loss, temp_smooth_loss, train_auc, ips_max, ips_min = model.loss(
#                        batch, out_dict, pred_or_impt="pred"
#                    )
#                    direct_loss = temp_smooth_loss
#            else:
#                out_dict = model(batch)
#                loss_result = model.loss(batch, out_dict)
#                loss = loss_result[0] if isinstance(loss_result, tuple) else loss_result
#                pred = out_dict["pred"].flatten()
#                true = out_dict["true"].flatten()
#                mask = true > -1
#                train_auc = roc_auc_score(
#                    y_true=true[mask].detach().cpu().numpy(),
#                    y_score=pred[mask].detach().cpu().numpy(),
#                )
#                xent_loss = loss
#                direct_loss = torch.zeros((), device=loss.device)
#                ips_max = torch.ones((), device=loss.device)
#                ips_min = torch.ones((), device=loss.device)

#            accelerator.backward(loss)
#            opt1.step()

#            train_losses.append(xent_loss.item())
#            direct_losses.append(direct_loss.item())
#            ips_max_list.append(ips_max.item())
#            ips_min_list.append(ips_min.item())
#            train_aucs.append(train_auc)

#        """
#        evaluation
#        """
#        total_preds = []
#        total_trues = []

#        with torch.no_grad():
#            for batch in valid_loader:
#                model.eval()
#                out_dict = model(batch)
#                pred = out_dict["pred"].flatten()
#                true = out_dict["true"].flatten()
#                mask = true > -1
#                true = true[mask]
#                pred = pred[mask]

#                total_preds.append(pred)
#                total_trues.append(true)

#            total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
#            total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()

#        train_loss = np.average(train_losses)
#        train_auc_avg = np.average(train_aucs)
#        ips_model_loss_avg = np.average(ips_model_losses) if ips_model_losses != [] else 0.0
#        direct_loss_avg = np.average(direct_losses) if direct_losses != [] else 0.0
#        ips_max_list_avg = np.average(ips_max_list) if ips_max_list != [] else 1.0
#        ips_min_list_avg = np.average(ips_min_list) if ips_min_list != [] else 1.0

#        valid_auc = roc_auc_score(y_true=total_trues, y_score=total_preds)

#        path = os.path.join("./saved_model", model_name + str(mode), data_name)
#        if not os.path.isdir(path):
#            os.makedirs(path)

#        if valid_auc > best_valid_auc:

#            path = os.path.join(
#                os.path.join("./saved_model", model_name + str(mode), data_name), "params_*"
#            )
#            for _path in glob.glob(path):
#                os.remove(_path)
#            best_valid_auc = valid_auc
#            best_loss = train_loss
#            best_epoch = i
#            print("saved")
#            torch.save(
#                {"epoch": i, "model_state_dict": model.state_dict(), },
#                os.path.join(
#                    os.path.join("./saved_model", model_name + str(mode), data_name),
#                    "params_{}".format(str(best_epoch)),
#                ),
#            )
#        if best_epoch and i - best_epoch > 50:
#            break

#        # clear lists to track next epochs
#        train_losses = []
#        valid_losses = []
#        direct_losses = []
#        ips_max_list = []
#        ips_min_list = []
#        train_aucs = []
#        ips_model_losses = []
#        imputation_losses = []

#        total_preds, total_trues = [], []

#        # evaluation on test dataset
#        with torch.no_grad():
#            for batch in test_loader:
#                model.eval()
#                out_dict = model(batch)

#                pred = out_dict["pred"].flatten()
#                true = out_dict["true"].flatten()
#                mask = true > -1
#                true = true[mask]
#                pred = pred[mask]

#                total_preds.append(pred)
#                total_trues.append(true)

#            total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
#            total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()

#        test_auc = roc_auc_score(y_true=total_trues, y_score=total_preds)
#        # 记录训练信息（替代原来的print）

#        logger.info(
#            f">>Fold {fold+1}:\t Epoch {i}\nIMPUTATION LOSS: {imputation_loss:.5f}\tXENT LOSS: {train_loss:.5f}\t TS: {ips_min_list_avg:.5f}\n"
#            f"direct_loss: {direct_loss_avg:.7f}\t ips_model_loss: {ips_model_loss_avg:.7f}\t ips_max: {ips_max_list_avg}\n"
#            f"TRAIN AUC: {train_auc_avg:.5f}\tVALID AUC: {valid_auc:.5f}\tTEST AUC: {test_auc:.5f}\n"
#        )
#        # else:
#        #     logger.info(
#        #         f">>Fold {fold}:\t Epoch {i}\tTRAIN LOSS: {train_loss:.5f}\n"
#        #         f"TRAIN AUC: {train_auc_avg:.5f}\tVALID AUC: {valid_auc:.5f}\tTEST AUC: {test_auc:.5f}\n"
#        #     )
#        # print(
#        #     "\n Fold {}:\t Epoch {}\t\tTRAIN LOSS: {:.5f}\tVALID AUC: {:.5f}\tTEST AUC: {:.5f}".format(
#        #         fold, i, train_loss, valid_auc, test_auc
#        #     )
#        # )
#    checkpoint = torch.load(
#        os.path.join(
#            os.path.join("./saved_model", model_name + str(mode), data_name),
#            "params_{}".format(str(best_epoch)),
#        )
#    )
#    src_file = os.path.join(
#            os.path.join("./saved_model", model_name + str(mode), data_name),
#            "params_{}".format(str(best_epoch)),
#        )
#    dst_dir = os.path.join("./saved_model", model_name + str(mode), data_name+'_folds')  # 目标文件夹（如："output/docs"）
#    new_name = os.path.join(
#            os.path.join("./saved_model", model_name + str(mode), data_name+'_folds'),
#            "params_{}".format(str(best_epoch))+f'_{fold}',
#        )  # 新的文件名
#    os.makedirs(dst_dir, exist_ok=True)  # 避免文件夹不存在导致的错误
#    shutil.copy2(src_file, new_name)
#    print(dst_dir)
#    print(new_name)
#    model.load_state_dict(checkpoint["model_state_dict"])

#    total_preds, total_trues = [], []
#    # # evaluation on test dataset

#    with torch.no_grad():
#        for batch in test_loader:

#            model.eval()

#            out_dict = model(batch)

#            pred = out_dict["pred"].flatten()
#            true = out_dict["true"].flatten()
#            mask = true > -1
#            true = true[mask]
#            pred = pred[mask]

#            # skills = skills[mask]
#            total_preds.append(pred)
#            total_trues.append(true)

#        total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
#        total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()

#    auc = roc_auc_score(y_true=total_trues, y_score=total_preds)
#    acc = accuracy_score(y_true=total_trues >= 0.5, y_pred=total_preds >= 0.5)
#    rmse = np.sqrt(mean_squared_error(y_true=total_trues, y_pred=total_preds))

#    # else:
#    logger.info("Best Model\tTEST AUC: {:.5f}\tTEST ACC: {:5f}\tTEST RMSE: {:5f}".format(
#        auc, acc, rmse
#    ))
#    return auc, acc, rmse
import pandas as pd
import numpy as np
import torch
import os
import glob
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
import shutil
if torch.cuda.is_available():
  torch.set_default_tensor_type(torch.cuda.FloatTensor)
 

def model_train(
      fold,
      model,
      accelerator,
      opt,
      scheduler,
      train_loader,
      valid_loader,
      test_loader,
      config,
      n_gpu,
      logger,
      early_stop=True,
      dr=False,
      use_ips=True,
      use_imputation=False,
):
  DR_losses = []
  train_losses = []
  direct_losses = []
  ips_max_list = []
  ips_min_list = []
  train_aucs = []
  ips_model_losses = []
  imputation_losses = []
  best_valid_auc = float("-inf")
  best_epoch = 0
  best_loss = 1e8
  imputation_loss = 0
  logger.info(f"===== Start Fold {fold + 1}/5 =====")
  num_epochs = config["train_config"]["num_epochs"]
  model_name = config["model_name"]
  data_name = config["data_name"]
  train_config = config["train_config"]
  log_path = train_config["log_path"]
  dr_training = train_config["dr_training"]
  mode = train_config["mode"]

  now = (datetime.now() + timedelta(hours=9)).strftime("%Y%m%d-%H%M%S")  # KST time
  if use_imputation:
      model, impute_model = model
      opt, impute_opt = opt
  if use_ips:
      opt1, opt2 = opt
  else:
      opt1, opt2 = opt, None

#    if dr:
#        scheduler_list, impute_scheduler = scheduler
#        scheduler_ori, scheduler_ips = scheduler_list


  for i in range(1, num_epochs + 1):
      if use_imputation:
          ### train imputation_model
          for batch in train_loader:
              impute_opt.zero_grad()
              model.eval()
              impute_model.train()
              with torch.no_grad():
                  out_dict = model(batch)
              impute_out_dict = impute_model(batch)
              loss2 = impute_model.loss(batch, out_dict, pred_or_impt="impt",dr=True,
                                                       imputation_out_dict=impute_out_dict, inv_prop=model.inv_prop)
              accelerator.backward(loss2)
#                if train_config["max_grad_norm"] > 0.0:
#                torch.nn.utils.clip_grad_norm_(
#                        impute_model.parameters(), max_norm=1.0
#                    )

              impute_opt.step()
              #impute_scheduler.step() # 
              imputation_losses.append(loss2.item())
          imputation_loss = np.average(imputation_losses)

      if use_ips:
          for batch in train_loader:
              """
              train IPS model
              """
              opt2.zero_grad()   #####
              model.train()
              out_dict = model(batch, get_state=True)    #####
              ips_model_loss = model.loss_ips_model(out_dict)   #####
              accelerator.backward(ips_model_loss)   #####
#            if train_config["max_grad_norm"] > 0.0:   ####
#            torch.nn.utils.clip_grad_norm_(   ####
#                    model.parameters(), max_norm=1.0   ####
#                )   ####

              ips_model_losses.append(ips_model_loss.item())
              opt2.step()   #####
              #scheduler_ips.step()
         
      for batch in train_loader:
          """
          training 
          """
          ### train prediction_model 
          opt1.zero_grad()
          model.train()
          if hasattr(model, "imp_model"):
              out_dict = model(batch, get_state=False)
              if use_imputation:
                  # impute_model.eval()
                  with torch.no_grad():
                      impute_out_dict = impute_model(batch)  #

                  loss, xent_loss, direct_loss, train_auc, ips_max, ips_min = model.loss(batch, out_dict, pred_or_impt="pred",
                                                                     imputation_out_dict=impute_out_dict,dr=True)
              else:
                  loss, xent_loss, temp_smooth_loss, train_auc, ips_max, ips_min = model.loss(batch, out_dict, pred_or_impt="pred")
                  direct_loss = temp_smooth_loss
          else:
              out_dict = model(batch)
              loss_result = model.loss(batch, out_dict)
              loss = loss_result[0] if isinstance(loss_result, tuple) else loss_result
              pred = out_dict["pred"].flatten()
              true = out_dict["true"].flatten()
              mask = true > -1
              train_auc = roc_auc_score(
                  y_true=true[mask].detach().cpu().numpy(),
                  y_score=pred[mask].detach().cpu().numpy(),
              )
              xent_loss = loss
              direct_loss = torch.zeros((), device=loss.device)
              ips_max = torch.ones((), device=loss.device)
              ips_min = torch.ones((), device=loss.device)

          accelerator.backward(loss)
#            if train_config["max_grad_norm"] > 0.0:
#            torch.nn.utils.clip_grad_norm_(
#                    model.parameters(), max_norm=1.0
#                )
          opt1.step()
          #scheduler_ori.step()

          DR_losses.append(loss.item())
          train_losses.append(xent_loss.item())
          direct_losses.append(direct_loss.item())
          ips_max_list.append(ips_max.item())
          ips_min_list.append(ips_min.item())
          train_aucs.append(train_auc)

      """
      evaluation
      """
      total_preds = []
      total_trues = []

      with torch.no_grad():
          for batch in valid_loader:
              model.eval()

              out_dict = model(batch)
              pred = out_dict["pred"].flatten()
              true = out_dict["true"].flatten()
              mask = true > -1
              true = true[mask]
              pred = pred[mask]

              total_preds.append(pred)
              total_trues.append(true)

          total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
          total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()

      train_loss = np.average(train_losses)
      DR_loss = np.average(DR_losses)
      train_auc_avg = np.average(train_aucs)
      ips_model_loss_avg = np.average(ips_model_losses) if ips_model_losses != [] else 0.0
      direct_loss_avg = np.average(direct_losses) if direct_losses != [] else 0.0
      ips_max_list_avg = np.average(ips_max_list) if ips_max_list != [] else 1.0
      ips_min_list_avg = np.average(ips_min_list) if ips_min_list != [] else 1.0

      valid_auc = roc_auc_score(y_true=total_trues, y_score=total_preds)

      path = os.path.join("./saved_model", model_name + str(mode), data_name)
      if not os.path.isdir(path):
          os.makedirs(path)

      if valid_auc > best_valid_auc:

          path = os.path.join(
              os.path.join("./saved_model", model_name + str(mode), data_name), "params_*"
          )
          for _path in glob.glob(path):
              os.remove(_path)
          best_valid_auc = valid_auc
          best_loss = train_loss
          best_epoch = i
          print("saved")
          torch.save(
              {"epoch": i, "model_state_dict": model.state_dict(), },
              os.path.join(
                  os.path.join("./saved_model", model_name + str(mode), data_name),
                  "params_{}".format(str(best_epoch)),
              ),
          )
      if best_epoch and i - best_epoch > 15:
          break

      # clear lists to track next epochs
      DR_losses = []
      train_losses = []
      valid_losses = []
      direct_losses = []
      ips_max_list = []
      ips_min_list = []
      train_aucs = []
      ips_model_losses = []
      imputation_losses = []

      total_preds, total_trues = [], []

      # evaluation on test dataset
      with torch.no_grad():
          for batch in test_loader:
              model.eval()
              out_dict = model(batch)

              pred = out_dict["pred"].flatten()
              true = out_dict["true"].flatten()
              mask = true > -1
              true = true[mask]
              pred = pred[mask]

              total_preds.append(pred)
              total_trues.append(true)

          total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
          total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()

      test_auc = roc_auc_score(y_true=total_trues, y_score=total_preds)
      # 记录训练信息（替代原来的print）

      logger.info(
          f">>Fold {fold+1}:\t Epoch {i}\nDR LOSS: {DR_loss:.5f}\nIMPUTATION LOSS: {imputation_loss:.5f}\tXENT LOSS: {train_loss:.5f}\t TS: {ips_min_list_avg:.5f}\n"
          f"direct_loss: {direct_loss_avg:.7f}\t ips_model_loss: {ips_model_loss_avg:.7f}\t ips_max: {ips_max_list_avg}\n"
          f"TRAIN AUC: {train_auc_avg:.5f}\tVALID AUC: {valid_auc:.5f}\tTEST AUC: {test_auc:.5f}\n"
      )
      # else:
      #     logger.info(
      #         f">>Fold {fold}:\t Epoch {i}\tTRAIN LOSS: {train_loss:.5f}\n"
      #         f"TRAIN AUC: {train_auc_avg:.5f}\tVALID AUC: {valid_auc:.5f}\tTEST AUC: {test_auc:.5f}\n"
      #     )
      # print(
      #     "\n Fold {}:\t Epoch {}\t\tTRAIN LOSS: {:.5f}\tVALID AUC: {:.5f}\tTEST AUC: {:.5f}".format(
      #         fold, i, train_loss, valid_auc, test_auc
      #     )
      # )
  checkpoint = torch.load(
      os.path.join(
          os.path.join("./saved_model", model_name + str(mode), data_name),
          "params_{}".format(str(best_epoch)),
      )
  )
  src_file = os.path.join(
          os.path.join("./saved_model", model_name + str(mode), data_name),
          "params_{}".format(str(best_epoch)),
      )
  dst_dir = os.path.join("./saved_model", model_name + str(mode), data_name+'_folds')  # 目标文件夹（如："output/docs"）
  new_name = os.path.join(
          os.path.join("./saved_model", model_name + str(mode), data_name+'_folds'),
          "params_{}".format(str(best_epoch))+f'_{fold}',
      )  # 新的文件名
  os.makedirs(dst_dir, exist_ok=True)  # 避免文件夹不存在导致的错误
  shutil.copy2(src_file, new_name)
  print(dst_dir)
  print(new_name)
  model.load_state_dict(checkpoint["model_state_dict"])

  total_preds, total_trues = [], []
  # # evaluation on test dataset

  with torch.no_grad():
      for batch in test_loader:

          model.eval()

          out_dict = model(batch)

          pred = out_dict["pred"].flatten()
          true = out_dict["true"].flatten()
          mask = true > -1
          true = true[mask]
          pred = pred[mask]

          # skills = skills[mask]
          total_preds.append(pred)
          total_trues.append(true)

      total_preds = torch.cat(total_preds).squeeze(-1).detach().cpu().numpy()
      total_trues = torch.cat(total_trues).squeeze(-1).detach().cpu().numpy()

  auc = roc_auc_score(y_true=total_trues, y_score=total_preds)
  acc = accuracy_score(y_true=total_trues >= 0.5, y_pred=total_preds >= 0.5)
  rmse = np.sqrt(mean_squared_error(y_true=total_trues, y_pred=total_preds))

  # else:
  logger.info("Best Model\tTEST AUC: {:.5f}\tTEST ACC: {:5f}\tTEST RMSE: {:5f}".format(
      auc, acc, rmse
  ))
  return auc, acc, rmse
