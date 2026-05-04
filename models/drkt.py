import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from models import dkt, akt, folibikt, simplekt, sparsekt, diskt
from types import MethodType

class DKTwithStateFetcher(dkt.DKT):

    def get_state(self, feed_dict):
        q = feed_dict['skills']
        r = feed_dict['responses']
        masked_r = r * (r > -1).long()
        q_input = q[:, :-1]
        r_input = masked_r[:, :-1]
        q_shft = q[:, 1:]
        r_shft = r[:, 1:]
        x = q_input + self.num_skills * r_input
        xemb = self.interaction_emb(x)
        h, _ = self.lstm_layer(xemb)  # [b, t-1, :]
        return h

class AKTwithStateFetcher(akt.AKT):

    def get_state(self, feed_dict):
        pid_data = feed_dict['questions']
        r = feed_dict['responses']
        c = feed_dict['skills']
        attention_mask = feed_dict['attention_mask']
        q_data = c
        target = r * (r > -1).long()
        # Batch First
        q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        pid_embed_data = None
        if self.num_questions > 0:  # have problem id
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
            pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
            q_embed_data = q_embed_data + pid_embed_data * \
                           q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            qa_embed_diff_data = self.qa_embed_diff(
                target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
            if self.separate_qr:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                                qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                                (
                                            qa_embed_diff_data + q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）

            c_reg_loss = (pid_embed_data ** 2.0).sum() * self.l2  # rasch部分loss
        else:
            c_reg_loss = 0.

        # BS.seqlen,embedding_size
        # Pass to the decoder
        # output shape BS,seqlen,embedding_size or embedding_size//2
        h = self.model(q_embed_data, qa_embed_data, pid_embed_data)

        return h[:, 1:]

class folibiKTwithStateFethcer(folibikt.folibiKT):

    def get_state(self, feed_dict):
        pid_data = feed_dict['questions']
        r = feed_dict['responses']
        c = feed_dict['skills']
        attention_mask = feed_dict['attention_mask']
        q_data = c
        target = r * (r > -1).long()
        emb_type = self.emb_type
        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        pid_embed_data = None
        if self.num_questions > 0: # have problem id
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
            pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            qa_embed_diff_data = self.qa_embed_diff(
                target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
            if self.separate_qr:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）

            c_reg_loss = (pid_embed_data ** 2.0).sum() * self.l2 # rasch部分loss
        else:
            c_reg_loss = 0.

        h = self.model(q_embed_data, qa_embed_data, pid_embed_data)

        return h[:, 1:]



class simpleKTwithStateFetcher(simplekt.simpleKT):

    def get_state(self, feed_dict):
        q = feed_dict['questions']
        c = feed_dict['skills']
        r = feed_dict['responses']
        qshft = q[:, 1:]
        cshft = c[:, 1:]
        masked_r = r * (r > -1).long()
        rshft = masked_r[:, 1:]
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((masked_r[:,0:1], rshft), dim=1)


        # Batch First
        q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        if self.num_questions > 0: # have problem id
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
            pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct # question encoder

        # BS.seqlen,embedding_size
        # Pass to the decoder
        # output shape BS,seqlen,embedding_size or embedding_size//2
        h = self.model(q_embed_data, qa_embed_data)
        return h[:, 1:]



class sparseKTwithStateFetcher(sparsekt.sparseKT):

    def get_state(self, feed_dict):
        pid_data = feed_dict['questions']
        r = feed_dict['responses']
        c = feed_dict['skills']
        attention_mask = feed_dict['attention_mask']
        q_data = c
        target = r * (r > -1).long()

        emb_type = self.emb_type
        sparse_ratio = self.sparse_ratio
        k_index = self.k_index
        num_skills = self.num_skills

        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        if self.num_questions > 0 and emb_type.find("norasch") == -1: # have problem id
            if emb_type.find("aktrasch") == -1:
                q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
                pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
                q_embed_data = q_embed_data + pid_embed_data * \
                    q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            else:
                q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
                pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
                q_embed_data = q_embed_data + pid_embed_data * \
                    q_embed_diff_data  # uq *d_ct + c_ct # question encoder

                qa_embed_diff_data = self.qa_embed_diff(
                    target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
                qa_embed_data = qa_embed_data + pid_embed_data * \
                        (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）

        # BS.seqlen,embedding_size
        # Pass to the decoder
        # output shape BS,seqlen,embedding_size or embedding_size//2
        if emb_type in ["qid", "qidaktrasch", "qid_scalar", "qid_norasch"]:
            h, attn_weights = self.model(q_embed_data, qa_embed_data)

        elif emb_type.find("attn") != -1:
            h, attn_weights = self.model(q_embed_data, qa_embed_data, emb_type, sparse_ratio, k_index)

        return h[:, 1:]

class disKTwithStateFetcher(diskt.DisKT):

    def get_state(self, feed_dict):
        q_seq = feed_dict['questions']
        s_seq = feed_dict['skills']
        r_seq = feed_dict['responses']
        counter_attention_mask, attention_mask = feed_dict['attention_mask']

        masked_r = r_seq * (r_seq > -1).long()

        pos_q_embed_data, pos_qa_embed_data, _ = self.rasch_emb(masked_r * s_seq, masked_r * q_seq, 2 - masked_r)
        neg_q_embed_data, neg_qa_embed_data, _ = self.rasch_emb((1 - masked_r) * s_seq, (1 - masked_r) * q_seq,
                                                                2 * masked_r)
        q_embed_data, qa_embed_data, pid_embed_data = self.rasch_emb(s_seq, q_seq, masked_r)

        y1, y2, y = pos_qa_embed_data, neg_qa_embed_data, qa_embed_data
        x = q_embed_data

        distance = F.pairwise_distance(y1.view(y1.size(0), -1), y2.view(y2.size(0), -1))
        reg_loss = torch.mean(distance) * 0.001

        h = self.model(x, y)
        d_output = h  # 
        y1, y2 = self.ffn(y1), self.ffn(y2)

        y1, y2 = self.dual_attention(h, h, y1, y2, counter_attention_mask)

        x = h - (y1 + y2)
        h = x - pid_embed_data
        return h[:, 1:]


class DRKT(nn.Module):
    def __init__(self, device, num_skills, embedding_size, state_fetcher=None, dropout=0.5, model_name='akt', imp_model=False, lambda_=0.1, use_ips=True):
        super(DRKT, self).__init__()
        self.device = device
        self.num_skills = num_skills
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.state_fetcher = state_fetcher
        self.imp_model = imp_model
        self.lambda_ = lambda_
        self.use_ips = use_ips
        if not imp_model and use_ips:
            self.propensity = nn.Sequential(
                #nn.Dropout(dropout),
                #nn.Linear(self.embedding_size, 2*self.embedding_size),
                nn.Linear(self.embedding_size, self.num_skills),
                nn.Sigmoid()
            )
            self.ips_one_hot_feat = torch.eye(self.num_skills).to(device)
        #self.mask_in = torch.zeros((1, 49), device=device)
        self.get_fetcher(model_name)

    def get_fetcher(self, name):
        print("base_model:", name)
        if name == 'akt':
            StateFetcher = AKTwithStateFetcher
        elif name == 'dkt':
            StateFetcher = DKTwithStateFetcher
        elif name == 'simplekt':
            StateFetcher = simpleKTwithStateFetcher
        elif name == 'folibikt':
            StateFetcher = folibiKTwithStateFethcer
        elif name == 'sparsekt':
            StateFetcher = sparseKTwithStateFetcher
        elif name == 'diskt':
            StateFetcher = disKTwithStateFetcher
        else:
            print("None")
        self.state_fetcher.get_state = MethodType(StateFetcher.get_state, self.state_fetcher)

    # def add_gumbel_noise(self, x, eps=1e-10):
    #     """
    #     为输入Tensor x添加Gumbel噪声
    #     Args:
    #         x: 输入Tensor
    #         eps: 防止log(0)的微小值
    #     Returns:
    #         添加噪声后的Tensor
    #     """
    #     # 生成[0,1)区间的均匀分布随机数
    #     uniform = torch.rand_like(x)
    #     # 计算Gumbel噪声：-log(-log(uniform + eps) + eps)
    #     gumbel_noise = -torch.log(-torch.log(uniform + eps) + eps)
    #     # 添加噪声到原始Tensor
    #     return x + gumbel_noise * 0.01

    def forward(self, feed_dict, get_state=False):
        q = feed_dict['skills']
        r = feed_dict['responses']
        self.n = q.size(0)
        h = self.state_fetcher.get_state(feed_dict)
        self.h = h
        if not self.imp_model and self.use_ips:
             propensity_target = F.embedding(q, self.ips_one_hot_feat)
             propensity_res = self.propensity(h)
             inv_prop_pred = torch.gather(propensity_res, dim=-1, index=q[:, 1:].unsqueeze(dim=-1)).squeeze(dim=-1)
             self.inv_prop = 1. / (inv_prop_pred.flatten().detach())
             if get_state:  # 只有预测模型需要得到知识状态的时候才会进来
                 out_dict = {
                     "propensity_res": propensity_res,  # [batch_size, seq_len-1, num_skills]
                     "propensity_target": propensity_target[:, 1:, :],
                     "true": r[:, 1:],
                 }
                 return out_dict
            # 1. 计算 Logits (未归一化的分数)
#            propensity_logits = self.propensity(h) # [B, T-1, num_skills]
#            
#            # 2. 计算当前时刻真实出现的 skill 的概率 (Propensity Score)
#            # 使用 Softmax 归一化，保证和为1
#            propensity_probs = F.softmax(propensity_logits, dim=-1)
#            
#            # 3. 提取真实发生题目的概率
#            # q[:, 1:] 是下一时刻的 skill id
#            # gather: 从 dim=-1 (skill维度) 中选取对应 q_next 的概率值
#            inv_prop_pred = torch.gather(propensity_probs, dim=-1, index=q[:, 1:].unsqueeze(dim=-1)).squeeze(dim=-1)
#            
#            # 4. 计算逆倾向分数
#            # 加上一个 eps 防止除以 0，或者 clamp
#            self.inv_prop = 1. / (inv_prop_pred.flatten().detach() + 1e-6) 
#            if get_state:
#                out_dict = {
#                    "propensity_logits": propensity_logits, # 存 Logits 方便计算 Loss
#                    "true_skill_indices": q[:, 1:],         # 存 Skill ID
#                    "true": r[:, 1:],
#                }
#                return out_dict
        elif not self.imp_model:
             self.inv_prop = torch.ones_like(r[:, 1:].flatten(), dtype=torch.float, device=r.device)
             if get_state:
                 raise RuntimeError("Propensity model is disabled when use_ips=False.")
        out_dict = self.state_fetcher(feed_dict)
        return out_dict  # ctf or not ctf

    def loss_ips_model(self, out_dict):
        propensity_res = out_dict["propensity_res"]
        propensity_target = out_dict["propensity_target"]
        bs, t = torch.where(out_dict["true"] > -1)
        ips_model_loss = F.binary_cross_entropy(propensity_res[bs, t], propensity_target[bs, t], reduction='mean')
        return ips_model_loss
#        """
#        训练 Propensity 模型：预测下一题是哪个 Skill
#        这变成了一个标准的序列多分类问题
#        """
#        propensity_logits = out_dict["propensity_logits"] # [B, T, K]
#        true_skill_indices = out_dict["true_skill_indices"] # [B, T]
#        
#        # 使用 mask 过滤掉 padding 部分
#        true_responses = out_dict["true"]
#        mask = true_responses > -1
#        
#        # Flatten for loss calculation
#        flat_logits = propensity_logits[mask]
#        flat_targets = true_skill_indices[mask]
#        
#        # 使用 Cross Entropy Loss (自带 Softmax)
#        ips_model_loss = F.cross_entropy(flat_logits, flat_targets)
#        
#        return ips_model_loss
        
        #0.78797 0.72975	0.42674	0.00746	0.01367	0.00626
    def loss(self, feed_dict, out_dict, pred_or_impt="pred", imputation_out_dict=None, dr=False, inv_prop=None):
        """
        :param feed_dict:
        :param out_dict:
        :param prediction_model_traning:
        :param imputation_y:
        :param ctf_result: pred_u, imputation_y1
        :return:
        """

        n = out_dict["pred"].size(0)
        pred = out_dict["pred"].flatten()
        true = out_dict["true"].flatten()
        mask = true > -1
        bs, t = torch.where(out_dict["true"] > -1)
        temp_smooth_loss = self.temporal_smoothness_loss(self.h, self.lambda_)
        #  预测模型计算的
        if pred_or_impt == 'pred':

            if "c_reg_loss" in out_dict:
                c_reg_loss = out_dict["c_reg_loss"]
            self.inv_prop = torch.clamp(self.inv_prop, 1., 10.)
            train_auc = roc_auc_score(y_true=true[mask].detach().cpu().numpy(),
                                      y_score=pred[mask].detach().cpu().numpy())

        if not dr:  # only ips
            xent_loss = F.binary_cross_entropy(pred[mask], true[mask], weight=self.inv_prop[mask], reduction='mean')
            if "c_reg_loss" in out_dict:
                loss = xent_loss + out_dict["c_reg_loss"] #+ temp_smooth_loss# + temp_smooth_loss
            else:
                loss = xent_loss #+ temp_smooth_loss
              # 12 :2025-10-24 20:00:12,638 - INFO - 0.78925	0.73203	0.42476	0.01467	0.01906	0.00713
            return loss, xent_loss, temp_smooth_loss, train_auc, self.inv_prop[mask].max(), self.inv_prop[mask].min()

        else:  ## dr
            imputation_y = imputation_out_dict["pred"].flatten()
            if pred_or_impt == 'impt':  # 训练插补模型
                #e_loss = F.binary_cross_entropy(pred[mask], true[mask], reduction='none')
                e_loss = F.mse_loss(pred[mask], true[mask], reduction='none')
                e_hat_loss = F.mse_loss(imputation_y[mask], pred[mask], reduction='none')
                #loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop[mask].detach() ** 2) * (1 - 1 / inv_prop[mask].detach())).mean() + temp_smooth_loss # MRDR 理论
                ##loss = (e_hat_loss * (inv_prop[mask].detach() ** 2) * (1 - 1 / inv_prop[mask].detach())).mean() + temp_smooth_loss
                if self.lambda_ != 0:
                    loss = (((e_loss - e_hat_loss) ** 2) * inv_prop[mask].detach()).mean() + temp_smooth_loss# + temp_smooth_loss  # inv_prop是预测模型的
                else:
                    loss = (((e_loss - e_hat_loss) ** 2) * inv_prop[mask].detach()).mean() #+ temp_smooth_loss# + temp_smooth_loss  # inv_prop是预测模型的
                if "c_reg_loss" in imputation_out_dict:
                    c_reg_loss = imputation_out_dict["c_reg_loss"]
                    loss = loss + c_reg_loss
                return loss
            else:
                pred_u = out_dict["ctf_result"]
                xent_loss = F.binary_cross_entropy(pred[mask], true[mask], weight=self.inv_prop[mask], reduction='mean')
                imputation_y1 = imputation_out_dict["ctf_result"]
                with torch.no_grad():
                    imputation_loss = F.binary_cross_entropy(pred[mask], imputation_y[mask], weight=self.inv_prop[mask], reduction='mean')  #
                # imputation_loss = F.mse_loss(pred[mask], imputation_y[mask], reduction='mean')  #
                direct_loss = F.mse_loss(pred_u[bs, t, :], imputation_y1[bs, t, :], reduction='mean') # 
                #direct_loss = F.binary_cross_entropy(pred_u[bs, t, :], imputation_y1[bs, t, :], reduction='mean') # 
                DRloss = (xent_loss - imputation_loss + direct_loss)  ## / n # + temp_smooth_loss
                if self.lambda_ != 0:
                    loss = DRloss ## / n # + temp_smooth_loss
                else:
                    loss = DRloss
                if "c_reg_loss" in out_dict:
                    loss = loss + out_dict["c_reg_loss"]
                return loss, xent_loss - imputation_loss, direct_loss, train_auc, self.inv_prop[mask].var(), temp_smooth_loss,

    def temporal_smoothness_loss(self, state, lambda_smooth=0.5):  # 调这个
        """
        state: (B, T, D)  学生知识状态序列
        lambda_smooth: 正则化强度系数
        """
        # 相邻时刻的差分
        diff = state[:, 1:, :] - state[:, :-1, :]  # shape: (B, T-1, D)
        # L2 平滑损失
        loss_smooth = (diff ** 2).mean() * lambda_smooth
        return loss_smooth
