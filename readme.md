# Using Rechours Framework to Evaluate the Effectiveness of ANS Model   **2024/12/31**
Augmented Negative Sampling (ANS)🚀 是一种用于协同过滤模型的增强负采样技术![new](/gif/new.gif)
本项目将ANS模型用于ReChorus框架进行测试😁，用于处理多种推荐算法的研究和复现工作😉

[张文涛☘️ 的 GitHub Page](https://github.com/Zwt122544/ANS).<img src="/gif/github.gif" width="20" height="20">

[👉ANS的论文地址](https://arxiv.org/abs/2308.05972)

[👉ANS的github项目](https://github.com/Asa9aoTK/ANS-Recbole)

## Requirement![new](/gif/new.gif)  

<details open>
<summary>点击展开安装依赖</summary>

克隆 repo，并要求在 [**Python>=3.8.0**]🌟 (https://www.python.org/)  环境中安装requirements.txt
<img src="/gif/python.gif" width="20" height="20">
```bash
pip install -r requirements.txt
```
其中库包含:
- torch==1.12.1
- cudatoolkit==10.2.89
- numpy==1.22.3
- ipython==8.10.0
- jupyter==1.0.0
- tqdm==4.66.1
- pandas==1.4.4
- scikit-learn==1.1.3
- scipy==1.7.3
- pickle
- yaml
</details>


 <img src="/gif/work1.gif" width="200" height="200">


## 模型结果展示

| Data                     | Metric                | AutoCF                 | LightGCN               | FPMC                   | SLRPlus                | GRU4Rec                | NeuMF                  |
|:-------------------------|:----------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| Grocery_and_Gourmet_Food | HR@5</br><br/>NDCG@5  | 0.1121</br><br/>0.0465 | 0.3858</br><br/>0.2659 | 0.3409</br><br/>0.2606 | 0.3242</br><br/>0.2249 | 0.3682</br><br/>0.2616 | 0.3261</br><br/>0.2242 |
| MIND_Large               | HR@5</br><br/>NDCG@5  | 0.2537</br><br/>0.0807 | 0.1078</br><br/>0.0631 | 0.1804</br><br/>0.1207 | 0.1098</br><br/>0.0716 | 0.2010</br><br/>0.1221 | 0.1020</br><br/>0.0638 |
| MovieLens-1M             | HR@5</br><br/>NDCG@5  | 0.6763</br><br/>0.2832 | 0.3520</br><br/>0.2382 | 0.4181</br><br/>0.2939 | 0.3693</br><br/>0.2455 | 0.4167</br><br/>0.2859 | 0.3319</br><br/>0.2277 |
 <img src="/gif/result.png">

## ANS模型结构
<img src="/gif/structure.png">

**🚀 点击展开查看代码**
<details open>
  
  
  ```python
  class ANS(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(ANS, self).__init__(config, dataset)
        self.emb_size = config["embedding_size"]
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.neg_seq_len = config["train_neg_sample_args"]["sample_num"]
        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.user_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.item_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.pos_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.neg_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.hard_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.conf_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.easy_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.margin_model = nn.Linear(self.emb_size, 1).to(self.device)
        self.eps=config["eps"]
        self.gamma=config["gamma"]

        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        embs = lightgcn_all_embeddings
        return embs[:self.n_users, :], embs[self.n_users:, :]
    



    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        neg_item_seq = neg_item.reshape((self.neg_seq_len, -1))
        neg_item_seq = neg_item_seq.T

        neg_item = neg_item_seq
        user_number = int(len(user) / self.neg_seq_len)
        user = user[0:user_number]
        pos_item = pos_item[0:user_number]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]




        s_e = u_embeddings
        p_e = pos_embeddings
        n_e = neg_embeddings
        batch_size = user.shape[0]
        
        gate_neg_hard = torch.sigmoid(self.item_gate(n_e) * self.user_gate(s_e).unsqueeze(1))
        n_hard =  n_e * gate_neg_hard
        n_easy =  n_e - n_hard
        
        p_hard =  p_e.unsqueeze(1) * gate_neg_hard
        p_easy =  p_e.unsqueeze(1) - p_hard
    
        import torch.nn.functional as F
        distance = torch.mean(F.pairwise_distance(n_hard, p_hard, p=2).squeeze(dim=1))
        temp = torch.norm(torch.mul(p_easy, n_easy),dim=-1)
        orth = torch.mean(torch.sum(temp,axis=-1))

        margin = torch.sigmoid(1/self.margin_model(n_hard * p_hard))

        random_noise = torch.rand(n_easy.shape).to(self.device)
        magnitude = torch.nn.functional.normalize(random_noise, p=2, dim=-1) * margin *0.1
        direction = torch.sign(p_easy - n_easy)
        noise = torch.mul(direction,magnitude)
        n_easy_syth = noise + n_easy
        n_e_ = n_hard + n_easy_syth        
        hard_scores = torch.sum(torch.mul(s_e.unsqueeze(dim=1), n_hard), axis=-1)  # [batch_size, K]
        easy_scores = torch.sum(torch.mul(s_e.unsqueeze(dim=1), n_easy), axis=-1)  # [batch_size, K]
        syth_scores = torch.sum(torch.mul(s_e.unsqueeze(dim=1), n_e_), axis=-1)  # [batch_size, K]
        norm_scores = torch.sum(torch.mul(s_e.unsqueeze(dim=1), n_e), axis=-1)  # [batch_size, K]
        sns_loss = torch.mean(torch.log(1 + torch.exp(easy_scores - hard_scores).sum(dim=1)))
        dis_loss = distance + orth
        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)  # [batch_size, n_negs]
        scores_false =  syth_scores - norm_scores

        indices = torch.max(scores + self.eps*scores_false, dim=1)[1].detach()
        neg_items_emb_ = n_e_.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        # [batch_size, n_hops+1, channel]
        neg_embeddings = neg_items_emb_[[[i] for i in range(batch_size)],range(neg_items_emb_.shape[1]), indices, :]
        

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1).squeeze(dim=1).sum(dim=-1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )


        loss = mf_loss + self.reg_weight * reg_loss + self.gamma * (sns_loss + dis_loss)
        # loss = mf_loss + self.gamma * (sns_loss)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
 ```
</details>



## Usage
添加完数据集后
运行下面的命令：
```
python main.py --model_name ANS --emb_size 64 --lr 1e-3 --l2 0 --loss_n BPR --dataset ML_1MCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10 --main_metric NDCG@2 --model_mode Impression
```

- `--model_name`: 模型
- `--dataset`: 数据集名称
- `--path`: 数据集路径


## License

This project is licensed under the MIT License. It references ideas and methodologies from the following projects:

- **[👉ANS的论文地址](https://arxiv.org/abs/2308.05972)**
- **[👉ANS的github项目](https://github.com/Asa9aoTK/ANS-Recbole)**

