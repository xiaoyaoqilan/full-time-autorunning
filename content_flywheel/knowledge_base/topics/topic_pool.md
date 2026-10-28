# 话题池 · pending（pipeline --gen 从这里取题，做完勾掉）

## 使用方法
- 每行一条，格式：`- [ ] 选题 | 目标类型(tech/crypto) | 角度提示`
- --gen 自动取第一条未勾选的，生成完打 [x] 并把成品路径记进 archive 索引
- 高流量回流（feedback.py）会自动在这里追加"衍生选题"

## 待做队列

- [ ] GPU 上的矩阵乘法到底怎么算的：从 CUDA core 到 Tensor Core | tech | 接第一层，用 day2 的 Block 引入
- [ ] 为什么 KV Cache 是推理时代的石油：显存账本精算 | tech | 用 32GiB 验证数据
- [ ] 模型量化怎么把 14GB 压到 3.5GB：INT4 的数学 | tech | 对称/非对称/AWQ 直觉
- [ ] 一条推文是怎么被 X 分发的：兴趣池与破圈机制 | tech | 结合我们自己 13 篇的实测数据
- [ ] Solana Alpenglow 之后：150 毫秒意味着什么 | crypto | 观察视角，不涉交易
- [ ] 稳定币正在变成传统金融的底层管道 | crypto | Visa/SoFi/银行存款代币，观察视角
- [ ] RWA 还在早期但方向清晰了 | crypto | 代币化股票季度数据，观察视角
- [ ] 从零写一个 mini-vLLM：分页 KV 的 200 行实现 | tech | 知乎高收藏型选题
- [ ] 大模型训练一次到底烧多少钱：FLOPs 账本 | tech | 算力经济学
- [ ] 为什么面试总考手写 Attention：它筛的是什么能力 | tech | 结合自己实习/面试经历

## 衍生选题区（feedback 自动追加，格式同上）
