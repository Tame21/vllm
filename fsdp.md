FSDP 和 DeepSpeed ZeRO-3 的核心目标相同：把模型参数、梯度和优化器状态切分到多张卡上。主要差异在实现和工程生态。
对比项	FSDP/FSDP2	DeepSpeed ZeRO-3
所属生态	PyTorch 原生	DeepSpeed 第三方框架
参数分片	Full Shard	ZeRO Stage 3
显存效果	与 ZeRO-3 接近	与 FSDP Full Shard 接近
配置方式	Accelerate/FSDP 配置	DeepSpeed JSON
启动方式	accelerate launch	FORCE_TORCHRUN=1 llamafactory-cli
CPU/NVMe offload	支持，但配置相对有限	功能更成熟、选项更多
LoRA 兼容性	对冻结参数、wrap 策略更敏感	在 Transformers Trainer 中通常更省心
Checkpoint	PyTorch state dict / DTensor	DeepSpeed 分片 checkpoint
昇腾支持	FSDP2 是当前重点路线之一	可用，但依赖适配版 DeepSpeed
调试难度	更接近原生 PyTorch	配置项较多，但 LlamaFactory 示例成熟

两者在训练时都大致这样工作：
27B BF16 参数约 54GB
            ↓ 四卡切分
每卡基础参数约 13.5GB
            +
当前层临时聚合、激活、LoRA 参数和通信缓冲
