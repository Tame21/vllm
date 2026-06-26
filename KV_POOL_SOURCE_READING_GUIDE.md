# vLLM-Ascend KV Pool 源码走读路线

这份路线面向 `AscendStoreConnector` 这条 KV Pool 主线，目标是从“配置如何生效”一路走到“请求命中、KV 加载、前向计算、KV 保存、请求结束清理”的完整机制。

建议按顺序阅读。每一阶段先看入口和数据结构，再看细节实现；不要一开始就钻进 Mooncake/Memcache 的底层接口，否则很容易迷路。

## 0. 先建立整体模型

KV Pool 不是模型里的 pooling，而是一个外部 KV cache 存储/复用机制。

核心分工：

- vLLM scheduler 侧决定一个请求有多少 token 可以从 KV Pool 命中，并构造本轮 worker 需要执行的 load/save 元数据。
- worker 侧拿到元数据后，真正把 KV cache 从后端加载到本地 block，或把本地 block 保存到后端。
- backend 侧屏蔽 Mooncake、Memcache、Yuanrong 等存储差异。
- layerwise 模式把整段 KV 的保存/加载拆成逐层操作，用传输和 attention 计算重叠来降低阻塞。

主目录：

```text
vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/
  ascend_store_connector.py   # vLLM KVConnector 接口适配层
  pool_scheduler.py           # scheduler 侧命中、调度、元数据构造
  pool_worker.py              # worker 侧 KV cache 注册、load/save 执行
  kv_transfer.py              # load/save 线程、分块传输、layerwise 批处理
  config_data.py              # key、metadata、request tracker 等核心数据结构
  backend/
    backend.py                # 后端抽象接口
    mooncake_backend.py       # Mooncake Store 后端
    memcache_backend.py       # Memcache 后端，layerwise/GVA 重点相关
    yuanrong_backend.py       # Yuanrong 后端
```

## 1. 从功能文档和启动参数开始

先读：

- `docs/source/user_guide/feature_guide/kv_pool.md`
- `docs/source/user_guide/feature_guide/layerwise_kv_pool.md`

重点不是部署细节，而是把这些配置项和源码变量对应起来：

- `kv_connector`: 通常是 `AscendStoreConnector`，旧名 `MooncakeConnectorStoreV1` 也会映射到它。
- `kv_role`: `kv_producer`、`kv_consumer`、`kv_both`。
- `backend`: `mooncake`、`memcache`、`yuanrong`。
- `load_async`: 是否异步加载。
- `consumer_is_to_put`: decode/consumer 节点是否也把 KV 写回 pool。
- `consumer_is_to_load`: consumer 是否从 KV Pool 读。
- `use_layerwise`: 是否启用逐层 load/save。
- `lookup_rpc_port`: scheduler 和 worker lookup RPC 的端口。
- `kv_load_failure_policy`: load 失败后是 `fail` 还是 `recompute`。

读完后你应该能回答：

- 当前实例是只写、只读，还是读写都做？
- KV Pool 是作为 PD 分离的共享前缀缓存，还是单实例/混合模式下的本地外部缓存？
- 普通模式和 layerwise 模式的阻塞点分别在哪里？

## 2. 看 connector 如何注册到 vLLM

阅读：

- `vllm_ascend/distributed/kv_transfer/__init__.py`
- `setup.py` 中 `ascend_kv_connector = vllm_ascend:register_connector`
- `vllm_ascend/__init__.py`

调用关系：

```text
vllm-ascend package/plugin 初始化
  -> vllm_ascend.register_connector()
    -> vllm_ascend.distributed.kv_transfer.register_connector()
      -> KVConnectorFactory.register_connector("AscendStoreConnector", ...)
```

这里确认两件事：

- `AscendStoreConnector` 如何挂到上游 vLLM 的 `KVConnectorFactory`。
- `MultiConnector` 被替换成 `AscendMultiConnector`，所以组合使用 Mooncake P2P + AscendStore KV Pool 时也要看 `ascend_multi_connector.py`。

建议顺手读：

- `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py`

它解释了多个 connector 并存时，命中 token 和元数据如何在多个 connector 之间协调。

## 3. 先读接口适配层 AscendStoreConnector

阅读：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py`

这是最好的代码入口。它把 vLLM 的标准 KV connector 生命周期拆成 scheduler 侧和 worker 侧。

重点方法：

```text
Scheduler side:
  get_num_new_matched_tokens()
  update_state_after_alloc()
  build_connector_meta()
  request_finished()
  request_finished_all_groups()
  update_connector_output()
  take_events()
  bind_gpu_block_pool()

Worker side:
  register_kv_caches()
  start_load_kv()
  wait_for_layer_load()
  save_kv_layer()
  wait_for_save()
  get_finished()
  get_block_ids_with_load_errors()
  build_connector_worker_meta()
```

阅读时只需要先记住：

- `role == KVConnectorRole.SCHEDULER` 时创建 `KVPoolScheduler`。
- `role == KVConnectorRole.WORKER` 时创建 `KVPoolWorker`。
- 非 layerwise 且 rank 0 worker 会启动 `LookupKeyServer`，给 scheduler 查 KV Pool 命中。
- `use_layerwise` 会改变 wait/save 的行为，也会要求 piecewise graph。

这一层不要深挖算法，只把它当“总插座”。

## 4. 接上 vLLM 上游调用点

阅读上游 vLLM：

- `vllm/v1/core/sched/scheduler.py`
- `vllm/distributed/kv_transfer/kv_transfer_state.py`
- `vllm/v1/worker/gpu/kv_connector.py`

同时看 vllm-ascend 的 worker 接入：

- `vllm_ascend/worker/worker.py`
- `vllm_ascend/worker/model_runner_v1.py`

关键调用链：

```text
Scheduler 初始化
  vllm/v1/core/sched/scheduler.py
    -> KVConnectorFactory.create_connector(... role=SCHEDULER ...)
    -> connector.bind_gpu_block_pool(...)

Worker 初始化
  vllm_ascend/worker/worker.py
    -> ensure_kv_transfer_initialized(vllm_config, kv_cache_config)
    -> KVConnectorFactory.create_connector(... role=WORKER ...)

KV cache 注册
  vllm_ascend/worker/model_runner_v1.py
    -> get_kv_transfer_group().register_kv_caches(kv_caches)
    -> AscendStoreConnector.register_kv_caches()
    -> KVPoolWorker.register_kv_caches()

每轮调度
  scheduler.py
    -> connector.get_num_new_matched_tokens()
    -> KVCacheManager.allocate_slots(... num_external_computed_tokens ...)
    -> connector.update_state_after_alloc()
    -> connector.build_connector_meta()

每轮 worker 前后向
  vllm/v1/worker/gpu/kv_connector.py
    -> pre_forward(): connector.start_load_kv(...)
    -> post_forward(): connector.wait_for_save()
    -> connector.get_finished()
    -> connector.build_connector_worker_meta()

worker 输出回到 scheduler
  scheduler.py
    -> connector.update_connector_output(kv_connector_output)

请求结束
  scheduler.py
    -> connector.request_finished()/request_finished_all_groups()
```

读这一阶段时，重点看“什么时候调用”，不要急着看每个方法里面做什么。

## 5. 理解核心数据结构和 key 设计

阅读：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py`

建议按这个顺序：

1. `KeyMetadata`
2. `PoolKey`
3. `LayerPoolKey`
4. `ChunkedTokenDatabase`
5. `LoadSpec`
6. `RequestTracker`
7. `ReqMeta`
8. `AscendConnectorMetadata`
9. layerwise 相关的 `LayerTransferTask`、`LayerLoadTask`、`LayerBatchReqMeta`
10. `AscendStoreKVConnectorWorkerMetadata`

最关键的是 key 的组成：

```text
model_name
pcp_rank / dcp_rank
head_or_tp_rank
pp_rank
kv_cache_group_id
cache_role
cache_family
chunk_hash
layer_id(layerwise only)
```

它决定了“什么 KV 可以复用”。读到这里要特别注意：

- `chunk_hash` 来自 vLLM block hash，用于表示 token 前缀块。
- `kv_cache_group_id` 支持 hybrid KV cache group。
- `cache_family` 支持压缩比例/混合 cache 布局，例如 DeepSeek V4 这类特殊模型。
- `LayerPoolKey` 在普通 key 上额外加 `layer_id`，用于 layerwise 逐层存储。
- `ChunkedTokenDatabase.process_tokens()` 把 token 长度和 block hash 转成可查询/可保存的 KV Pool key。
- `prepare_value()` 和 `prepare_value_layer()` 把 block id 转成后端读写需要的内存地址和 size。

这一阶段建议手动画一个表：一行是一个 block/chunk，一列是 `PoolKey.to_string()` 结果，一列是 block id，一列是本地 KV cache 地址。

## 6. 深入 scheduler 侧：命中、分配后更新、构造元数据

阅读：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`

先看 `KVPoolScheduler.__init__()`：

- 读取 `kv_role`、`consumer_is_to_load`、`consumer_is_to_put`、`load_async`、`save_decode_cache`。
- 创建 backend scheduler client。
- 创建 `ChunkedTokenDatabase`。
- 维护 `_request_trackers`、`_loading_req_ids`、`_preempted_req_ids` 等状态。

然后按这条主线读：

```text
get_num_new_matched_tokens(request, num_computed_tokens)
  -> 查询 KV Pool 中最长可命中的前缀
  -> 返回 num_external_tokens 和是否 async load

update_state_after_alloc(request, blocks, num_external_tokens)
  -> scheduler 已经给请求分配本地 KV block
  -> 创建/更新 RequestTracker
  -> 记录这个请求后续需要 load/save 哪些 block

build_connector_meta(scheduler_output)
  -> 遍历 scheduled_new_reqs / scheduled_cached_reqs
  -> _process_new_request()
  -> _process_running_cached_request()
  -> _process_preempted_cached_request()
  -> _process_async_load_request()
  -> 生成 AscendConnectorMetadata，交给 worker

update_connector_output(connector_output)
  -> worker 告知哪些请求 load/save 完成
  -> 清理 loading 状态、处理 completed events

request_finished()/request_finished_all_groups()
  -> 请求结束时决定是否触发最终保存
  -> 返回 async_save 和 kv_transfer_params
```

非 layerwise 普通命中链路中还要看：

- `LookupKeyClient`
- `get_zmq_rpc_path_lookup()`

对应 worker 侧的 `LookupKeyServer`。scheduler 通过 ZMQ 问 worker：“这些 key 后端里存在多少 token？”

这一阶段你要能回答：

- vLLM 自己已经命中的 prefix cache 和 KV Pool 外部命中如何叠加？
- `num_external_tokens` 是如何影响本地 block 分配和后续 forward 的？
- 为什么 `update_state_after_alloc()` 必须在 `get_num_new_matched_tokens()` 之后？
- preempted/cached/new request 在元数据构造时有什么差异？

## 7. 深入 worker 侧：注册 KV cache、启动线程、执行 load/save

阅读：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`

先看 `KVPoolWorker.__init__()` 和 `_init_kv_transfer_config()`：

- 解析 `kv_role`、backend、parallel、layerwise、hybrid cache 等配置。
- 创建 backend 实例。
- 创建 `ChunkedTokenDatabase`。
- 根据普通/layerwise、GVA/key path 选择发送和接收线程。

然后看：

```text
register_kv_caches(kv_caches)
  -> 收集每层 KV cache tensor 的地址、block 大小、stride
  -> token_database.set_group_buffers(...)
  -> backend.register_buffer(...)

start_load_kv(metadata)
  -> 从 AscendConnectorMetadata 中找 load_spec
  -> 把 load 任务投递给接收线程

wait_for_layer_load()
  -> layerwise 模式中，在每层 attention 计算前等待该层 KV 到位

save_kv_layer(metadata)
  -> layerwise 模式中，每层 attention 后保存该层 KV

wait_for_save(metadata)
  -> 非 layerwise 模式中，forward 后统一保存 KV

get_finished(finished_req_ids, meta)
  -> 返回 worker 已完成 sending/recving 的请求集合

lookup_scheduler(...)
  -> 给 scheduler lookup RPC 用，计算后端已有多少 token
```

这一层最重要的是把“metadata 中的 token/block/key 信息”映射成“后端 get/put 的地址数组”。

如果只关心普通 KV Pool，先读非 layerwise 的路径：

- `KVCacheStoreSendingThread`
- `KVCacheStoreRecvingThread`

如果关心 layerwise，再读：

- `KVCacheStoreLayerSendingThread`
- `KVCacheStoreLayerRecvingThread`
- `KVCacheStoreKeyLayerSendingThread`
- `KVCacheStoreKeyLayerRecvingThread`

## 8. 看 kv_transfer.py：真正的传输线程和 layerwise 批处理

阅读：

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`

建议先读基类：

- `KVTransferThread`

它提供：

- request queue
- finished request 记录
- stored request 计数
- KV event 生成
- `_get_block_size()` 等公共工具

然后分两条线：

普通模式：

```text
KVCacheStoreSendingThread
  -> 根据 ReqMeta 生成 keys/addrs/sizes
  -> backend.put(keys, addrs, sizes)

KVCacheStoreRecvingThread
  -> 根据 ReqMeta.load_spec 生成 keys/addrs/sizes
  -> backend.get(keys, addrs, sizes)
```

Layerwise 模式：

```text
LayerBatchBuilder
  -> 预计算跨层共享 block 数据
  -> 每层生成 addr/size/gva 数组

KVCacheStoreLayerSendingThread / KVCacheStoreKeyLayerSendingThread
  -> 每层保存

KVCacheStoreLayerRecvingThread / KVCacheStoreKeyLayerRecvingThread
  -> 每层加载

LayerLoadTask / LayerTransferTask
  -> 描述某层要加载/保存哪些 block range
```

重点理解：

- 普通模式是“整段 KV 一次性 get/put”。
- layerwise 模式是“第 i 层 KV 到位后即可算第 i 层，同时预取/保存其他层”。
- GVA path 更偏 memcache 直接设备地址传输；key path 更偏传统 key/address/size 组织。

## 9. 看 backend 抽象和三个后端实现

先读：

- `backend/backend.py`

抽象接口只有几个核心方法：

```text
set_device()
register_buffer(ptrs, lengths)
exists(keys)
batch_get_key_info(keys)
batch_alloc(keys, sizes)
put(keys, addrs, sizes)
get(keys, addrs, sizes)
```

然后按你实际使用的后端读：

### Mooncake

阅读：

- `backend/mooncake_backend.py`

关注：

- `MooncakeStoreConfig.load_from_env()`
- `MOONCAKE_CONFIG_PATH`
- `global_segment_size`
- SSD offload 配置
- `register_buffer()`
- `exists()`
- `put()`
- `get()`

Mooncake 是默认 backend，普通 KV Pool 走读优先看它。

### Memcache

阅读：

- `backend/memcache_backend.py`
- `vllm_ascend/memcache_comm_fence.py`

关注：

- `register_buffer()`
- `batch_get_key_info()`
- `batch_alloc()`
- `put()`
- `get()`
- layerwise/GVA 相关路径

如果你要重点研究 `use_layerwise: true`，Memcache 是必须看的。

### Yuanrong

阅读：

- `backend/yuanrong_backend.py`

关注：

- `YuanrongConfig`
- `YuanrongHelper`
- 批量切片 `_iter_slices()`
- `put()/get()/exists()` 的错误处理和批处理逻辑

## 10. 看 attention 中 layerwise hook 如何插入前向计算

阅读：

- `vllm_ascend/attention/utils.py`
- `vllm_ascend/attention/sfa_v1.py`
- `vllm_ascend/attention/mla_v1.py`
- 可对照上游：`vllm/model_executor/layers/attention/kv_transfer_utils.py`

关键调用：

```text
attention layer before compute
  -> connector.wait_for_layer_load(layer_name)

attention layer after KV ready / after compute
  -> connector.save_kv_layer(layer_name, kv_cache_layer, attn_metadata)
```

这一阶段要把 layerwise 的时间线串起来：

```text
worker.start_load_kv(metadata)
  -> recv thread 开始预取 layer 0/1/...
attention layer 0
  -> wait_for_layer_load(layer0)
  -> compute attention layer0
  -> save_kv_layer(layer0)
attention layer 1
  -> wait_for_layer_load(layer1)
  -> compute attention layer1
  -> save_kv_layer(layer1)
...
```

对比普通模式：

```text
start_load_kv()
  -> 等整段需要加载的 KV 完成
forward
wait_for_save()
  -> forward 后整段保存
```

## 11. 关注异常和 recompute 机制

阅读：

- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/core/recompute_scheduler.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
- 测试：`tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py`

重点关键词：

- `kv_load_failure_policy`
- `get_block_ids_with_load_errors()`
- `invalid_block_ids`
- `recompute`
- `num_computed_tokens` 回退

读这部分时关注失败路径：

```text
backend.get() 部分失败
  -> worker 记录 failed block ids
  -> KVConnectorOutput.invalid_block_ids
  -> scheduler/model runner 根据 policy fail 或 recompute
  -> recompute 时回退 num_computed_tokens 并重新调度
```

## 12. 读测试来验证理解

优先读这些单测：

- `tests/ut/distributed/ascend_store/test_config_data.py`
- `tests/ut/distributed/ascend_store/test_backend.py`
- `tests/ut/distributed/ascend_store/test_pool_scheduler.py`
- `tests/ut/distributed/ascend_store/test_pool_worker.py`
- `tests/ut/distributed/ascend_store/test_kv_transfer.py`
- `tests/ut/distributed/ascend_store/test_ascend_store_connector.py`
- `tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py`

读测试的顺序：

1. `test_config_data.py`：确认 key 和 metadata 如何生成。
2. `test_backend.py`：确认后端接口契约。
3. `test_pool_scheduler.py`：确认命中、tracker、metadata 构造。
4. `test_pool_worker.py`：确认 worker 如何注册 KV cache、投递任务。
5. `test_kv_transfer.py`：确认 transfer thread 的 get/put 行为。
6. `test_ascend_store_connector.py`：确认 connector 对外生命周期。
7. failure tests：确认异常和 recompute。

测试是很好的“断言版文档”。看源码卡住时，先搜对应函数在测试里怎么构造输入。

## 13. 一条完整请求的调用链

### 有 KV Pool 命中的 prefill/load 路径

```text
HTTP/OpenAI request
  -> vLLM engine 创建 Request
  -> Scheduler 调度新请求
    -> KVCacheManager 查本地 prefix cache
    -> AscendStoreConnector.get_num_new_matched_tokens()
      -> KVPoolScheduler 查询外部 KV Pool 命中
    -> KVCacheManager.allocate_slots(... num_external_computed_tokens ...)
    -> AscendStoreConnector.update_state_after_alloc()
      -> RequestTracker 记录 block ids / token len
    -> AscendStoreConnector.build_connector_meta()
      -> AscendConnectorMetadata(requests=[ReqMeta(load_spec=...)])
  -> Worker 收到 SchedulerOutput 和 connector metadata
    -> pre_forward/start_load_kv()
      -> KVPoolWorker.start_load_kv()
      -> recv thread backend.get(...)
    -> forward
    -> post_forward/get_finished()
  -> Scheduler.update_connector_output()
```

### 保存 KV 到 pool 的路径

```text
请求被调度并完成一段 prefill/decode
  -> build_connector_meta()
    -> ReqMeta(can_save=True, save_start_token, save_end_token)
  -> Worker forward
  -> 非 layerwise: wait_for_save()
       -> send thread backend.put(...)
  -> layerwise: 每层 save_kv_layer()
       -> layer send thread backend.put(...)
  -> get_finished()
  -> Scheduler.update_connector_output()
  -> 请求结束时 request_finished()/request_finished_all_groups()
```

### lookup 路径

```text
Scheduler KVPoolScheduler.get_num_new_matched_tokens()
  -> LookupKeyClient.lookup(token_len, block_hashes, group_ids)
  -> ZMQ
  -> Worker LookupKeyServer
  -> KVPoolWorker.lookup_scheduler()
  -> backend.exists(keys) 或 batch_get_key_info(keys)
  -> 返回最长命中 token 数
```

## 14. 推荐实际走读顺序

第一轮只看骨架：

1. `docs/source/user_guide/feature_guide/kv_pool.md`
2. `vllm_ascend/distributed/kv_transfer/__init__.py`
3. `ascend_store_connector.py`
4. `vllm/v1/core/sched/scheduler.py` 中 connector 调用点
5. `vllm/v1/worker/gpu/kv_connector.py`

第二轮看数据如何流动：

1. `config_data.py`
2. `pool_scheduler.py`
3. `pool_worker.py`
4. `kv_transfer.py`

第三轮看存储后端：

1. `backend/backend.py`
2. `backend/mooncake_backend.py`
3. 如果开 layerwise，再看 `backend/memcache_backend.py`
4. 如使用 Yuanrong，再看 `backend/yuanrong_backend.py`

第四轮看与模型前向的结合：

1. `vllm_ascend/attention/utils.py`
2. `vllm_ascend/attention/sfa_v1.py`
3. `vllm_ascend/attention/mla_v1.py`

第五轮看异常、回退和测试：

1. `vllm_ascend/core/recompute_scheduler.py`
2. `tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py`
3. `tests/ut/distributed/ascend_store/*.py`

## 15. 走读时建议记录的几个问题

每读完一个阶段，建议在旁边记答案：

- 当前对象运行在 scheduler 进程还是 worker 进程？
- 当前方法处理的是“查命中”、“分配后更新”、“构造元数据”、“加载”、“保存”还是“结束清理”？
- 输入里的 token 数、block ids、block hashes 分别来自哪里？
- 这个阶段是否区分 `kv_producer`、`kv_consumer`、`kv_both`？
- 普通模式和 layerwise 模式在这里是否分叉？
- 如果后端 key 不存在或 get 失败，会走 fail 还是 recompute？
- 多 KV cache group、PP、TP、PCP、DCP 是否会改变 key 或地址计算？

## 16. 常用搜索命令

```bash
rg -n "AscendStoreConnector|KVPoolScheduler|KVPoolWorker|ReqMeta|RequestTracker" vllm-ascend/vllm_ascend
rg -n "get_num_new_matched_tokens|update_state_after_alloc|build_connector_meta|request_finished" vllm vllm-ascend
rg -n "start_load_kv|wait_for_save|save_kv_layer|wait_for_layer_load" vllm vllm-ascend
rg -n "lookup_scheduler|LookupKeyClient|LookupKeyServer|get_zmq_rpc_path_lookup" vllm-ascend/vllm_ascend
rg -n "batch_get_key_info|batch_alloc|register_buffer|backend.get|backend.put" vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store
```

## 17. 最小心智模型

把 KV Pool 看成一张外部哈希表：

```text
PoolKey(token block hash + model/parallel/cache metadata)
  -> 外部后端中某段 KV cache 数据
```

scheduler 负责问：“这些 key 有没有？有多少连续前缀可用？”

worker 负责做：“把这些 key 对应的数据搬到我的 KV cache block 地址里；或者把我的 KV cache block 地址里的数据存成这些 key。”

layerwise 负责优化：“不要等所有层都搬完；每层需要时等该层，算完该层就尽快保存该层。”

