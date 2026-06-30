{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",

  "zero_optimization": {
    "stage": 3,
    "overlap_comm": false,
    "contiguous_gradients": true,

    "reduce_bucket_size": 10000000,
    "stage3_prefetch_bucket_size": 10000000,
    "stage3_param_persistence_threshold": 10000,

    "stage3_max_live_parameters": 100000000,
    "stage3_max_reuse_distance": 100000000,

    "stage3_gather_16bit_weights_on_model_save": false
  },

  "fp16": {
    "enabled": "auto",
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 12,
    "hysteresis": 2,
    "min_loss_scale": 1
  },

  "bf16": {
    "enabled": false
  },

  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true,
    "synchronize_checkpoint_boundary": false,
    "profile": false
  },

  "wall_clock_breakdown": false
}
