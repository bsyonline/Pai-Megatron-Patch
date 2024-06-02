class ThroughputCalculator:
    """Throughput rate calculator with tflops and tokens processing rate metrics
    """

    def __init__(self, args):

        # 获取训练配置
        layers_num = args.num_layers
        hidden_size = args.hidden_size
        sequence_length = args.max_position_embeddings
        vocab_size = args.padded_vocab_size
        batch_size = args.global_batch_size
        ffn_hidden_size = args.ffn_hidden_size
        recompute_granularity = args.recompute_granularity

        # 单个transformer layer的flops计算
        qkv_flops = 6 * batch_size * sequence_length * hidden_size ** 2  # qkv transformation
        attn_weighting_flops = 4 * batch_size * sequence_length ** 2 * hidden_size  # attention matrix computation and attention over values
        post_proj_flops = 2 * batch_size * sequence_length * hidden_size ** 2  # post-attention linear projection
        # feed-forward network computation
        if args.swiglu:
            up_proj_flops = 4 * batch_size * hidden_size * sequence_length * ffn_hidden_size
            down_proj_folps = 4 * batch_size * hidden_size * sequence_length * ffn_hidden_size
            ffn_flops = up_proj_flops + down_proj_folps
        else:
            ffn_flops = 16 * batch_size * sequence_length * hidden_size ** 2

        if recompute_granularity is None:
            # one forwar dpass, two backward pass
            transformer_layer_flops = 3 * (qkv_flops + attn_weighting_flops + post_proj_flops + ffn_flops)
        elif recompute_granularity == "full":
            # two forwar dpass, two backward pass
            transformer_layer_flops = 4 * (qkv_flops + attn_weighting_flops + post_proj_flops + ffn_flops)
        else:
            # one forward pass but with extra attention recomputation, two backward pass
            transformer_layer_flops = 4 * (qkv_flops + attn_weighting_flops) + 3 * (post_proj_flops + ffn_flops)

        # llm head的flops计算
        llm_head_flops = 6 * batch_size * sequence_length * hidden_size * vocab_size
        # 计算所有flops
        self.overall_tflops = (layers_num * transformer_layer_flops + llm_head_flops) / 1e12

        # 计算token量
        self.overall_tokens = sequence_length * batch_size

    def get_tflops_rate(self, step_time):
        """
        使用浮点计算速率指标，衡量单次迭代的吞吐速率
        Args:
            args: 训练参数
            step_time: 单次迭代的耗时
        Return:
            flops_rate: 吞吐率(TFlops/s/GPU)
        """
        return self.overall_tflops / (step_time + 1e-12)

    def get_tokens_rate(self, step_time, world_size=1):
        """
        使用单个GPU上的token处理速率指标, 衡量单次迭代的吞吐速率
        Args:
            args: 训练参数
            step_time: 单次迭代的耗时
        Return:
            tokens_rate: 吞吐率(Tokens/s)
        """
        return self.overall_tokens / (step_time + 1e-12) / world_size