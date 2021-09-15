import torch
from torch.autograd import Function

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1) # 一共有512个编码, 每一个编码256维度.
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)# 拍平算的快.一次性矩阵运算完毕.

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
# 矩阵范数:https://blog.csdn.net/zaishuiyifangxym/article/details/81673491
            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)  # beta*first+alpha(second@third),   alpha(second@third)的范数 等价于 2 codebook_sqr**0.5*inputs_sqr**0.5 用均值不等式. 这个小于codebook_sqr + inputs_sqr .所以保证了距离大于0这个性质.

            _, indices_flatten = torch.min(distances, dim=1) #距离最近的索引就是我们要的.
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices) # 通过autograd.Function的返回值，看来是会被设为 requires_grad的，除非在forward内设定ctx.mark_non_differentiable(out)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook) # 调用vq的forward方法.
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook) # save_for_backward就会把数据传递给backward方法了.
        ctx.mark_non_differentiable(indices_flatten)  # ctx 的相关学习:https://pytorch.org/docs/stable/autograd.html?highlight=ctx

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]: # 对索引进行求梯度.
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st]
