#! /usr/bin/env python3
import torch
from torch import nn
from torch.autograd import grad

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


def s_test(test_input_ids, test_att_masks, test_label_ids, model, z_loader, gpu=0, damp=0.01, scale=25.0,
           recursion_depth=5000):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test"""
    v = grad_z(test_input_ids, test_att_masks, test_label_ids, model, gpu) # list[50].the list of gradients of each parameter to loss
    # #########################
    # # TODO: why requires_grad. Some type of parameter needs gradient
    # #########################
    # params = [p for p in model.parameters() if p.requires_grad]
    # v = list(grad(loss, params, create_graph=True))
    h_estimate = v.copy()  # Initial Inverse-HVP estimate

    ################################
    # TODO: Dynamically set the recursion depth so that iterations stops
    # once h_estimate stabilises
    ################################
    for i in range(recursion_depth):
        # take just one random sample from training dataset
        # easiest way to just use the DataLoader once, break at the end of loop
        #########################
        # TODO:  is it to choose training data randomly every time
        #########################
        for train_input_ids, train_att_masks, train_label_ids in z_loader:
            #if gpu >= 0:
            train_input_ids, train_att_masks, train_label_ids = train_input_ids.to(device), train_att_masks.to(
                    device), train_label_ids.to(device)

            outputs = model(input_ids=train_input_ids, attention_mask=train_att_masks)
            y = outputs.logits
            # print(y.shape)
            # print(train_label_ids.shape)
            loss = calc_loss(y.view(-1, model.config.vocab_size), train_label_ids.view(-1)) #
            params = [p for p in model.parameters() if p.requires_grad]
            hv = hvp(loss, params, h_estimate) # hvp based on new random training data.it returns list of torch tensors,
            # contains product of Hessian and v. H −1 j −1, ˆ θv

            # Recursively calculate h_estimate
            #print("hv computed") 
            #########################
            # TODO:  why it's not the same equation as in that paper
            #########################
            with torch.no_grad():# don't track the grad in calculation
                # _v: vector in initial gradient.
                #  (1 - damp) * _h_e applies a damping factor (damp) to adjust the influence of the previous estimate
                #  to ensure numerical stability and control the magnitude of updates. The damping term helps prevent
                #  numerical explosion during the iteration process. The term "-_hv / scale" subtracts the current
                #  Hessian vector product (considering the current h_estimate) adjusted by scale from the update,
                #  where scale is a scaling factor used to control the step size of each iteration update.
                #
                # (1 - damp) * _h_e应用了一个阻尼因子（damp）来调节前一次估计的影响，以确保数值稳 定性并控制更新幅度。阻尼项有助于防止迭代过程中的数值爆炸。
                # -_hv / scale将当前的Hessian向量积（考虑了当前h_estimate）按比例调整并从更新中减去，其中scale是一个缩放因子，用于控制每次迭代更新的步长。
                h_estimate = [ # list of tensor [50]
                    _v + (1 - damp) * _h_e - _hv / scale
                    for _v, _h_e, _hv in zip(v, h_estimate, hv)]
                break #这样做是每次都随机生成一次training data
        # display_progress("Calc. s_test recursions: ", i, recursion_depth)

    # print("h_estimate: ", h_estimate)
    return h_estimate

def calc_loss(y, t):
    """Calculates the loss

    Arguments:
        y: torch tensor, input with size
        t: torch tensor, target expected by loss of size

    Returns:
        loss: scalar, the loss"""
    criterion = nn.NLLLoss()
    loss = criterion(y, t)
    return loss


def grad_z(test_input_ids, test_att_masks, test_label_ids, model, gpu=0):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        src_list: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        output_trg_list: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU

    Returns:
        grad_z: list of torch tensor, containing the gradients
            from each parameter to loss"""
    model.eval()
    # initialize
    #if gpu >= 0:
    test_input_ids, test_att_masks, test_label_ids = (test_input_ids.to(device), test_att_masks.to(device),
                                                                     test_label_ids.to(device))

    y = model(input_ids=test_input_ids, attention_mask=test_att_masks) # y tensor [batch size,sequence length, Vocabulary length][1,200,3208].
    #output_ids = model.generate(input_ids=test_input_ids, max_length=10, num_return_sequences=1, top_k=10, top_p=0.75, attention_mask=attention_mask) 
    #logits=outputs[0][:, -1, :] 
    logits = y.logits
    # print(f'size of y: {y.size()}')
    # print(f'test_label_ids size is:{test_label_ids.size()}')
    loss = calc_loss(logits.view(-1, model.config.vocab_size), test_label_ids.view(-1))
    #print(loss)
    # print(f'size of y.view: {y.view(-1, sp_vocab_size).size()}')
    # print(f'output_trg_list.view size is:{ output_trg_list.view(-1).size()}')
    #########################
    # TODO: why requires_grad. Some type of parameter needs gradient
    #########################
    params = [p for p in model.parameters() if p.requires_grad] # 在深度学习中，模型的损失函数通常是关于模型参数的复合函数，
    # 链式法则使我们能够计算出损失函数相对于任何参数的梯度（导数）。是在计算损失函数（loss）对于params列表中每个参数（parameter）的梯度。
    # Compute gradients list from model parameters to loss. One select p that need grad
    return list(grad(loss, params, create_graph=True))


def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed, like parameters
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian, h_estimate, result of grad_z

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(w) != len(v):
        raise (ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True) # f反向传播用于计算损失函数相对于模型参数（如权重和偏置）的梯度

# Hessian矩阵是损失函数相对于模型参数的二阶导数构成的矩阵。给定向量v，Hessian向量积Hv描述了当参数沿向量v方向微小变化时，梯度（一阶导数）的变化。
    # 直观上，它提供了关于损失曲面局部形状的信息。
    # Elementwise products
    elemwise_products = 0 #在前面的步骤中，通过第一次反向传播计算了损失函数y相对于参数w的一阶导数（梯度）。接着，将这些梯度与向量v进行元素乘
    # 积并求和，得到一个标量elemwise_products。这个标量是梯度与v的点积，反映了梯度在v方向上的投影。
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

 #接下来，对elemwise_products进行反向传播，即计算这个标量相对于模型参数w的梯度，实际上通过以下步骤实现了Hessian向量积Hv的计算：链式法则：
    # 根据链式法则，elemwise_products相对于w的导数（梯度）包含了两部分信息：原始梯度如何随w变化（即Hessian矩阵），以及这种变化如何沿着向量v
    # 方向累积（即向量积）。
    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads
