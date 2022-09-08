import torch

HOOK_MODULES = ["encoder", "contextual", "predictive"]
HOOK_TYPES = ["activations", "errors"]


#############################################
def init_layers_hook_dict(model):
    """
    init_layers_hook_dict(model)

    Initialized a dictionary in which to collect hook data.

    Required args
    -------------
    - model : torch nn.Module
        Model.

    Returns
    -------
    - hook_dict : dict
        Dictionary for collecting hook data, with keys:
        for hook_module in ["contextual", "encoder", "predictive"]
            for hook_type in ["activations", "errors"]
                '{hook_module}''{hook_type}' (list): 
                    Empty list for collecting data across batches.
    - count_dict : dict
        Dictionary for collecting hook data, with keys:
        for hook_module in ["contextual", "encoder", "predictive"]
            for hook_type in ["activations", "errors"]
                '{hook_module}''{hook_type}' (int): 
                    Zero value, to start counting number of outputs collected 
                    per batch.
    - layer_dict : list
        Module layers 
    """

    layer_dict = {
        "encoder"   : model.avg_pool, # occurs after the backbone module
        "contextual": model.agg.cell_list[0].out_gate,
        "predictive":  model.network_pred[2],
    }

    hook_dict = dict()
    count_dict = dict()
    for hook_module in HOOK_MODULES:
        hook_dict[hook_module] = {hook_type: [] for hook_type in HOOK_TYPES}
        count_dict[hook_module] = {hook_type: 0 for hook_type in HOOK_TYPES}

    return hook_dict, count_dict, layer_dict


#############################################
def reset_count_dict(count_dict):
    """
    reset_count_dict(count_dict)

    Resets values in the count dictionary to zero, in place.

    Required args
    -------------
    - count_dict : dict
        Dictionary for collecting hook data, with keys:
        for hook_module in ["contextual", "encoder", "predictive"]
            for hook_type in ["activations", "errors"]
                '{hook_module}''{hook_type}' (int): 
                    Number of outputs collected per batch.
    """

    for hook_module in count_dict.keys():
        for hook_type in count_dict[hook_module].keys():
            count_dict[hook_module][hook_type] = 0

    return


#############################################
def extract_output(output, batch_size=32, pred_step=2, hook_module="encoder", 
                   count=0):
    """
    extract_output(output)

    Extracts output activations or error gradient information.

    Required args
    -------------
    - output : list or 4-5D tensor
        Tensor or list of tensor (with only 1 tensor) with output activations 
        or error gradients for the specified layer. 
        If 'hook_module' is 'encoder', tensor has dims: B * N x C x 1 x D x D
        Otherwise, tensor has dims: B x C x D x D

    Optional args
    -------------
    - batch_size : int
    - pred_step : int
    - hook_module : str
    - count : int
    """

    if hook_module == "contextual" and count >= pred_step:
        return None

    if isinstance(output, tuple):
        if len(output) == 1:
            output = output[0]
        else:
            raise ValueError("Expected tuple of length 1.")

    output = output.detach()

    if hook_module == "encoder":
        B_N, C, _, D, _ = output.shape # ignore unary dimension L, and second D
        N = B_N // batch_size
        output = output.view(batch_size, N, C, D, D)[:, -pred_step :]

    return output


#############################################
def init_hook_dict(model, batch_size):
    """
    init_hook_dict(model, batch_size)

    Required args
    -------------
    - model : torch nn.Module
        Model.
    - batch_size : int
        Number of sequences per batch.

    Returns
    -------
    - hook_dict : dict
        Dictionary for collecting hook data, with keys:
        for hook_module in ["contextual", "encoder", "predictive"]
            for hook_type in ["activations", "errors"]
                '{hook_module}''{hook_type}' (list): 
                    Empty list for collecting data across batches.
    - activation_hooks : list
        Forward hook for each module.
    - error_hooks : list
        Backward hook for each module.
    """

    hook_dict, count_dict, layer_dict = init_layers_hook_dict(model)
    pred_step = model.pred_step

    # register forward and backward hooks
    def get_hook(hook_module="encoder", hook_type="activations"):
        """
        get_hook()

        Returns hook for the specified module and type.

        Optional args
        -------------
        - hook_module : str (default="encoder")
            Module type, i.e., 'contextual', 'encoder', 'predictive'.
        - hook_type : str (default="activations")
            Type of hook data, i.e., 'activations' and 'errors'.

        Returns
        -------
        - hook : function
            Hook data collection function.
        """

        def hook(layer, inp, output):
            """
            hook(layer, inp, output)

            Hook data collection function

            Required args
            -------------
            - layer : nn.Module
                Layer to which the hook corresponds (ignored).
            - inp : list or nd Tensor
                Activations or gradients of the input to the layer (ignored).
            - output : list or nd Tensor
                Activations or gradients of the output of the layer
            """

            subdict = hook_dict[hook_module][hook_type]

            count = count_dict[hook_module][hook_type]
            output = extract_output(
                output, batch_size, pred_step=pred_step, 
                hook_module=hook_module, count=count
            )

            if output is None:
                return

            subdict.append(output.cpu())

            # keep only for predicted steps
            if hook_module in ["contextual", "predictive"] and count == pred_step - 1:
                subdict[-pred_step] = torch.stack(subdict[-pred_step:], 1)
                if pred_step > 1:
                    del hook_dict[hook_module][hook_type][-pred_step + 1:]
            
            # count for modules visited more than once
            count_dict[hook_module][hook_type] += 1
            if hook_module == "encoder":
                reset_count_dict(count_dict)

        return hook


    activation_hooks = [
        layer_dict[hook_module].register_forward_hook(
            get_hook(hook_module, "activations")
            ) for hook_module in HOOK_MODULES
        ]

    error_hooks = [
        layer_dict[hook_module].register_full_backward_hook(
            get_hook(hook_module, "errors")
            ) for hook_module in HOOK_MODULES
        ]
    
    return hook_dict, activation_hooks, error_hooks


