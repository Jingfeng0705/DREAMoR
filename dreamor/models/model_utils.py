import numpy as np    
def step(model, loss_func, data, dataset, device, cur_epoch, mode='train', use_gt_p=1.0):
    '''
    Given data for the current training step (batch),
    pulls out the necessary needed data,
    runs the model,
    calculates and returns the loss.

    - use_gt_p : the probability of using ground truth as input to each step rather than the model's own prediction
                 (1.0 is fully supervised, 0.0 is fully autoregressive)
    '''
    use_sched_samp = use_gt_p < 1.0
    batch_in, batch_out, meta = data

    prep_data = model.prepare_input(batch_in, device, data_out=batch_out, return_input_dict=True, return_global_dict=use_sched_samp)
    if use_sched_samp:
        x_past, x_t, gt_dict, input_dict, global_gt_dict = prep_data
    else:
        x_past, x_t, gt_dict, input_dict = prep_data

    B, T, S_in, _ = x_past.size()
    S_out = x_t.size(2)

    if not use_sched_samp:
        # fully supervised phase
        # start by using gt at every step, so just form all steps from all sequences into one large batch
        #       and get per-step predictions
        x_past_batched = x_past.reshape((B*T, S_in, -1))
        x_t_batched = x_t.reshape((B*T, S_out, -1))
        out_dict = model(x_past_batched, x_t_batched)
    else:
        # in scheduled sampling or fully autoregressive phase
        init_input_dict = dict()
        for k in input_dict.keys():
            init_input_dict[k] = input_dict[k][:,0,:,:] # only need first step for init
        # this out_dict is the global state
        sched_samp_out = model.scheduled_sampling(x_past, x_t, init_input_dict, p=use_gt_p, 
                                                                    gender=meta['gender'],
                                                                    betas=meta['betas'].to(device),
                                                                    need_global_out=(not model.detach_sched_samp))
        if model.detach_sched_samp:
            out_dict = sched_samp_out
        else:
            out_dict, _ = sched_samp_out
        # gt must be global state for supervision in this case
        if not model.detach_sched_samp:
            print('USING global supervision')
            gt_dict = global_gt_dict

    # loss can be computed per output step in parallel
    # batch dicts accordingly
    for k in out_dict.keys():
        if k == 'posterior_distrib' or k == 'prior_distrib':
            m, v = out_dict[k]
            m = m.reshape((B*T, -1))
            v = v.reshape((B*T, -1))
            out_dict[k] = (m, v)
        else:
            out_dict[k] = out_dict[k].reshape((B*T*S_out, -1))
    for k in gt_dict.keys():
        gt_dict[k] = gt_dict[k].reshape((B*T*S_out, -1))

    gender_in = np.broadcast_to(np.array(meta['gender']).reshape((B, 1, 1, 1)), (B, T, S_out, 1))
    gender_in = gender_in.reshape((B*T*S_out, 1))
    betas_in = meta['betas'].reshape((B, T, 1, -1)).expand((B, T, S_out, 16)).to(device)
    betas_in = betas_in.reshape((B*T*S_out, 16))
    loss, stats_dict = loss_func(out_dict, gt_dict, cur_epoch, gender=gender_in, betas=betas_in)

    return loss, stats_dict