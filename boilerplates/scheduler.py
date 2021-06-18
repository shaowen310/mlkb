import transformers


def build_optimizer_and_scheduler(model, init_lr, n_train_steps, warmup_proportion):
    no_decay = ['bias', 'layer_norm', 'LayerNorm']  # no decay for parameters of layer norm and bias
    param_groups = [{
        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay':
        0.01
    }, {
        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]
    optimizer = transformers.AdamW(param_groups, lr=init_lr)
    n_warmup_steps = int(n_train_steps * warmup_proportion)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=n_warmup_steps,
                                                             num_training_steps=n_train_steps)
    return optimizer, scheduler
