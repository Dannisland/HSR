def get_model(cfg, logger):
    if cfg.arch == 'InvEDRS_arb':
        from models.inv_arb_edrs_loop3 import InvArbEDRS as Model
        model = Model(cfg)     
    else:
        raise Exception('architecture not supported yet'.format(cfg.arch))
    return model

def get_model_msg(cfg, logger):
    if cfg.arch == 'InvEDRS_arb':
        from models.inv_arb_edrs_msg import InvArbEDRS as Model
        model = Model(cfg)
    else:
        raise Exception('architecture not supported yet'.format(cfg.arch))
    return model
