def get_model(cfg, logger):
    if cfg.arch == 'InvEDRS_arb':
        from models.inv_arb_edrs import InvArbEDRS as Model
        model = Model(cfg)
    elif cfg.arch == 'InvEDRS_loop3':
        from models.inv_arb_edrs import InvArbEDRS_3loop as Model
        model = Model(cfg)
    else:
        raise Exception('architecture not supported yet'.format(cfg.arch))
    return model
