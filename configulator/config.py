from ml_collections import ConfigDict

def default_config():
    cfg = ConfigDict()
    cfg.name = 'rag_1'
    cfg.verbose = False
    cfg.mode = 'valid'

    # text splitter
    cfg.chunksize = 200
    cfg.overlap = 0.4

    return cfg