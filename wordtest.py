import logging
import coloredlogs
import pickle

logger = logging.getLogger('__file__')
coloredlogs.install(level='INFO', logger=logger)

def pickle_io(path, mode='r', obj=None):
    """
    Convinient pickle load and dump.
    """
    if mode in ['rb', 'r']:
        logger.info("Loading obj from {}...".format(path))
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        logger.info("Load obj successfully!")
        return obj
    elif mode in ['wb', 'w']:
        logger.info("Dumping obj to {}...".format(path))
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info("Dump obj successfully!")

class WordTestResource(object):

    def __init__(self, resource_path, verbose=False):

        resource = pickle_io(resource_path, mode='r')

        self.id2index = resource['id2index']
        self.index2id = resource['index2id']
        self.num_skills = len(self.id2index)

        if verbose:
            self.word2id = resource['word2id']
            self.id2all = resource['id2all']
            # rank0 already be set to a large number
            self.words_by_rank = resource['words_by_rank']
            self.pos2id = resource['pos2id']
            self.words_by_rank.sort(key=lambda x: x[u'rank'])
            self.id_by_rank = [x[u'word_id'] for x in self.words_by_rank]

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'