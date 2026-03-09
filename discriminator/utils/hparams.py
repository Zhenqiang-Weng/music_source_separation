
import os
import yaml


def load_hparam_str(hp_str):
    path = 'temp-restore.yaml'
    with open(path, 'w') as f:
        f.write(hp_str)
    ret = HParam(path)
    os.remove(path)
    return ret


def load_hparam(filename):
    stream = open(filename, 'r', encoding='utf-8')
    docs = yaml.load_all(stream, Loader=yaml.Loader)
    hparam_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            hparam_dict[k] = v
    return hparam_dict


def merge_dict(user, default):
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_dict(user[k], v)
    return user


def parse_dict(user):
    assert isinstance(user, dict), "Invalid input"
    hp_dct = dict()
    for key, value in user.items():
        ks = key.split('.')[::-1]
        for k in ks:
            tmp_dct = dict()
            tmp_dct[k] = value
            value = tmp_dct

        hp_dct.update(tmp_dct)
    return hp_dct



class Dotdict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value


class HParams(Dotdict):
    def __init__(self, file):
        super(Dotdict, self).__init__()
        self.hp_dict =load_hparam(file)
        self.initialize()

    def initialize(self):
        hp_dotdict = Dotdict(self.hp_dict)
        for key, value in hp_dotdict.items():
            setattr(self, key, value)
    def	parse_and_update(self, dct):
        tmp_dict = self.hp_dict
        self.hp_dict = merge_dict(parse_dict(dct),tmp_dict)
        self.initialize()

    def	save(self, path):
        with open(os.path.join(path, 'config.yaml'), 'w') as f:
            yaml.dump(self.hp_dict,f,encoding='utf-8',allow_unicode=True)
    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__


if	__name__ == "__main_":
    hparams = HParams(os.path.join('..', 'config', 'default.yaml'))
    with open(os.path.join('..', 'data', 'english.list'), 'r') as fp:
        symbols = fp.read().strip().split('\n')
    text_hparams, model_hparams = hparams.text, hparams.model
    text_hparams.symbols = symbols
    text_hparams.symbols = \
        text_hparams.specials + text_hparams.symbols + text_hparams.punctuations
    model_hparams.n_symbols = len(text_hparams.symbols)
    print(text_hparams)
    print(model_hparams)
    dct = {'text.symbols': symbols, 'model.n_symbols': len(symbols)}
    hparams.parse_and_update(dct)
    print(hparams)
