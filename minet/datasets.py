# -*- coding: utf 8 -*-
from __future__ import division
try:
    import cPickle
except ImportError:
    import pickle as cPickle
import gzip
import tarfile
import os
from scipy.io import wavfile
import numpy as np
import tables
import numbers
import glob
import random
import string
import fnmatch
import theano
from matplotlib.pyplot import specgram

def load_mnist():
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = 'mnist.pkl.gz'
    data_file = os.path.join(data_path, dataset)
    if os.path.isfile(data_file):
        dataset = data_file

    if (not os.path.isfile(data_file)):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print('Downloading data from %s' % url)
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = cPickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32')
    train_y = train_y.astype('int32')

    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval


def load_cifar10():
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = 'cifar-10-python.tar.gz'
    data_file = os.path.join(data_path, dataset)
    if os.path.isfile(data_file):
        dataset = data_file

    if (not os.path.isfile(data_file)):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        print('Downloading data from %s' % url)
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    tar = tarfile.open(data_file)
    os.chdir(data_path)
    tar.extractall()
    tar.close()

    data_path = os.path.join(data_path, "cifar-10-batches-py")
    batch_files = glob.glob(os.path.join(data_path, "*batch*"))
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for f in batch_files:
        batch_file = open(f, 'rb')
        d = cPickle.load(batch_file)
        batch_file.close()
        fname = f.split(os.path.sep)[-1]
        if "data" in fname:
            data = d['data']
            labels = d['labels']
            train_data.append(data)
            train_labels.append(labels)
        elif "test" in fname:
            data = d['data']
            labels = d['labels']
            test_data.append(data)
            test_labels.append(labels)

    # Split into 40000 train 10000 valid 10000 test
    train_x = np.asarray(train_data)
    train_y = np.asarray(train_labels)
    test_x = np.asarray(test_data)
    test_y = np.asarray(test_labels)
    valid_x = train_x[-10000:]
    valid_y = train_y[-10000:]
    train_x = train_x[:-10000]
    train_y = train_y[:-10000]

    test_x = test_x.astype('float32')
    test_y = test_y.astype('int32')
    valid_x = valid_x.astype('float32')
    valid_y = valid_y.astype('int32')
    train_x = train_x.astype('float32')
    train_y = train_y.astype('int32')

    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval


def load_scribe():
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = 'scribe.pkl'
    data_file = os.path.join(data_path, dataset)
    if os.path.isfile(data_file):
        dataset = data_file

    if (not os.path.isfile(data_file)):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
            url = 'https://dl.dropboxusercontent.com/u/15378192/scribe2.pkl'
        except AttributeError:
            import urllib.request as urllib
            url = 'https://dl.dropboxusercontent.com/u/15378192/scribe3.pkl'
        print('Downloading data from %s' % url)
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    with open(data_file, 'rb') as pkl_file:
        data = cPickle.load(pkl_file)

    data_x, data_y = [], []
    for x, y in zip(data['x'], data['y']):
        data_y.append(np.asarray(y, dtype=np.int32))
        data_x.append(np.asarray(x, dtype=theano.config.floatX).T)

    train_x = data_x[:750]
    train_y = data_y[:750]
    valid_x = data_x[750:900]
    valid_y = data_y[750:900]
    test_x = data_x[900:]
    test_y = data_y[900:]
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval


# A tricky trick for monkeypatching an instancemethod that is
# CPython :( there must be a better way
class _cVLArray(tables.VLArray):
    pass


def load_fruitspeech():
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = 'audio.tar.gz'
    data_file = os.path.join(data_path, dataset)
    if os.path.isfile(data_file):
        dataset = data_file

    if not os.path.isfile(data_file):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
            url = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
        except AttributeError:
            import urllib.request as urllib
            url = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
        print('Downloading data from %s' % url)
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    if not os.path.exists(os.path.join(data_path, "audio")):
        tar = tarfile.open(data_file)
        os.chdir(data_path)
        tar.extractall()
        tar.close()

    h5_file_path = os.path.join(data_path, "saved_fruit.h5")
    if not os.path.exists(h5_file_path):
        data_path = os.path.join(data_path, "audio")

        audio_matches = []
        for root, dirnames, filenames in os.walk(data_path):
            for filename in fnmatch.filter(filenames, '*.wav'):
                audio_matches.append(os.path.join(root, filename))

        random.seed(1999)
        random.shuffle(audio_matches)

        # http://mail.scipy.org/pipermail/numpy-discussion/2011-March/055219.html
        h5_file = tables.openFile(h5_file_path, mode='w')
        data_x = h5_file.createVLArray(h5_file.root, 'data_x',
                                       tables.Float32Atom(shape=()),
                                       filters=tables.Filters(1))
        data_x_shapes = h5_file.createVLArray(h5_file.root, 'data_x_shapes',
                                              tables.Int32Atom(shape=()),
                                              filters=tables.Filters(1))
        data_y = h5_file.createVLArray(h5_file.root, 'data_y',
                                       tables.Int32Atom(shape=()),
                                       filters=tables.Filters(1))
        for wav_path in audio_matches:
            # Convert chars to int classes
            word = wav_path.split(os.sep)[-1][:-6]
            chars = [ord(c) - 97 for c in word]
            data_y.append(np.array(chars, dtype='int32'))
            fs, d = wavfile.read(wav_path)
            # Preprocessing from A. Graves "Towards End-to-End Speech
            # Recognition"
            Pxx, _, _, _ = specgram(d, NFFT=256, noverlap=128)
            data_x_shapes.append(np.array(Pxx.T.shape, dtype='int32'))
            data_x.append(Pxx.T.astype('float32').flatten())
        h5_file.close()

    h5_file = tables.openFile(h5_file_path, mode='r')
    data_x = h5_file.root.data_x
    data_x_shapes = h5_file.root.data_x_shapes
    data_y = h5_file.root.data_y
    # A dirty hack to only monkeypatch data_x
    data_x.__class__ = _cVLArray

    # override getter so that it gets reshaped to 2D when fetched
    old_getter = data_x.__getitem__

    def getter(self, key):
        if isinstance(key, numbers.Integral) or isinstance(key, np.integer):
            return old_getter(key).reshape(data_x_shapes[key]).astype(
                theano.config.floatX)
        elif isinstance(key, slice):
            start, stop, step = self._processRange(key.start, key.stop,
                                                   key.step)
            return [o.reshape(s) for o, s in zip(
                self.read(start, stop, step), data_x_shapes[slice(
                    start, stop, step)])]

    # Patch __getitem__ in custom subclass, applying to all instances of it
    _cVLArray.__getitem__ = getter

    train_x = data_x[:80]
    train_y = data_y[:80]
    valid_x = data_x[80:90]
    valid_y = data_y[80:90]
    test_x = data_x[90:]
    test_y = data_y[90:]
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval


def load_cmuarctic():
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    urls = ['http://www.speech.cs.cmu.edu/cmu_arctic/packed/cmu_us_awb_arctic-0.95-release.tar.bz2',
            'http://www.speech.cs.cmu.edu/cmu_arctic/packed/cmu_us_bdl_arctic-0.95-release.tar.bz2',
            'http://www.speech.cs.cmu.edu/cmu_arctic/packed/cmu_us_clb_arctic-0.95-release.tar.bz2',
            'http://www.speech.cs.cmu.edu/cmu_arctic/packed/cmu_us_jmk_arctic-0.95-release.tar.bz2',
            'http://www.speech.cs.cmu.edu/cmu_arctic/packed/cmu_us_ksp_arctic-0.95-release.tar.bz2',
            'http://www.speech.cs.cmu.edu/cmu_arctic/packed/cmu_us_rms_arctic-0.95-release.tar.bz2',
            'http://www.speech.cs.cmu.edu/cmu_arctic/packed/cmu_us_slt_arctic-0.95-release.tar.bz2',
            ]

    data_files = []

    for url in urls:
        dataset = url.split('/')[-1]
        data_file = os.path.join(data_path, dataset)
        data_files.append(data_file)
        if os.path.isfile(data_file):
            dataset = data_file
        if not os.path.isfile(data_file):
            try:
                import urllib
                urllib.urlretrieve('http://google.com')
            except AttributeError:
                import urllib.request as urllib
            print('Downloading data from %s' % url)
            urllib.urlretrieve(url, data_file)

    print('... loading data')

    folder_paths = []
    for data_file in data_files:
        folder_name = data_file.split(os.sep)[-1].split("-")[0]
        folder_path = os.path.join(data_path, folder_name)
        folder_paths.append(folder_path)
        if not os.path.exists(folder_path):
            tar = tarfile.open(data_file)
            os.chdir(data_path)
            tar.extractall()
            tar.close()

    h5_file_path = os.path.join(data_path, "saved_cmu.h5")
    if not os.path.exists(h5_file_path):
        # http://mail.scipy.org/pipermail/numpy-discussion/2011-March/055219.html
        h5_file = tables.openFile(h5_file_path, mode='w')
        data_x = h5_file.createVLArray(h5_file.root, 'data_x',
                                       tables.Float32Atom(shape=()),
                                       filters=tables.Filters(1))
        data_x_shapes = h5_file.createVLArray(h5_file.root, 'data_x_shapes',
                                              tables.Int32Atom(shape=()),
                                              filters=tables.Filters(1))
        data_y = h5_file.createVLArray(h5_file.root, 'data_y',
                                       tables.Int32Atom(shape=()),
                                       filters=tables.Filters(1))
        data_meta = h5_file.createVLArray(h5_file.root, 'data_meta',
                                          tables.StringAtom(200),
                                          filters=tables.Filters(1))
        for folder_path in folder_paths:
            audio_matches = []
            for root, dirnames, filenames in os.walk(folder_path):
                for filename in fnmatch.filter(filenames, '*.wav'):
                    audio_matches.append(os.path.join(root, filename))

            f = open(os.path.join(folder_path, "etc", "txt.done.data"))
            read_raw_text = f.readlines()
            f.close()
            # Remove all punctuations
            list_text = [t.strip().lower().translate(
                string.maketrans("", ""), string.punctuation).split(" ")[1:-1]
                for t in read_raw_text]
            # Get rid of numbers, even though it will probably hurt
            # recognition on certain examples
            cleaned_lookup = {lt[0]: " ".join(lt[1:]).translate(
                None, string.digits).strip() for lt in list_text}
            data_meta.append(folder_path.split(os.sep)[-1])

            for wav_path in audio_matches:
                lookup_key = wav_path.split(os.sep)[-1][:-4]
                # Some files aren't consistent!
                if "_" in cleaned_lookup.keys()[0] and "_" not in lookup_key:
                    # Needs an _ to match text format... sometimes!
                    lookup_key = lookup_key[:6] + "_" + lookup_key[6:]
                elif "_" not in cleaned_lookup.keys()[0]:
                    lookup_key = lookup_key.translate(None, "_")
                try:
                    words = cleaned_lookup[lookup_key]
                    # Convert chars to int classes
                    chars = [ord(c) - 97 for c in words]
                    # Make spaces last class
                    chars = [c if c >= 0 else 26 for c in chars]
                    data_y.append(np.array(chars, dtype='int32'))
                    # Convert chars to int classes
                    fs, d = wavfile.read(wav_path)
                    # Preprocessing from A. Graves "Towards End-to-End Speech
                    # Recognition"
                    Pxx, _, _, _ = plt.specgram(d, NFFT=256, noverlap=128)
                    data_x_shapes.append(np.array(Pxx.T.shape, dtype='int32'))
                    data_x.append(Pxx.T.astype('float32').flatten())
                except KeyError:
                    # Necessary because some labels are missing in some folders
                    print("Skipping %s due to missing key" % wav_path)

        h5_file.close()

    h5_file = tables.openFile(h5_file_path, mode='r')
    data_x = h5_file.root.data_x
    data_x_shapes = h5_file.root.data_x_shapes
    data_y = h5_file.root.data_y
    # A dirty hack to only monkeypatch data_x
    data_x.__class__ = _cVLArray

    # override getter so that it gets reshaped to 2D when fetched
    old_getter = data_x.__getitem__

    def getter(self, key):
        if isinstance(key, numbers.Integral) or isinstance(key, np.integer):
            return old_getter(key).reshape(data_x_shapes[key]).astype(
                theano.config.floatX)
        elif isinstance(key, slice):
            start, stop, step = self._processRange(key.start, key.stop,
                                                   key.step)
            return [o.reshape(s) for o, s in zip(
                self.read(start, stop, step), data_x_shapes[slice(
                    start, stop, step)])]

    # Patch __getitem__ in custom subclass, applying to all instances of it
    _cVLArray.__getitem__ = getter

    train_x = data_x[:6000]
    train_y = data_y[:6000]
    valid_x = data_x[6000:7500]
    valid_y = data_y[6000:7500]
    test_x = data_x[7500:]
    test_y = data_y[7500:]
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval


def load_librispeech():
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = 'dev-clean.tar.gz'
    data_file = os.path.join(data_path, dataset)
    if os.path.isfile(data_file):
        dataset = data_file

    if not os.path.isfile(data_file):
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
            url = 'http://www.openslr.org/resources/12/dev-clean.tar.gz'
        except AttributeError:
            import urllib.request as urllib
            url = 'http://www.openslr.org/resources/12/dev-clean.tar.gz'
        print('Downloading data from %s' % url)
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    if not os.path.exists(os.path.join(data_path, "LibriSpeech", "dev-clean")):
        tar = tarfile.open(data_file)
        os.chdir(data_path)
        tar.extractall()
        tar.close()

    h5_file_path = os.path.join(data_path, "saved_libri.h5")
    if not os.path.exists(h5_file_path):
        data_path = os.path.join(data_path, "LibriSpeech", "dev-clean")

        audio_matches = []
        for root, dirnames, filenames in os.walk(data_path):
            for filename in fnmatch.filter(filenames, '*.flac'):
                audio_matches.append(os.path.join(root, filename))

        text_matches = []
        for root, dirnames, filenames in os.walk(data_path):
            for filename in fnmatch.filter(filenames, '*.txt'):
                text_matches.append(os.path.join(root, filename))

        # http://mail.scipy.org/pipermail/numpy-discussion/2011-March/055219.html
        h5_file = tables.openFile(h5_file_path, mode='w')
        data_x = h5_file.createVLArray(h5_file.root, 'data_x',
                                       tables.Float32Atom(shape=()),
                                       filters=tables.Filters(1))
        data_x_shapes = h5_file.createVLArray(h5_file.root, 'data_x_shapes',
                                              tables.Int32Atom(shape=()),
                                              filters=tables.Filters(1))
        data_y = h5_file.createVLArray(h5_file.root, 'data_y',
                                       tables.Int32Atom(shape=()),
                                       filters=tables.Filters(1))
        for full_t in text_matches:
            f = open(full_t, 'r')
            for line in f.readlines():
                word_splits = line.strip().split(" ")
                file_tag = word_splits[0]
                words = word_splits[1:]
                # Convert chars to int classes
                chars = [ord(c) - 97 for c in (" ").join(words).lower()]
                # Make spaces last class
                chars = [c if c >= 0 else 26 for c in chars]
                data_y.append(np.array(chars, dtype='int32'))
                audio_path = [a for a in audio_matches if file_tag in a]
                if len(audio_path) != 1:
                    raise ValueError("More than one match for"
                                     "tag %s!" % file_tag)
                if not os.path.exists(audio_path[0][:-5] + ".wav"):
                    r = os.system("ffmpeg -i %s %s.wav" % (audio_path[0],
                                                           audio_path[0][:-5]))
                    if r:
                        raise ValueError("A problem occured converting flac to"
                                         "wav, make sure ffmpeg is installed")
                wav_path = audio_path[0][:-5] + '.wav'
                fs, d = wavfile.read(wav_path)
                # Preprocessing from A. Graves "Towards End-to-End Speech
                # Recognition"
                Pxx, _, _, _ = plt.specgram(d, NFFT=256, noverlap=128)
                data_x_shapes.append(np.array(Pxx.T.shape, dtype='int32'))
                data_x.append(Pxx.T.astype('float32').flatten())
            f.close()
        h5_file.close()

    h5_file_path = os.path.join(data_path, "saved_libri.h5")
    h5_file = tables.openFile(h5_file_path, mode='r')
    data_x = h5_file.root.data_x
    data_x_shapes = h5_file.root.data_x_shapes
    data_y = h5_file.root.data_y
    # A dirty hack to only monkeypatch data_x
    data_x.__class__ = _cVLArray

    # override getter so that it gets reshaped to 2D when fetched
    old_getter = data_x.__getitem__

    def getter(self, key):
        if isinstance(key, numbers.Integral) or isinstance(key, np.integer):
            return old_getter(key).reshape(data_x_shapes[key]).astype(
                theano.config.floatX)
        elif isinstance(key, slice):
            start, stop, step = self._processRange(key.start, key.stop,
                                                   key.step)
            return [o.reshape(s) for o, s in zip(
                self.read(start, stop, step), data_x_shapes[slice(
                    start, stop, step)])]

    # Patch __getitem__ in custom subclass, applying to all instances of it
    _cVLArray.__getitem__ = getter

    train_x = data_x[:2000]
    train_y = data_y[:2000]
    valid_x = data_x[2000:2500]
    valid_y = data_y[2000:2500]
    test_x = data_x[2500:]
    test_y = data_y[2500:]
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval
