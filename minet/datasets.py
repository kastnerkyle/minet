# -*- coding: utf 8 -*-
from __future__ import division
import tarfile
import os
from scipy.io import wavfile
import numpy as np
import tables
import numbers
import random
import string
import fnmatch
import theano
from lxml import etree
try:
    import urllib.request as urllib  # for backwards compatibility
except ImportError:
    import urllib2 as urllib


def get_dataset_dir(dataset_name, data_dir=None, folder=None, create_dir=True):
    if not data_dir:
        data_dir = os.getenv("MINET_DATA", os.path.join(
            os.path.expanduser("~"), "minet_data"))
    if folder is None:
        data_dir = os.path.join(data_dir, dataset_name)
    else:
        data_dir = os.path.join(data_dir, folder)
    if not os.path.exists(data_dir) and create_dir:
        os.makedirs(data_dir)
    return data_dir


def download(url, server_fname, local_fname=None, progress_update_percentage=5):
    """
    An internet download utility modified from
    http://stackoverflow.com/questions/22676/
    how-do-i-download-a-file-over-http-using-python/22776#22776
    """
    u = urllib.urlopen(url)
    if local_fname is None:
        local_fname = server_fname
    full_path = local_fname
    meta = u.info()
    with open(full_path, 'wb') as f:
        try:
            file_size = int(meta.get("Content-Length"))
        except TypeError:
            print("WARNING: Cannot get file size, displaying bytes instead!")
            file_size = 100
        print("Downloading: %s Bytes: %s" % (server_fname, file_size))
        file_size_dl = 0
        block_sz = int(1E7)
        p = 0
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            if (file_size_dl * 100. / file_size) > p:
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl *
                                               100. / file_size)
                print(status)
                p += progress_update_percentage


def check_fetch_iamondb():
    partial_path = get_dataset_dir("iamondb")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    strokes_path = os.path.join(partial_path, "lineStrokes")
    ascii_path = os.path.join(partial_path, "ascii")
    if not os.path.exists(strokes_path) or not os.path.exists(ascii_path):
        raise ValueError("You must download the data from IAMOnDB, and"
                         "unpack in %s" % partial_path)
    return strokes_path, ascii_path


def plot_scatter_iamondb_example(X, y=None):
    import matplotlib.pyplot as plt
    rgba_colors = np.zeros((len(X), 4))
    normed = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # for red the first column needs to be one
    rgba_colors[:, 0] = normed[:, 0]
    # for blue last color column needs to be one
    rgba_colors[:, 2] = np.abs(1 - normed[:, 0])
    # the fourth column needs to be alphas
    rgba_colors[:, 3] = np.ones((len(X),)) * .4 + .4 * normed[:, 0]
    if len(X[0]) == 3:
        plt.scatter(X[:, 1], X[:, 2], color=rgba_colors)
    elif len(X[0]) == 2:
        plt.scatter(X[:, 0], X[:, 1], color=rgba_colors)
    if y is not None:
        plt.title(y)
    plt.axis('equal')


def plot_lines_iamondb_example(X, y=None):
    import matplotlib.pyplot as plt
    val_index = np.where(X[:, 0] != 1)[0]
    contiguous = np.where((val_index[1:] - val_index[:-1]) == 1)[0] + 1
    non_contiguous = np.where((val_index[1:] - val_index[:-1]) != 1)[0] + 1
    prev_nc = 0
    for nc in val_index[non_contiguous]:
        ind = ((prev_nc <= contiguous) & (contiguous < nc))[:-1]
        prev_nc = nc
        plt.plot(X[val_index[ind], 1], X[val_index[ind], 2])
    plt.plot(X[prev_nc:, 1], X[prev_nc:, 2])
    if y is not None:
        plt.title(y)
    plt.axis('equal')

# A trick for monkeypatching an instancemethod that when method is a
# c-extension? there must be a better way
class _textEArray(tables.EArray):
    pass


class _handwritingEArray(tables.EArray):
    pass


def fetch_iamondb():
    strokes_path, ascii_path = check_fetch_iamondb()

    stroke_matches = []
    for root, dirnames, filenames in os.walk(strokes_path):
        for filename in fnmatch.filter(filenames, '*.xml'):
            stroke_matches.append(os.path.join(root, filename))

    ascii_matches = []
    for root, dirnames, filenames in os.walk(ascii_path):
        for filename in fnmatch.filter(filenames, '*.txt'):
            ascii_matches.append(os.path.join(root, filename))

    partial_path = get_dataset_dir("iamondb")
    hdf5_path = os.path.join(partial_path, "iamondb.h5")
    if not os.path.exists(hdf5_path):
        # setup tables
        compression_filter = tables.Filters(complevel=5, complib='blosc')
        hdf5_file = tables.openFile(hdf5_path, mode='w')
        handwriting = hdf5_file.createEArray(hdf5_file.root, 'handwriting',
                                             tables.Int32Atom(),
                                             shape=(0, 3),
                                             filters=compression_filter,
                                             expectedrows=len(ascii_matches))
        handwriting_poslen = hdf5_file.createEArray(hdf5_file.root,
                                                    'handwriting_poslen',
                                                    tables.Int32Atom(),
                                                    shape=(0, 2),
                                                    filters=compression_filter,
                                                    expectedrows=len(
                                                        ascii_matches))
        text = hdf5_file.createEArray(hdf5_file.root, 'text',
                                      tables.Int32Atom(),
                                      shape=(0, 1),
                                      filters=compression_filter,
                                      expectedrows=len(ascii_matches))
        text_poslen = hdf5_file.createEArray(hdf5_file.root, 'text_poslen',
                                             tables.Int32Atom(),
                                             shape=(0, 2),
                                             filters=compression_filter,
                                             expectedrows=len(ascii_matches))

        current_text_pos = 0
        current_handwriting_pos = 0
        for na, ascii_file in enumerate(ascii_matches):
            if na % 100 == 0:
                print("Reading ascii file %i of %i" % (na, len(ascii_matches)))
            with open(ascii_file) as fp:
                cleaned = [t.strip() for t in fp.readlines()
                           if 'OCR' not in t
                           and 'CSR' not in t
                           and t != '\r\n'
                           and t != '\n']

                # Find correspnding XML file for ascii file
                file_id = ascii_file.split(os.sep)[-2]
                submatches = [sf for sf in stroke_matches if file_id in sf]
                # Sort by file number
                submatches = sorted(submatches,
                                    key=lambda x: int(
                                        x.split(os.sep)[-1].split(
                                            "-")[-1][:-4]))
                # Skip files where ascii length and number of XML don't match
                # TODO: Figure out why this is happening
                if len(cleaned) != len(submatches):
                    continue

                for n, stroke_file in enumerate(submatches):
                    with open(stroke_file) as fp:
                        tree = etree.parse(fp)
                        root = tree.getroot()
                        # Get all the values from the XML
                        # 0th index is stroke ID, will become up/down
                        s = np.array([[i, int(Point.attrib['x']),
                                       int(Point.attrib['y'])]
                                       for StrokeSet in root
                                       for i, Stroke in enumerate(StrokeSet)
                                       for Point in Stroke])
                        # flip y axis
                        s[:, 2] = -s[:, 2]
                        # Get end of stroke points
                        c = s[1:, 0] != s[:-1, 0]
                        ci = np.where(c == True)[0]
                        nci = np.where(c == False)[0]
                        # set pen down
                        s[0, 0] = 0
                        s[nci, 0] = 0
                        # set pen up
                        s[ci, 0] = 1
                        s[-1, 0] = 1

                        lh = len(s)
                        for i in range(lh):
                            handwriting.append(s[i][None])
                        handwriting_poslen.append(
                            np.array([current_handwriting_pos, lh])[None])
                        current_handwriting_pos += lh

                        lt = len(cleaned[n])
                        for i in range(lt):
                            text.append(
                                np.array(ord(cleaned[n][i]))[None, None])
                        text_poslen.append(
                            np.array([current_text_pos, lt])[None])
                        current_text_pos += lt
        hdf5_file.close()
    hdf5_file = tables.openFile(hdf5_path, mode='r')
    handwriting = hdf5_file.root.handwriting
    handwriting_poslen = hdf5_file.root.handwriting_poslen
    text = hdf5_file.root.text
    text_poslen = hdf5_file.root.text_poslen

    # Monkeypatch text
    # A dirty hack to only monkeypatch text
    text.__class__ = _textEArray

    def text_getter(self, key):
        if isinstance(key, numbers.Integral) or isinstance(key, np.integer):
            p, l = text_poslen[key]
            return "".join(map(chr, self.read(p, p+l, 1)))
        elif isinstance(key, slice):
            start, stop, step = self._processRange(key.start, key.stop,
                                                   key.step)
            if key.stop is None:
                stop = len(text_poslen)
            if key.start is None:
                start = 0

            if stop <= start:
                # replicate slice where stop <= start
                return []

            if stop >= len(text_poslen):
                stop = len(text_poslen)
            elif key.stop < 0 and key.stop is not None:
                stop = len(text_poslen) + key.stop
            if key.start < 0 and key.start is not None:
                start = len(text_poslen) + key.start

            return ["".join(map(chr, self.read(text_poslen[k][0],
                                               sum(text_poslen[k]), 1)))
                    for k in range(start, stop, step)]

    # Patch __getitem__ in custom subclass, applying to all instances of it
    _textEArray.__getitem__ = text_getter

    # Monkeypatch handwriting
    # A dirty hack to only monkeypatch handwriting
    handwriting.__class__ = _handwritingEArray

    def handwriting_getter(self, key):
        if isinstance(key, numbers.Integral) or isinstance(key, np.integer):
            p, l = handwriting_poslen[key]
            return self.read(p, p+l, 1)
        elif isinstance(key, slice):
            start, stop, step = self._processRange(key.start, key.stop,
                                                   key.step)
            if key.stop is None:
                stop = len(text_poslen)
            if key.start is None:
                start = 0

            if stop <= start:
                # replicate slice where stop <= start
                return []

            if stop >= len(text_poslen):
                stop = len(text_poslen)
            elif key.stop < 0 and key.stop is not None:
                stop = len(text_poslen) + key.stop
            if key.start < 0 and key.start is not None:
                start = len(text_poslen) + key.start

            return [self.read(handwriting_poslen[k][0],
                              sum(handwriting_poslen[k]), 1)
                    for k in range(start, stop, step)]

    # Patch __getitem__ in custom subclass, applying to all instances of it
    _handwritingEArray.__getitem__ = handwriting_getter
    X = handwriting
    y = text
    return (X, y)


"""
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
"""
