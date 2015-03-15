__version_info__ = ('0', '0', '1')
__version__ = '.'.join(__version_info__)

from .recurrent import RNN
from .recurrent import EncDecRNN
from .recurrent import GMMRNN
__all__ = ['RNN',
           'EncDecRNN',
           'GMMRNN']
