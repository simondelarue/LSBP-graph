import LSBP
import os
import inspect
import numpy
import pathlib

from LSBP.ranking import PageRank

if __name__=='__main__':
    
    DATA_DIR = 'data'

    print(inspect.getfile(numpy))

    sid = pathlib.Path(__file__).parent
    print(os.path.expanduser(f'~/{sid}'))
    print(sid)

    pr = PageRank(f'{DATA_DIR}/soc-LiveJournal1_toy_bis.txt')
    print(pr)

    """class FileDataset(Dataset):
        '''Base class for datasets that are stored in a local file.
        Small datasets that are part of the river package inherit from this class.
        '''

        def __init__(self, filename, **desc):
            super().__init__(**desc)
            self.filename = filename

        @property
        def path(self):
            return pathlib.Path(__file__).parent.joinpath(self.filename)

        @property
        def _repr_content(self):
            content = super()._repr_content
            content["Path"] = str(self.path)
            return content"""