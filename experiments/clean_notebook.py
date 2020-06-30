import sys
import io
from IPython.nbformat import current


def remove_outputs(nb):
    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type == 'code':
                cell.outputs = []


if __name__ == '__main__':
    fname = sys.argv[1]
    with io.open(fname, 'r') as f:
        nb = current.read(f, 'json')
        remove_outputs(nb)
        print(current.writes(nb, 'json'))
