import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from images2gif import writeGif
from PIL import Image


def plot_2D_phase_space(filename):
    df = pd.read_csv(os.path.join('outputs', filename))

    plt.figure(figsize=(10, 10))
    plt.plot(df['v_C1'][60000:120001], df['v_C2'][60000:120001], 'b')
    plt.title('v_C1-v_C2 phase space with R=%s Ohm' % filename[8:12])

    plt.xlim(-8, 8)
    plt.xlabel('v_C1 (V)')

    plt.ylim(-2, 2)
    plt.ylabel('v_C2 (V)')

    plt.grid()
    plt.savefig(
        os.path.join('outputs2','plot_%s.png' % filename.split('.')[0]),
        format='png')
    plt.close()


if __name__ == '__main__':
    filenames = sorted((fn for fn in os.listdir('outputs') if fn.endswith('.csv')))[34:]
    
    for filename in filenames:
        print 'found %s, plotting...' % filename
        plot_2D_phase_space(filename)
        print 'plotting done!\n'

    print 'making animation'
    filenames = sorted((fn for fn in os.listdir('outputs2') if fn.endswith('.png')))
    frames = [Image.open(os.path.join('outputs2', fn)) for fn in filenames]
    writeGif('anime_%d.gif' % int(time.time()), frames, duration=0.3)
    print 'DONE!'