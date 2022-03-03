import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
import os


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    # im = ax.imshow(spectrogram**0.125, origin='lower', aspect='auto', 
                #    interpolation='nearest')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_gate_outputs_to_numpy(gate_targets, gate_outputs):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(range(len(gate_targets)), gate_targets, alpha=0.5,
               color='green', marker='+', s=1, label='target')
    ax.scatter(range(len(gate_outputs)), gate_outputs, alpha=0.5,
               color='red', marker='.', s=1, label='predicted')

    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("Gate State")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_spectrogram_to_numpy_dur(spectrogram, phone_seq, note_len, ph_du, sampling_rate, hop_length):
    frame_len = hop_length / sampling_rate
    list_cum_num_pho = []
    cum_num_pho = 0
    # for du in ph_du:
        # cum_num_pho += int(np.round(du / frame_len))  # sec
        # cum_num_pho += int(du)  # num of frame
        # list_cum_num_pho.append(cum_num_pho)
    # ratio-sec-nFrame
    assert ph_du.shape == note_len.shape
    for du_ratio, this_p_notelen in zip(ph_du, note_len):
        cum_num_pho += int(np.round(du_ratio * this_p_notelen / frame_len))  # sec
        list_cum_num_pho.append(cum_num_pho)        
    
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')

    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    for xc, label in zip(list_cum_num_pho, phone_seq):
        plt.axvline(x=xc, c='r')
    
    xticks_label = []
    for i in range(len(list_cum_num_pho)):
        if i == 0:
            xticks_label.append(list_cum_num_pho[i]/2)
        else:
            xticks_label.append((list_cum_num_pho[i]+list_cum_num_pho[i-1])/2)
    plt.xticks(xticks_label, phone_seq)
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_spectrogram_to_numpy_dur_v2(spectrogram, phone_seq, note_len, ph_du, sampling_rate, hop_length):
    frame_len = hop_length / sampling_rate
    list_cum_num_pho = []
    cum_num_pho = 0
    # for du in ph_du:
        # cum_num_pho += int(np.round(du / frame_len))  # sec
        # cum_num_pho += int(du)  # num of frame
        # list_cum_num_pho.append(cum_num_pho)
    # ratio-sec-nFrame
    assert ph_du.shape == note_len.shape
    for du_ratio, this_p_notelen in zip(ph_du, note_len):
        cum_num_pho += int(np.round(du_ratio * this_p_notelen/ frame_len))  # sec
        list_cum_num_pho.append(cum_num_pho)        
    
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')

    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    for xc, label in zip(list_cum_num_pho, phone_seq):
        plt.axvline(x=xc, c='r')
    
    xticks_label = []
    for i in range(len(list_cum_num_pho)):
        if i == 0:
            xticks_label.append(list_cum_num_pho[i]/2)
        else:
            xticks_label.append((list_cum_num_pho[i]+list_cum_num_pho[i-1])/2)
    plt.xticks(xticks_label, phone_seq)
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def plot_spectrogram(idx, name, spectrogram, phone_seq, ph_du, sampling_rate, hop_length):
    frame_len = hop_length / sampling_rate
    list_cum_num_pho = []
    cum_num_pho = 0
    # for du_ratio, this_p_notelen in zip(ph_du, note_len):
    #     cum_num_pho += int(np.round(du_ratio * this_p_notelen/ frame_len))  # sec
    #     list_cum_num_pho.append(cum_num_pho)        
    for du in ph_du:
        cum_num_pho += int(np.round(du / frame_len))  # sec
        list_cum_num_pho.append(cum_num_pho)
    
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')

    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    for xc, label in zip(list_cum_num_pho, phone_seq):
        plt.axvline(x=xc, c='r')
    
    xticks_label = []
    for i in range(len(list_cum_num_pho)):
        if i == 0:
            xticks_label.append(list_cum_num_pho[i]/2)
        else:
            xticks_label.append((list_cum_num_pho[i]+list_cum_num_pho[i-1])/2)
    plt.xticks(xticks_label, phone_seq)
    plt.tight_layout()

    fig.canvas.draw()
    # data = save_figure_to_numpy(fig)
    # plt.plot(data)

    path = os.path.join('plot', name)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, '{}.png'.format(idx+1)))

    plt.close()
    return 0
