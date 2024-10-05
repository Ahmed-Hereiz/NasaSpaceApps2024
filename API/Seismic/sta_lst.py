import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import highpass
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset

class STALTA:
    def __init__(self,
            sta_len: int=120,
            lta_len: int=600,
            thr_on: int=4, 
            thr_off: int=1.5,
            minfreq: int=0.5,
            maxfreq: int=1
        ):

        self.sta_len = sta_len
        self.lta_len = lta_len
        self.thr_on = thr_on
        self.thr_off = thr_off
        self.minfreq = minfreq
        self.maxfreq = maxfreq

    def filter_data(self, st, minfreq, maxfreq):
        st_filt = st.copy()
        st_filt.filter('bandpass',freqmin=minfreq,freqmax=maxfreq)
        tr_filt = st_filt.traces[0].copy()
        tr_times_filt = tr_filt.times()
        tr_data_filt = tr_filt.data
        return tr_times_filt, tr_data_filt

    def get_start(self, st, idx = 0, filter = True):
        tr = st[idx]
        tr_data = tr.data
        tr_times = tr.times()
        sampling_rate = tr.stats.sampling_rate

        if filter:
            tr_times, tr_data = self.filter_data(st, self.minfreq, self.maxfreq)

        cft = classic_sta_lta(tr_data, int(self.sta_len * sampling_rate), int(self.lta_len * sampling_rate))
        on_off = np.array(trigger_onset(cft, self.thr_on, self.thr_off))
        
        try:
            return tr_times[on_off[0][0]]
        except Exception as e:
            return -1

    def get_vis(self, st, idx = 0, filter = True, show = False):
        tr = st[idx]
        tr_times = tr.times()
        tr_data = tr.data
        arrival = self.get_start(st, idx, filter)
        
        fig, ax = plt.subplots(1,1,figsize=(12,3))
        ax.axvline(x = arrival, color='red', label='Trig. On')
            

        # Plot seismogram
        ax.plot(tr_times,tr_data)
        ax.set_xlim([min(tr_times),max(tr_times)])
        ax.legend()
        plt.show()

