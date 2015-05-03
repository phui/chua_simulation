import os
import sys
import pickle
import numpy as np
import pandas as pd
from math import log
from multiprocessing import Pool


class Simple3DMarkovModel(object):
    def __init__(self, df, N):
        self._data = df.copy()
        self._intervals = [np.linspace(np.min(df[name]), np.max(df[name]), N)[1:]
                          for name in self._data.columns[:-1]]

    @staticmethod
    def entropy(trans_df):
        total = trans_df['count'].sum()
        count_series = trans_df['count'] / total
        return count_series.apply(lambda p: -p * log(p, 2)).sum()

    def _get_encoder(self, dim):
        def encoder(pt):
            idx = 0
            for upper_bound in self._intervals[dim]:
                #import pdb; pdb.set_trace()
                if pt < upper_bound:
                    return idx
                idx += 1
            return idx
        return encoder

    def build(self):
        cols = self._data.columns
        for i in range(len(cols)-1):
            self._data[cols[i]] = self._data[cols[i]].apply(self._get_encoder(i))

        cord_map = dict()
        trans_matrix = dict()
        stat_prob = dict()
        loop_records = list()
        seen_set = set()

        prev_cord = tuple(self._data.iloc[0][:-1])
        prev_key = 0
        key_idx = 1
        cord_map[prev_cord] = prev_key
        stat_prob[prev_key] = 1
        for idx in range(1, len(self._data)):
            cur_cord = tuple(self._data.iloc[idx][:-1])
            if cur_cord == prev_cord:
                continue

            cur_key = -1
            if cur_cord in cord_map:
                cur_key = cord_map[cur_cord]
            else:
                cur_key = key_idx
                cord_map[cur_cord] = cur_key
                key_idx += 1

            trans_matrix[(prev_key, cur_key)] = trans_matrix.get((prev_key, cur_key), 0) + 1
            stat_prob[cur_key] = stat_prob.get(cur_key, 0) + 1
            prev_key = cur_key
            prev_cord = cur_cord
        del cord_map

        loop_start_idx = 0
        prev_cord = tuple(self._data.iloc[0][0:2])
        seen_set.add(prev_cord)
        for idx in range(1, len(self._data)):
            cord = tuple(self._data.iloc[idx][0:2])
            if cord in seen_set:
                if prev_cord == cord:
                    continue
                loop_records.append( (loop_start_idx, idx) )
                loop_start_idx = idx

                seen_set.clear()
                seen_set.add(cord)
            else:
                seen_set.add(cord)
            prev_cord = cord

        del seen_set

        stat_prob = pd.DataFrame(stat_prob.items(), columns=['node', 'prob'])
        total = stat_prob['prob'].sum()
        stat_prob['prob'] /= total
        stat_prob.sort_index()

        trans_matrix =  pd.DataFrame([(key[0], key[1], val) for key, val in trans_matrix.items()],
                            columns=['from', 'to', 'count'])
        ents = trans_matrix.groupby('from').apply(Simple3DMarkovModel.entropy).sort_index()
        ent_rate = ents * stat_prob['prob']
        ent_rate = ent_rate.sum()

        return ent_rate, ents, loop_records


def build_model(fpath):
    df = pd.read_csv(os.path.join('simulation data', fpath))
    with open('test.pkl', 'w') as f:
        mc = Simple3DMarkovModel(df, 100)
        ent_rate, ents, loop_records = mc.build()
        pickle.dump([ent_rate, ents, loop_records], f)


if __name__ == '__main__':
    build_model("output_R1740_1430123192.csv")
