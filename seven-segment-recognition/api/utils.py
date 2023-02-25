import collections
import math

import numpy as np


class BeamSearchDecoder:

    def __init__(self, vocab, beam_len=30, blank=0):

        self.vocab_dict = {0: ""}

        for i, char in enumerate(vocab):
            self.vocab_dict[i + 1] = char

        self.beam_len = beam_len
        self.blank = blank

    def _create_new_beam(self):
        fn = lambda: (-float("inf"), -float("inf"))
        return collections.defaultdict(fn)

    def _log_sum_exp(self, *args):
        if all(a == -float("inf") for a in args):
            return -float("inf")

        a_max = max(args)

        lsp = math.log(
            sum(
                math.exp(a - a_max) for a in args
            )
        )
        return a_max + lsp

    def decode(self, probs):
        T, S = probs.shape
        probs = np.log(probs)

        beam = [(tuple(), (0.0, -float("inf")))]

        for t in range(T):

            next_beam = self._create_new_beam()

            for s in range(S):
                p = probs[t, s]

                for prefix, (p_b, p_nb) in beam:

                    if s == self.blank:
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_b = self._log_sum_exp(n_p_b, p_b + p, p_nb + p)
                        next_beam[prefix] = (n_p_b, n_p_nb)
                        continue

                    end_t = prefix[-1] if prefix else None
                    n_prefix = prefix + (s,)
                    n_p_b, n_p_nb = next_beam[n_prefix]
                    if s != end_t:
                        n_p_nb = self._log_sum_exp(n_p_nb, p_b + p, p_nb + p)
                    else:
                        n_p_nb = self._log_sum_exp(n_p_nb, p_b + p)

                    next_beam[n_prefix] = (n_p_b, n_p_nb)

                    if s == end_t:
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_nb = self._log_sum_exp(n_p_nb, p_nb + p)
                        next_beam[prefix] = (n_p_b, n_p_nb)

            beam = sorted(next_beam.items(),
                          key=lambda x: self._log_sum_exp(*x[1]),
                          reverse=True)
            beam = beam[:self.beam_len]

        best = beam[0]
        return "".join([self.vocab_dict[idx] for idx in best[0]])
