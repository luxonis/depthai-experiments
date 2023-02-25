import numpy as np


class CTCCodec:
    """ Convert between text-label and text-index """
    CHARACTERS = dict(zip(range(37), '0123456789abcdefghijklmnopqrstuvwxyz#'))

    @staticmethod
    def decode(preds):
        """Convert text-index into text-label."""
        texts = []
        index = 0
        # Select max probabilty (greedy decoding) then decode index to character
        preds = preds.astype(np.float16)
        preds_index = np.argmax(preds, 2)
        preds_index = preds_index.transpose(1, 0)
        preds_index_reshape = preds_index.reshape(-1)
        preds_sizes = np.array([preds_index.shape[1]] * preds_index.shape[0])

        for l in preds_sizes:
            t = preds_index_reshape[index:index + l]

            # NOTE: t might be zero size
            if t.shape[0] == 0:
                continue

            char_list = []
            for i in range(l):
                # removing repeated characters and blank.
                if not (i > 0 and t[i - 1] == t[i]):
                    if CTCCodec.CHARACTERS[t[i]] != '#':
                        char_list.append(CTCCodec.CHARACTERS[t[i]])
            text = ''.join(char_list)
            texts.append(text)

            index += l

        return texts


codec = CTCCodec()
