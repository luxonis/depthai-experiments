import json

DIST_THRESHOLD = 700 # 70%
ids = []

def check_similarity(reid, cfg):
    # For first id
    if len(ids) == 0:
        ids.append({
            'reid': reid,
            'cnt': 0
        })
        return

    for i, obj in enumerate(ids):
        node.io['a'].send(obj['reid'])
        node.io['b'].send(reid)
        cos = node.io['cos'].get().getFirstLayerFp16()
        if DIST_THRESHOLD < cos[0]:
            # Person re-identified
            obj['reid'] = reid
            obj['cnt'] += 1
            # dict that gets sent to host/SPI
            out = {
                'cnt': obj['cnt'],
                'id': i,
                'bb': [cfg.getCropXMin(), cfg.getCropYMin(), cfg.getCropXMax(), cfg.getCropYMax()],
            }
            b = Buffer(150)
            b.setData(json.dumps(out).encode('ascii'))
            node.io['out'].send(b)
            return
    # Person wans't reidentified. Add new person
    node.warn('New person added')
    ids.append({
        'reid': reid,
        'cnt': 0
    })

while True:
    cfg = node.io['cfg'].get()
    reid = node.io['reid'].get()
    check_similarity(reid, cfg)

