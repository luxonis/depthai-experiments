import time
ids = []
while True:
    cfg = node.io['manip_cfg'].tryGet()
    if cfg is not None:
        reid = node.io['reid'].get()

        if len(ids) == 0:
            ids.append({
                'reid': reid,
                'cnt': 0
            })
            continue

        for i, obj in enumerate(ids):
            # Crop frame and run person_reidentification model on it
            node.io['a'].send(reid)
            node.io['b'].send(obj['reid'])
            start = time.time()
            cos = node.io['cos'].get().getFirstLayerFp16()
            node.warn(f"Compare time: {time.time()-start}")
            node.warn(f"Cos dist {cos}")