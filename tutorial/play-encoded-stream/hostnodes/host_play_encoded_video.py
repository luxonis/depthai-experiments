import depthai as dai
import sys

WIDTH, HEIGHT = 800, 600

class PlayEncodedVideo(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

        self.verbose = True
        self.output = self.createOutput(possibleDatatypes=[dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)])
        
    
    def build(self, enc_out : dai.Node.Output, proc) -> "PlayEncodedVideo":
        self.proc = proc
        self.link_args(enc_out)
        self.sendProcessingToPipeline(True)
        self.inputs['encoded_vid'].setBlocking(True)
        self.inputs['encoded_vid'].setMaxSize(30)
        return self
    

    def process(self, encoded_vid) -> None:
        try:
            data = encoded_vid.getData()
            if self.verbose:
                latms = (dai.Clock.now() - encoded_vid.getTimestamp()).total_seconds() * 1000
                # Writing to a different channel (stderr)
                # Also `ffplay` is printing things, adding a separator
                print(f'Latency: {latms:.3f} ms === ', file=sys.stderr)
            self.proc.stdin.write(data)
        except:
            self.stopPipeline()
            
            
    def set_verbose(self, verbose: bool) -> None:
        self.verbose = verbose