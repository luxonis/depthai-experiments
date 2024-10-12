import depthai as dai
from host_node.draw_text import TextMessage
from host_node.object_counter import ObjectCount


class CounterTextSerializer(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.output = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.Buffer, True)
            ]
        )

    def build(
        self, counter: dai.Node.Output, label_map: list[str]
    ) -> "CounterTextSerializer":
        self.link_args(counter)
        self._label_map = label_map
        return self

    def process(self, object_count: dai.Buffer) -> None:
        assert isinstance(object_count, ObjectCount)

        text = ""
        for i, label in enumerate(self._label_map):
            count = 0
            if i in object_count.label_counts:
                count = object_count.label_counts[i]
            text += f"{label}: {count}\n"

        self.output.send(TextMessage(text))
