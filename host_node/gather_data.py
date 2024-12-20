from dataclasses import dataclass
from queue import PriorityQueue
import time
import depthai as dai
from datetime import timedelta


@dataclass
class SyncData:
    seq_number: int
    timestamp: timedelta
    timestamp_device: timedelta


    @staticmethod
    def get_from(source: dai.Buffer) -> "SyncData":
        return SyncData(
            source.getSequenceNum(), 
            source.getTimestamp(), 
            source.getTimestampDevice()
        )


    def copy_to(self, target: dai.Buffer) -> None:
        target.setSequenceNum(self.seq_number)
        target.setTimestamp(self.timestamp)
        target.setTimestampDevice(self.timestamp_device)


class ExpectedDataLen(dai.node.Buffer):
    def __init__(self, expected_len: int, sync_data: SyncData) -> None:
        super().__init__(0)
        self.expected_len = expected_len
        self.sync_data = sync_data
        sync_data.copy_to(self)


class GatheredData(dai.Buffer):
    def __init__(self, data: list[dai.Buffer], sync_data: SyncData | None = None) -> None:
        super().__init__(0)
        self.set_data(data, sync_data)


    def set_data(self, data: list[dai.Buffer], sync_data: SyncData | None = None) -> None:
        self._data = data
        if sync_data:
            self._sync_data = sync_data
        else:
            self._sync_data = SyncData.get_from(data[0])
        self._sync_data.copy_to(self)


    def get_data(self) -> list[dai.Buffer]:
        return self._data
    

    def get_sync_data(self) -> SyncData:
        return self._sync_data
    

class GatherData(dai.node.ThreadedHostNode):
    FPS_TOLERANCE_DIVISOR = 2
    INPUT_CHECKS_PER_FPS = 2


    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._camera_fps = 30
        self._unmatched_data: list[dai.Buffer] = []
        self._data_by_ts: dict[float, list[dai.Buffer]] = {}
        self._expected_len_by_ts: dict[float, ExpectedDataLen] = {}
        self._ready_timestamps = PriorityQueue()

        self.input_data = dai.Node.Input(self)
        self.input_expected_len = dai.Node.Input(self)
        self.output = dai.Node.Output(self)
        
    
    def build(self) -> "GatherData":
        return self
    

    def set_camera_fps(self, fps: int) -> None:
        self._camera_fps = fps


    def run(self) -> None:
        while self.isRunning():
            all_data = self.input_data.tryGetAll()
            if all_data:
                for data in all_data:
                    self._add_data(data)

            all_expected_len = self.input_expected_len.tryGetAll()
            if all_expected_len:
                for expected_len in all_expected_len:
                    self._add_expected_len(expected_len)

            if all_data or all_expected_len:
                ready_data = self._pop_ready_data()
                if ready_data:
                    self._clear_old_data(ready_data)
                    self.output.send(ready_data)
            time.sleep(1 / self.INPUT_CHECKS_PER_FPS / self._camera_fps)


    def _add_data(self, data: dai.Buffer) -> None:
        data_ts = self._get_total_seconds_ts(data)
        matching_ts = self._get_matching_ts(data_ts)
        
        if matching_ts is not None:
            self._add_data_by_ts(data, matching_ts)
            self._update_ready_timestamps(matching_ts)
        else:
            self._unmatched_data.append(data)


    def _get_matching_ts(self, data_ts: float) -> float | None:
        for timestamp in self._expected_len_by_ts.keys():
            if self._timestamps_in_tolerance(timestamp, data_ts):
                return timestamp
        return None
    

    def _add_expected_len(self, expected_len: ExpectedDataLen) -> None:
        expected_len_ts = self._get_total_seconds_ts(expected_len)
        self._expected_len_by_ts[expected_len_ts] = expected_len
        self._try_gather_data(expected_len_ts)
        self._update_ready_timestamps(expected_len_ts)


    def _try_gather_data(self, expected_len_ts: float) -> None:
        matched_data: list[dai.Buffer] = []
        for data in self._unmatched_data:
            data_ts = self._get_total_seconds_ts(data)
            if self._timestamps_in_tolerance(expected_len_ts, data_ts):
                self._add_data_by_ts(data, expected_len_ts)
                matched_data.append(data)
        
        for matched_recognition in matched_data:
            self._unmatched_data.remove(matched_recognition)

    
    def _timestamps_in_tolerance(self, timestamp1: float, timestamp2: float) -> bool:
        difference = abs(timestamp1 - timestamp2)
        return difference < (1 / self._camera_fps / self.FPS_TOLERANCE_DIVISOR)


    def _add_data_by_ts(self, data: dai.Buffer, timestamp: float) -> None:
        if timestamp in self._data_by_ts:
            self._data_by_ts[timestamp].append(data)
        else:
            self._data_by_ts[timestamp] = [data]


    def _update_ready_timestamps(self, timestamp: float) -> None:
        if not self._timestamp_ready(timestamp):
            return
        
        self._ready_timestamps.put(timestamp)
        

    def _timestamp_ready(self, timestamp: float) -> None:
        expected_data_len = self._expected_len_by_ts.get(timestamp)
        if not expected_data_len:
            return False
        elif len(expected_data_len.expected_len) == 0:
            return True
        
        data = self._data_by_ts.get(timestamp)
        if not data:
            return False
        
        return len(expected_data_len.expected_len) == len(data)


    def _pop_ready_data(self) -> GatheredData | None:
        if self._ready_timestamps.empty():
            return None
        
        timestamp = self._ready_timestamps.get()
        expected_len = self._expected_len_by_ts.pop(timestamp)
        data = self._data_by_ts.pop(timestamp, None)
        if data is None:
            print("Error data is none for timestamp: ", timestamp)

        return GatheredData(data, expected_len.sync_data)

    
    def _clear_old_data(self, ready_data: GatheredData) -> None:
        current_timestamp = self._get_total_seconds_ts(ready_data)
        self._clear_unmatched_data(current_timestamp)
        self._clear_matched_data(current_timestamp)


    def _clear_unmatched_data(self, current_timestamp: float) -> None:
        unmatched_data_to_remove = []
        for data in self._unmatched_data:
            if self._get_total_seconds_ts(data) < current_timestamp:
                unmatched_data_to_remove.append(data)
        
        for data in unmatched_data_to_remove:
            self._unmatched_data.remove(data)


    def _get_total_seconds_ts(self, buffer_like: dai.Buffer) -> float:
        return buffer_like.getTimestamp().total_seconds()

        
    def _clear_matched_data(self, current_timestamp) -> None:
        keys_to_pop = []
        for expected_len_ts in self._expected_len_by_ts.keys():
            if expected_len_ts < current_timestamp:
                keys_to_pop.append(expected_len_ts)
        
        for expected_len_ts in keys_to_pop:
            self._expected_len_by_ts.pop(expected_len_ts)
            self._data_by_ts.pop(expected_len_ts, None)
