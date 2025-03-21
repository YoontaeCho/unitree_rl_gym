import numpy as np

class MetricUtils:
    def __init__(self):
        self._pos_diff = []
        self._torque_diff = []
        self._pos_jitter = []
        self.prev_q, self.prev_dq, self.prev_ddq, self.prev_tau = None, None, None, None
        self.prev_prev_dq = None
        self.counter = 0
        self.loaded_action = None
        self.exp_name = None
    
    def pos_diff(self, curr_q, prev_q):
        """calculate pos diff metric"""
        self._pos_diff.append(np.average(np.abs(curr_q - prev_q)))
    
    def torque_diff(self, curr_tau, prev_tau):
        """calculate torque diff metric"""
        self._torque_diff.append(np.average(np.abs(curr_tau - prev_tau)))
    
    def pos_jitter(self, curr_dq, prev_dq, prev_prev_dq):
        """calculate pos jitter metric"""
        self._pos_jitter.append(np.average(np.abs((curr_dq - prev_dq) - (prev_dq - prev_prev_dq))))


    def calculate_metrics(self, curr_q, curr_dq, curr_ddq, curr_tau, logpath):
        """calculate metrics"""
        if self.prev_prev_dq is not None:
            self.pos_diff(curr_q, self.prev_q)
            # print(curr_dq)
            # print(curr_q)
            self.pos_jitter(curr_dq, self.prev_dq, self.prev_prev_dq)
            self.torque_diff(curr_tau, self.prev_tau)
            # Save merics.
            np.save(F'{logpath}/metric_{self.exp_name}.npy',
                    np.array(
                        [np.average(self._pos_diff), 
                         np.average(self._pos_jitter), 
                         np.average(self._torque_diff)]
                         ))
        # if self.counter == len(self.loaded_action) - 1:
        print("\n--------------------------METRICS--------------------------")
        print("pos_diff", np.average(self._pos_diff))
        print("pos_jitter", np.average(self._pos_jitter))
        print("torque_diff", np.average(self._torque_diff))
    
        self.prev_q, self.prev_dq, self.prev_ddq, self.prev_tau = curr_q, curr_dq, curr_ddq, curr_tau
        self.prev_prev_dq = self.prev_dq