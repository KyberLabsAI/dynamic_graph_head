"""__init__
License: BSD 3-Clause License
Copyright (C) 2021, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

import os, os.path
import time
import numpy as np
import traceback

import array
import datetime
import json
from websocket_server import WebsocketServer
import threading
import signal
import sys

from mim_data_utils import DataLogger, DataReader
import matplotlib.pylab as plt

class ThreadHead(threading.Thread):
    def __init__(self, dt, safety_controllers, heads, utils, env=None):
        threading.Thread.__init__(self)

        self.dt = dt
        self.env = env

        if type(heads) != dict:
            self.head = heads # Simple-edge-case for single head setup.
            self.heads = {
                'default': heads
            }
        else:
            self.heads = heads

        self.utils = utils
        for (name, util) in utils:
            self.__dict__[name] = util

        self.ti = 0
        self.streaming = False
        self.streaming_event_loop = None
        self.logging = False
        self.log_writing = False

        self.timing_control = 0.
        self.timing_utils   = 0.
        self.timing_logging = 0.
        self.absolute_time  = 0.
        self.time_start_recording = 0.

        self.running_controller = False
        self.last_exception = None

        # Start the websocket thread/server and publish data if requested.
        self.ws_thread = None
        self.ws_is_running = False

        self.active_controllers = None
        if type(safety_controllers) != list and type(safety_controllers) != tuple:
            safety_controllers = [safety_controllers]
        self.safety_controllers = safety_controllers

    def switch_controllers(self, controllers):
        # Switching the controller changes the fields.
        # Therefore, stopping streaming and logging.
        was_streaming = self.streaming
        self.stop_logging()

        if type(controllers) != list and type(controllers) != tuple:
            controllers = [controllers]

        # Warmup the controllers and run them once to
        # get all fields propaged and have the data ready
        # for logging / streaming.
        try:
            for ctrl in controllers:
                ctrl.warmup(self)
                ctrl.run(self)

            self.active_controllers = controllers
        except KeyboardInterrupt as exp:
            raise exp
        except:
            traceback.print_exc()
            print('!!! ThreadHead: Error during controller warmup & run -> Switching to safety controller.')
            self.active_controllers = self.safety_controllers

            for ctrl in self.active_controllers:
                ctrl.warmup(self)
                ctrl.run(self)

        # If we were streaming before, calling `start_streaming()`
        # again to stream the new fields of the new controller.
        if was_streaming:
            self.start_streaming()

    def ws_thread_fn(self):
        print("Starting websocket thread for streaming.", self)

        server = WebsocketServer(host='127.0.0.1', port=5678)
        self.ws_is_running = True
        server.run_forever(threaded=True)
        last_ti = -1

        while self.streaming:
            # Do not access the controller data until the current control cycle
            # has finished. Doing wait over signal from main controller thread to
            # keep things seperated.
            if self.ti == last_ti:
                time.sleep(0.0001)
                continue

            last_ti = self.ti

            data = {}
            data['time'] = self.ti * self.dt

            for name, value in self.fields_access.items():
                val = value['ctrl'].__dict__[value['key']]
                if type(val) == np.ndarray and val.ndim == 1:
                    type_str = 'd' if val.dtype == np.float64 else 'f'
                    data[name] = str(array.array(type_str, val.data))
                else:
                    # Fake sending data as an array to the client.
                    data[name] = "array('d', [" + str(val) + "])"

            server.send_message_to_all(json.dumps(data))

        server.shutdown_gracefully()
        self.ws_is_running = False
        print("Shuted down websocket thread for streaming.", self)


    def init_log_stream_fields(self, LOG_FIELDS=['all']):
        fields = []
        fields_access = {}
        for i, ctrl in enumerate(self.active_controllers):
            ctrl_dict = ctrl.__dict__
            for key, value in ctrl_dict.items():
                if key.endswith('_'):
                    print(f"  Not logging variable '{key}' as names ending in '_' indicates it should not be logged.")
                    continue

                if(key in LOG_FIELDS or LOG_FIELDS==['all']):
                    # Support only single-dim numpy arrays and scalar only.
                    if type(value) == float or type(value) == int or issubclass(type(value), np.generic):
                        field_size = 1
                    elif type(value) == np.ndarray and value.ndim == 1:
                        field_size = value.shape[0]
                    else:
                        print(f"  Not logging '{key}' as field type '{str(type(value))}' is unsupported.")
                        continue

                    if len(self.active_controllers) == 1:
                        name = key
                    else:
                        name = 'ctrl%02d.%s' % (i, key)
                    fields.append(name)
                    fields_access[name] = {
                        'ctrl': ctrl,
                        'key': key,
                        'size': field_size
                    }

        self.fields = fields
        self.fields_access = fields_access

        # init timings logs
        self.fields_timings                         = ['timing_utils', 'timing_control', 'timing_logging']
        self.fields_access_timing                   = {}
        self.fields_access_timing['timing_utils']   = {'ctrl' : self, 'key' : 'timing_utils', 'size' : 1}
        self.fields_access_timing['timing_control'] = {'ctrl' : self, 'key' : 'timing_control', 'size' : 1}
        self.fields_access_timing['timing_logging'] = {'ctrl' : self, 'key' : 'timing_logging', 'size' : 1}
        self.fields_access_timing['absolute_time']  = {'ctrl' : self, 'key' : 'absolute_time', 'size' : 1}

    def start_streaming(self):
        if self.streaming:
            print('!!! ThreadHead: Already streaming data. Updating fields to log.')
            self.init_log_stream_fields()
            return

        self.streaming = True

        if self.ws_thread is None:
            self.ws_thread = threading.Thread(target=self.ws_thread_fn)
            self.ws_thread.start()

        # If no logging yet, then setup the fields to log.
        if not self.logging:
            self.init_log_stream_fields()

        print('!!! ThreadHead: Start streaming data.')

    def stop_streaming(self):
        if not self.streaming:
            return

        self.streaming = False
        self.ws_thread = None

        # Wait till the websocket server is shut down. Doing this
        # to avoid stopping one th and starting another one
        # having two websocket threads running at the same time.
        while self.ws_is_running:
            time.sleep(0.01)

        print('!!! ThreadHead: Stop streaming data.')

    def start_logging(self, log_duration_s=30, log_filename=None, LOG_FIELDS=['all']):
        if self.logging:
            print('ThreadHead: Already logging data.')
            return
        self.log_duration_s = log_duration_s

        # If no logging yet, then setup the fields to log.
        if not self.streaming:
            self.init_log_stream_fields(LOG_FIELDS=LOG_FIELDS)

        if not log_filename:
            log_filename = time.strftime("%Y-%m-%d_%H-%M-%S") + '.mds'

        self.data_logger = DataLogger(log_filename)
        self.log_filename = log_filename

        for name, meta in self.fields_access.items():
            meta['log_id'] = self.data_logger.add_field(name, meta['size'])

        # log timings
        self.fields_access_timing['timing_utils']['log_id']   = self.data_logger.add_field('timing_utils', self.fields_access_timing['timing_utils']['size'])
        self.fields_access_timing['timing_control']['log_id'] = self.data_logger.add_field('timing_control', self.fields_access_timing['timing_control']['size'])
        self.fields_access_timing['timing_logging']['log_id'] = self.data_logger.add_field('timing_logging', self.fields_access_timing['timing_logging']['size'])
        self.fields_access_timing['absolute_time']['log_id']  = self.data_logger.add_field('absolute_time', self.fields_access_timing['absolute_time']['size'])

        print('!!! ThreadHead: Start logging to file "%s" for %0.2f seconds.' % (
            self.data_logger.filepath, log_duration_s))
        self.logging = True

    def log_data(self):
        if not self.logging:
            return

        # Indicate that writing is happening to the file and that the file
        # should not be closed right now.
        self.log_writing = True
        dl = self.data_logger
        dl.begin_timestep()
        for name, meta in self.fields_access.items():
            dl.log(meta['log_id'], meta['ctrl'].__dict__[meta['key']])
        # add timings
        for name, meta in self.fields_access_timing.items():
            dl.log(meta['log_id'], meta['ctrl'].__dict__[meta['key']])
        dl.end_timestep()
        self.log_writing = False

        if dl.file_index * self.dt >= self.log_duration_s:
            self.stop_logging()

    def stop_logging(self):
        if not self.logging:
            return

        self.logging = False

        # If there are logs written to the file right now, wait a bit to finish
        # the current logging iteration.
        if self.log_writing:
            time.sleep(10 * self.dt)

        self.data_logger.close_file()
        abs_filepath = os.path.abspath(self.data_logger.filepath)
        print('!!! ThreadHead: Stop logging to file "%s".' % (abs_filepath))

        # Optionally generate timing plots when user presses ctrl+c key
        print('\n Press Ctrl+C to plot the timings [FIRST MAKE SURE THE ROBOT IS AT REST OR IN A SAFETY MODE] \n')
        print(' Only works if thread_head.plot_timing() is called in the main \n')
        return abs_filepath


    def plot_timing(self):
        signal.signal(signal.SIGINT, lambda sig, frame : print("\n"))
        signal.pause()
        r = DataReader(self.log_filename)
        N = r.data['absolute_time'].shape[0]
        clock_time = np.linspace(self.dt, N * self.dt, N)
        absolute_time_to_clock = r.data['absolute_time'].reshape(-1) - clock_time
        fix, axes = plt.subplots(6, sharex=True, figsize=(8, 12))
        axes[0].plot(r.data['timing_utils'] * 1000)
        axes[1].plot(r.data['timing_control'] * 1000)
        axes[2].plot(r.data['timing_logging'] * 1000)
        axes[3].plot((r.data['timing_utils'] + r.data['timing_control'] + r.data['timing_logging']) * 1000)
        axes[4].plot((r.data['absolute_time'][1:] - r.data['absolute_time'][:-1])* 1000)
        axes[5].plot(absolute_time_to_clock * 1000)
        for ax, title in zip(axes, ['Utils', 'Control', 'Logging', 'Total Computation', 'Cycle Duration', "Cumulative Delay (Absolute Time - Clock Time)"]):
            ax.grid(True)
            ax.set_title(title)
            ax.set_ylabel('Duration [ms]')
            if title != "Cumulative Delay (Absolute Time - Clock Time)":
                ax.axhline(1000*self.dt, color='black')
            else:
                ax.axhline(0., color='black')
        signal.signal(signal.SIGINT, lambda sig, frame : sys.exit(0))
        print('\n Press Ctrl+C again to close the timing plots and exit. \n')
        plt.show()
        signal.pause()

    def run_main_loop(self, sleep=False, new_controllers=None):
        self.absolute_time = time.time() - self.time_start_recording

        # Read data from the heads / shared memory.
        for head in self.heads.values():
            head.read()

        # Process the utils.
        start = time.time()
        try:
            for (name, util) in self.utils:
                util.update(self)
        except KeyboardInterrupt as exp:
            raise exp
        except:
            traceback.print_exc()
            print('!!! Error with running util "%s" -> Switching to safety controller.' % (name))
            self.switch_controllers(self.safety_controllers)

        self.timing_utils = time.time() - start

        if new_controllers:
            self.switch_controllers(new_controllers)

        # Run the active contollers.
        start = time.time()
        try:
            for ctrl in self.active_controllers:
                ctrl.run(self)
        except KeyboardInterrupt as exp:
            raise exp
        except BaseException as e:
            self.last_exception = e
            traceback.print_exc()
            print('!!! ThreadHead: Error with running controller -> Switching to safety controller.')
            self.switch_controllers(self.safety_controllers)

        self.timing_control = time.time() - start

        self.ti += 1

        # Write the computed control back to shared memory.
        for head in self.heads.values():
            head.write()

        # If an env is povided, step it.
        if self.env:
            # Step the simulation multiple times if thread_head is running
            # at a lower frequency.
            for i in range(int(self.dt/self.env.dt)):
                # Need to apply the commands at each timestep of the simulation
                # again.
                for head in self.heads.values():
                    head.sim_step()

                # Step the actual simulation.
                self.env.step(sleep=sleep)

        start = time.time()
        self.log_data()
        self.timing_logging = time.time() - start

    def stop(self):
        self.stop_logging()
        self.stop_streaming()
        self.run_loop = False

    def start(self):
        # Put the safety controller as active controller. Doing this here
        # such that a call for `th.start_streaming()`` right after
        # `th.start()` has a controller setup.
        if self.active_controllers == None:
            self.run_main_loop(new_controllers=self.safety_controllers)

        super().start()

    def run(self):
        """ Use this method to start running the main loop in a thread. """
        self.run_loop = True
        self.time_start_recording = time.time()
        next_time = 0.

        while self.run_loop:
            t = time.time() - self.time_start_recording - next_time
            if t >= 0 or hasattr(self.head, 'blocking'):
                self.run_main_loop()
                next_time += self.dt
            else:
                time.sleep(np.core.umath.maximum(-t, 0.00001))

    def run_blocking_head(self):
        """ Runs the main loop as fast as possible. Synced by the head. """
        self.run_loop = True
        self.time_start_recording = time.time()
        while self.run_loop:
            self.run_main_loop()

    def sim_run_timed(self, total_sim_time):
        self.run_loop = True
        self.time_start_recording = time.time()
        next_time = self.dt
        while self.run_loop:
            t = time.time() - self.time_start_recording - next_time
            if t >= 0:
                self.run_main_loop()
                next_time += self.dt
            else:
                time.sleep(np.core.umath.maximum(-t, 0.00001))
            if(next_time >= total_sim_time):
                self.run_loop = False

    def sim_run(self, timesteps, sleep=False):
        """ Use this method to run the setup for `timesteps` amount of timesteps. """

        if self.active_controllers == None:
            timesteps -= 1
            self.run_main_loop(sleep, new_controllers=self.safety_controllers)

        for i in range(timesteps):
            self.run_main_loop(sleep)