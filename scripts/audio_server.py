#!/usr/bin/env python

import rospy
import pyaudio
import socket
import select


class AudioServer:
    def __init__(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 4096
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                      input=True, frames_per_buffer=CHUNK,
                                      stream_callback=self._callback)
        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.read_list = [self.serversocket]

        self._server_name = rospy.get_param('/dialogflow_client/server_name', '127.0.0.1')
        self._port = rospy.get_param('/dialogflow_client/port', 4444)

        rospy.loginfo("DF_CLIENT: Audio Server Started!")

    def _connect(self):
        """Create a socket, listen on the server:port and wait for a connection."""
        self.serversocket.bind((self._server_name, self._port))
        rospy.loginfo("DF_CLIENT: Waiting for connection...")
        self.serversocket.listen(1)

    def _callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback to continuously get audio data from the mic and put it in a buffer.
         :param in_data: Audio data received from mic.
         :return: A tuple with a signal to keep listening to audio input device
         :rtype: tuple(None, int)
        """
        for s in self.read_list[1:]:
            s.send(in_data)
        return None, pyaudio.paContinue

    def start(self):
        """Main function that attempts to create a socket and establish a connection with a client."""
        self._connect()
        try:
            while True:
                # select() waits until an object is readable. Here this means
                # that it will wait until there is data to be read from the socket
                readable, writable, errored = select.select(self.read_list, [], [])
                for s in readable:
                    if s is self.serversocket:
                        (clientsocket, address) = self.serversocket.accept()
                        self.read_list.append(clientsocket)
                        rospy.loginfo("DF_CLIENT: Connection from {}".format(address))
                    else:
                        data = s.recv(1024)
                        if not data:
                            self.read_list.remove(s)
        except KeyboardInterrupt as k:
            rospy.logwarn("DF_CLIENT: Caught Keyboard Interrupt: {}".format(k))
        except socket.error as e:
            rospy.logwarn("DF_CLIENT: Caught Socket Error: {}\n Restarting...".format(e))
            self._connect()

        rospy.loginfo("DF_CLIENT: Finished recording")

    def __del__(self):
        self.serversocket.close()
        # Stop Recording
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()


if __name__ == '__main__':
    rospy.init_node('dialogflow_audio_server')
    a = AudioServer()
    a.start()
