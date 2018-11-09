#!/usr/bin/env python

# Dialogflow
import dialogflow_v2
from dialogflow_v2.types import InputAudioConfig, QueryInput, TextInput, StreamingDetectIntentRequest
from dialogflow_v2.gapic.enums import AudioEncoding

# System
from errno import ECONNREFUSED
import pyaudio
import Queue
import socket
from threading import Thread
from uuid import uuid4
# Use to convert Struct messages to JSON
from google.protobuf.json_format import MessageToJson

# ROS
import rospy
from std_msgs.msg import String
from dialogflow_ros.msg import DialogflowResult, DialogflowParameter, DialogflowContext


class DialogflowClient(object):
    def __init__(self, language_code='en-US'):
        # First we want to know whether we'll get audio locally or externally.
        # Default behavior is no.
        self.USE_AUDIO_SERVER = rospy.get_param('/dialogflow_client/use_audio_server', False)
        self.DEBUG = rospy.get_param('/dialogflow_client/debug', False)
        # Mic stream input setup
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        self.CHUNK = 4096
        self._buff = Queue.Queue()  # Buffer to hold audio data
        self._closed = False
        self.audio = pyaudio.PyAudio()
        # If we are using the audio server then we need to setup our audio collection differently
        if self.USE_AUDIO_SERVER:
            # Create an audio stream
            self.stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                          rate=RATE, output=True,
                                          frames_per_buffer=self.CHUNK)
            # Audio data thread to get data from server
            self.data_thread = Thread(target=self._get_server_data)
            self.data_thread.daemon = True
            # Socket for connection
            self.s = None
            self._connected = False
            self._server_name = rospy.get_param('/dialogflow_client/server_name', '127.0.0.1')
            self._port = rospy.get_param('/dialogflow_client/port', 4444)
            # Retry 3 times to connect
            MAX_CONNECTION_RETRY = 3
            for _ in range(0, MAX_CONNECTION_RETRY):
                try:
                    self._connect()
                except socket.error as e:
                    rospy.logwarn("DF_CLIENT: Socket exception caught!\n{}\nRetrying...".format(e))
                    rospy.sleep(1)
                    continue
                break
            # Yay :)
            if self._connected:
                self.data_thread.start()
            # Nay :c
            else:
                rospy.logerr("DF_CLIENT: Unable to connect to audio server! Make sure it is running and"
                             "you are connected on the same network.")
                rospy.signal_shutdown("Unable to connect to audio server.")
                self.__del__()
        # Typical stream config uses stream_callback to get data instead
        else:
            self.stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                          rate=RATE, input=True,
                                          frames_per_buffer=self.CHUNK,
                                          stream_callback=self._get_audio_data)

        # Dialogflow params
        project_id = rospy.get_param('/dialogflow_client/project_id', 'my-project-id')
        session_id = str(uuid4())   # Random
        self._language_code = language_code
        # DF Audio Setup
        audio_encoding = AudioEncoding.AUDIO_ENCODING_LINEAR_16
        self._audio_config = InputAudioConfig(audio_encoding=audio_encoding,
                                              language_code=self._language_code,
                                              sample_rate_hertz=RATE)
        # Create a session
        self._session_cli = dialogflow_v2.SessionsClient()
        self._session = self._session_cli.session_path(project_id, session_id)
        rospy.logdebug("DF_CLIENT: Session Path: {}".format(self._session))

        # ROS Pubs/subs
        results_topic = rospy.get_param('/dialogflow_client/results_topic', '/dialogflow_client/results')
        requests_topic = rospy.get_param('/dialogflow_client/requests_topic', '/dialogflow_client/requests')
        self._results_pub = rospy.Publisher(results_topic, DialogflowResult, queue_size=10)
        self._request_sub = rospy.Subscriber(requests_topic, String, self._request_cb)
        rospy.loginfo("DF_CLIENT: Ready!")

    # ========================================= #
    #           ROS Utility Functions           #
    # ========================================= #

    def _request_cb(self, msg):
        """ROS Callback that sends text received from a topic to Dialogflow,
        :param msg: A std_msgs String message.
        """
        df_msg = self.detect_intent_text(msg.data)
        self._results_pub.publish(df_msg)
        rospy.loginfo("DF_CLIENT: Response from Dialogflow:\n{}".format(df_msg))

    # ==================================== #
    #           Utility Functions          #
    # ==================================== #

    def _connect(self):
        """Creates a socket to listen for audio data from the server"""
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self._server_name, self._port))
        rospy.loginfo("DF_CLIENT: Connected to socket")
        self._connected = True

    def _get_server_data(self):
        """Daemon thread that receives data from the audio socket and puts in a buffer.
        Works just like _get_audio_data but data comes from server, not mic.
        """
        try:
            while True:
                data = self.s.recv(self.CHUNK)
                self._buff.put(data)
                # Play audio
                if self.DEBUG:
                    self.stream.write(data)
        except KeyboardInterrupt as e:
            rospy.logwarn("DF_CLIENT: Shutdown from thread: {}".format(e))
            self.__del__()

    def _get_audio_data(self, in_data, frame_count, time_info, status):
        """PyAudio callback to continuously get audio data from the mic and put it in a buffer.
         :param in_data: Audio data received from mic.
         :return: A tuple with a signal to keep listening to audio input device
         :rtype: tuple(None, int)
        """
        self._buff.put(in_data)
        # Play audio
        if self.DEBUG:
            self.stream.write(in_data)
        return None, pyaudio.paContinue

    def _generator(self):
        """Generator function that continuously yields audio chunks from the buffer.
        Used to stream data to the Google Speech API Asynchronously.
        :return A streaming request with the audio data.
        First request carries config data per Dialogflow docs.
        :rtype: Iterator[:class:`StreamingDetectIntentRequest`]
        """
        while not self._closed:
            # First message contains session, query_input, and params
            query_input = QueryInput(audio_config=self._audio_config)
            yield StreamingDetectIntentRequest(session=self._session,
                                               query_input=query_input,
                                               single_utterance=True)
            # Read in a stream till the end using a non-blocking get()
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                except Queue.Empty:
                    rospy.logwarn_throttle(10, "DF_CLIENT: Audio queue is empty!")
                    break

                yield StreamingDetectIntentRequest(input_audio=chunk)

    def _fill_context(self, context):
        """Utility function that fills the context received from Dialogflow into the ROS msg.
        :param context: The output_context received from Dialogflow.
        :return: The ROS DialogflowContext msg.
        :rtype: DialogflowContext
        """
        df_context = DialogflowContext()
        df_context.name = context.name
        df_context.lifespan_count = context.lifespan_count
        df_context.parameters = [DialogflowParameter(name=name, value=value)
                                 for name, value in context.parameters.items()]
        return df_context

    def _fill_ros_msg(self, query_result):
        """Utility function that fills the result received from Dialogflow into the ROS msg.
        :param query_result: The query_result received from Dialogflow.
        :return: The ROS DialogflowResult msg.
        :rtype: DialogflowResult
        """
        df_msg = DialogflowResult()
        df_msg.fulfillment_text = str(query_result.fulfillment_text)
        df_msg.query_text = str(query_result.query_text)
        df_msg.action = str(query_result.action)
        df_msg.parameters = [DialogflowParameter(name=str(name), value=str(value))
                             for name, value in query_result.parameters.items()]
        df_msg.contexts = [self._fill_context(context) for context in query_result.output_contexts]
        df_msg.intent = query_result.intent
        rospy.logdebug("DF_CLIENT: Results:\n"
                       "Query Text: {}\n"
                       "Detected intent: {} (Confidence: {})\n"
                       "Fulfillment text: {}\n"
                       "Action: {}".format(query_result.query_text, query_result.intent.display_name,
                                           query_result.intent_detection_confidence, df_msg.fulfillment_text,
                                           df_msg.action))
        return df_msg

    # ======================================== #
    #           Dialogflow Functions           #
    # ======================================== #

    def detect_intent_text(self, text):
        """Use the Dialogflow API to detect a user's intent. Goto the Dialogflow
        console to define intents and params.
        :param text: Google Speech API fulfillment text
        :return query_result: Dialogflow's query_result with action parameters
        :rtype: DialogflowResult
        """
        text_input = TextInput(text=text, language_code=self._language_code)
        query_input = QueryInput(text=text_input)
        response = self._session_cli.detect_intent(session=self._session,
                                                   query_input=query_input)
        df_msg = self._fill_ros_msg(response.query_result)
        self._results_pub.publish(df_msg)
        return df_msg

    def detect_intent_stream(self):
        """Gets data from an audio generator (mic) and streams it to Dialogflow.
        We use a stream for VAD and single utterance detection."""
        # Generator yields audio chunks.
        requests = self._generator()
        responses = self._session_cli.streaming_detect_intent(requests)
        response = None
        for response in responses:
            rospy.logdebug('DF_CLIENT: Intermediate transcript: "{}".'.format(response.recognition_result.transcript))
        # The result from the last response is the final transcript along with the detected content.
        final_resp = response.query_result
        if final_resp.query_text == '':
            rospy.logwarn("DF_CLIENT: No audio received!")
        df_msg = self._fill_ros_msg(final_resp)
        # Pub
        self._results_pub.publish(df_msg)
        return df_msg

    def start(self):
        """Start the dialogflow client"""
        rospy.loginfo("DF_CLIENT: Spinning...")
        rospy.spin()

    def __del__(self):
        """Close as cleanly as possible"""
        rospy.loginfo("DF_CLIENT: Shutting down")
        self._closed = True
        self._buff.put(None)
        exit()


if __name__ == '__main__':
    rospy.init_node('dialogflow_client')
    df = DialogflowClient()
    df.start()
