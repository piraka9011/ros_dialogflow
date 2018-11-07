#!/usr/bin/env python

import dialogflow_v2
from dialogflow_v2.types import InputAudioConfig, QueryInput, TextInput, StreamingDetectIntentRequest
from dialogflow_v2.gapic.enums import AudioEncoding
import pyaudio
import Queue
from uuid import uuid4
# Use to convert Struct messages to JSON
from google.protobuf.json_format import MessageToJson

import rospy
from std_msgs.msg import String
from dialogflow_ros.msg import DialogflowResult, DialogflowParameter, DialogflowContext


class DialogflowClient(object):
    def __init__(self, language_code='en-US'):
        # Mic stream input setup
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        self.CHUNK = 4096
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                      rate=RATE, input=True,
                                      frames_per_buffer=self.CHUNK,
                                      stream_callback=self._get_data)
        self._buff = Queue.Queue()  # Buffer to hold audio data
        self.closed = False

        # Dialogflow params
        # Project ID: frasier-robocup-qual
        # self.project_id = rospy.get_param('/project_id', 'my-project')
        self.project_id = 'frasier-robocup-qual'
        self.session_id = str(uuid4())
        self._language_code = language_code
        # DF Audio Setup
        audio_encoding = AudioEncoding.AUDIO_ENCODING_LINEAR_16
        self._audio_config = InputAudioConfig(audio_encoding=audio_encoding,
                                              language_code=self._language_code,
                                              sample_rate_hertz=RATE)
        # Create a session
        self._session_cli = dialogflow_v2.SessionsClient()
        self._session = self._session_cli.session_path(self.project_id, self.session_id)
        rospy.logdebug("Session Path: {}".format(self._session))

        # ROS Pubs/subs
        results_topic = rospy.get_param('/results_topic', '/dialogflow_client/results')
        requests_topic = rospy.get_param('/requests_topic', '/dialogflow_client/results')
        self._results_pub = rospy.Publisher(results_topic, DialogflowResult, queue_size=10)
        self._request_sub = rospy.Subscriber(requests_topic, String, self._request_cb)
        rospy.loginfo("DF_CLIENT: Ready")

    def _request_cb(self, msg):
        df_msg = self.detect_intent_text(msg.data)
        self._results_pub.publish(df_msg)
        rospy.loginfo("DF_CLIENT: Response from Dialogflow:\n{}".format(df_msg))

    def _get_data(self, in_data, frame_count, time_info, status):
        """Daemon thread to continuously get audio data from the server and put
         it in a buffer.
        """
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def _generator(self):
        """Generator function that continuously yields audio chunks from the buffer.
        Used to stream data to the Google Speech API Asynchronously.
        """
        while not self.closed:
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
                    break

                yield StreamingDetectIntentRequest(input_audio=chunk)

    def _fill_context(self, context):
        df_context = DialogflowContext()
        df_context.name = context.name
        df_context.lifespan_count = context.lifespan_count
        df_context.parameters = [DialogflowParameter(name=name, value=value)
                                 for name, value in context.parameters.items()]
        return df_context

    def _fill_ros_msg(self, query_result):
        df_msg = DialogflowResult()
        df_msg.fulfillment_text = query_result.fulfillment_text
        df_msg.action = query_result.action
        df_msg.parameters = [DialogflowParameter(name=name, value=value)
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

    def detect_intent_text(self, text):
        """Use the Dialogflow API to detect a user's intent. Goto the Dialogflow
        console to define intents and params.
        @:param text: Google Speech API fulfillment text
        @:return query_result: Dialogflow's query_result with action parameters
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
            rospy.logdebug('Intermediate transcript: "{}".'.format(response.recognition_result.transcript))
        # The result from the last response is the final transcript along with the detected content.
        # Make sure we actually got something (This my not be necessary, need to test)
        if response is not None:
            # Get data
            final_resp = response.query_result
            df_msg = self._fill_ros_msg(final_resp)
            # Pub
            self._results_pub.publish(df_msg)
            return df_msg
        else:
            return None

    def start(self):
        """Start the dialogflow client"""
        rospy.loginfo("DF_CLIENT: Spinning...")
        rospy.spin()

    def shutdown(self):
        """Close as cleanly as possible"""
        rospy.loginfo("DF_CLIENT: Shutting down")
        self.closed = True
        self._buff.put(None)
        exit()


if __name__ == '__main__':
    rospy.init_node('dialogflow_client', log_level=rospy.DEBUG)
    df = DialogflowClient()
    df.start()
