#!/usr/bin/env python

import dialogflow
from dialogflow.types import InputAudioConfig, QueryInput, StreamingDetectIntentRequest
from dialogflow.enums import AudioEncoding
import Queue
import rospy
from std_msgs.msg import String
from google.protobuf.json_format import MessageToJson
from dialogflow_ros.msg import DialogflowResult
import pyaudio


class DialogflowClient(object):
    def __init__(self):
        # Audio stream input setup
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
        self.session_id = 'debug'
        self.session = None
        self.language_code = 'en-US'
        # Audio Setup
        audio_encoding = AudioEncoding.AUDIO_ENCODING_LINEAR_16
        sample_rate = 16000
        self.audio_config_ = InputAudioConfig(audio_encoding=audio_encoding,
                                              language_code=self.language_code,
                                              sample_rate_hertz=sample_rate)
        # Create a session
        self.session_cli_ = dialogflow.SessionsClient()
        self.session_ = self.session_cli_.session_path(self.project_id, self.session_id)

        # Audio stream setup
        self.closed = False
        self._buff = Queue.Queue()
        self.CHUNK = 4096

        # ROS Pubs/subs
        results_topic = rospy.get_param('/results_topic', '/dialogflow_results')
        text_topic = rospy.get_param('/text_topic', '/dialogflow_text')
        self.results_pub = rospy.Publisher(results_topic, DialogflowResult, queue_size=10)
        rospy.Subscriber(text_topic, String, self._mic_callback)

    def _get_data(self, in_data, frame_count, time_info, status):
        """Daemon thread to continuously get audio data from the server and put
         it in a buffer.
        """
        # Uncomment this if you want to hear the audio being replayed.
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def _generator(self):
        """Generator function that continuously yields audio chunks from the buffer.
        Used to stream data to the Google Speech API Asynchronously.
        """
        while not self.closed:
            # Check first chunk of data
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Read in a stream till the end using a non-blocking get()
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except Queue.Empty:
                    break

            yield b''.join(data)

    def detect_intent_text(self, text):
        """Use the Dialogflow API to detect a user's intent. Goto the Dialogflow
        console to define intents and params.
        @:param text: Google Speech API fulfillment text
        @:return query_result: Dialogflow's query_result with action parameters
        """
        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(self.project_id, self.session_id)
        text_input = dialogflow.types.TextInput(text=text, language_code=self.language_code)
        query_input = QueryInput(text=text_input)
        response = session_client.detect_intent(session=session, query_input=query_input)
        return response.query_result

    def detect_intent_audio(self, audio):
        # First message contains session, query_input, and params
        query_input = QueryInput(audio_config=self.audio_config_)
        init_req = StreamingDetectIntentRequest(session=self.session_,
                                                query_input=query_input,
                                                single_utterance=True)
        self.session_cli_.streaming_detect_intent(init_req)
        # Now send data stream
        requests = (StreamingDetectIntentRequest(input_audio=content) for content in self._generator())
        responses = self.session_cli_.streaming_detect_intent(requests)
        final_resp = responses[-1].query_result
        print("Detected intent: {}\n"
              "Fullfillment test{}".format(final_resp.intent.display_name, final_resp.fullfillment_text))

    def start(self):
        """Start the dialogflow client"""
        rospy.loginfo("Starting Dialogflow client")
        rospy.spin()

    def shutdown(self):
        """Close as cleanly as possible"""
        rospy.loginfo("Shutting down")
        self.closed = True
        self._buff.put(None)
        exit()

    def _mic_callback(self, msg):
        """Callback initiated whenever data received on the dialogflow/text topic
        i.e. whenever we get a proper response from the google speech client.
        Gets all the required results from dialogflow and sends it to the command
        parser.
        """
        # Send text to Dialogflow
        response = self.detect_intent_text(msg.data)
        rospy.loginfo("Got a response from Dialogflow")
        # Convert Google's Protobuf struct into JSON
        parameters = MessageToJson(response.parameters)
        # Create response
        response_msg = DialogflowResult()
        rospy.loginfo("Publishing fulfillment text: {}".format(response.fulfillment_text))
        response_msg.fulfillment_text = response.fulfillment_text
        # Check if we need to send any params
        if parameters == '{}':
            rospy.logwarn("No parameters detected")
        else:
            rospy.loginfo("Publishing parameters: {}".format(parameters))
            response_msg.parameters = parameters
        # Extract functions/actions to run
        if response.action == 'input.unknown':
            rospy.logwarn("No action associated with intent")
        else:
            rospy.loginfo("Publishing action: {}".format(response.action))
            response_msg.action = response.action
        # Publish response
        self.results_pub.publish(response_msg)


if __name__ == '__main__':
    rospy.init_node('dialogflow_client')
    df = DialogflowClient()
    df.start()
