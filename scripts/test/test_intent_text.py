#!/usr/bin/env python

import rospy
from dialogflow_ros import DialogflowClient

if __name__ == '__main__':
    rospy.init_node('test_intent_text', log_level=rospy.DEBUG)
    dc = DialogflowClient()
    dc.detect_intent_text('Hello World')
