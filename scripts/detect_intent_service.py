#!/usr/bin/env python

import rospy
from dialogflow_ros.srv import DialogflowService
from dialogflow_ros import DialogflowClient


class DialogflowService:
    def __init__(self):
        self._dc = DialogflowClient()
        self._service = rospy.Service('/dialogflow_client/intent_service', DialogflowService, self._service_cb)

    def _service_cb(self, req):
        if req.voice:
            df_msg = self._dc.detect_intent_stream()
        else:
            df_msg = self._dc.detect_intent_text(req.text)
        return DialogflowServiceResponse(success=True, result=df_msg)


if __name__ == '__main__':
    rospy.init_node('dialogflow_service')
    ds = DialogflowService()
    rospy.loginfo("DF_CLIENT: Dialogflow Service is running...")
    rospy.spin()
