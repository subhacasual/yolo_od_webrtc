import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from yolo_com_od import detect_objects


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    # print(image.shape)
    # print(type(image))
    result_frame = detect_objects(image)
    
    return av.VideoFrame.from_ndarray(result_frame, format="bgr24")


webrtc_streamer(key="sample", video_frame_callback=video_frame_callback)