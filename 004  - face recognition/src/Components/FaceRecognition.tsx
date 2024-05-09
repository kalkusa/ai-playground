import { useEffect, useRef } from "react";
import styles from "./FaceRecognition.module.css";
import * as faceapi from "face-api.js";

const FaceRecognition = () => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const loadModels = async () => {
      try {
        const MODEL_URL = process.env.PUBLIC_URL + "/models";

        await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
        await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
        await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        await faceapi.nets.mtcnn.loadFromUri(MODEL_URL);
        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
        await faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL);
        await faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL);

        console.log("Models loaded");
      } catch (e) {
        console.log(e);
      }
    };
    loadModels();
  }, []);

  useEffect(() => {
    const startVideoStreaming = async () => {
      try {
        const constraints = { audio: false, video: true };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error("Error accessing the camera:", error);
      }
    };

    startVideoStreaming();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <section id="face-recognition">
      <video
        id="face-recognition-video"
        className={styles.video}
        ref={videoRef}
        autoPlay
        muted
      />
    </section>
  );
};

export default FaceRecognition;
