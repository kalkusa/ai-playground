import { useEffect, useRef } from "react";
import styles from "./FaceRecognition.module.css";
import * as faceapi from "face-api.js";

const FaceRecognition = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

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
          videoRef.current.onloadedmetadata = () => {
            if (canvasRef.current) {
              canvasRef.current.width = videoRef?.current?.videoWidth ?? 0;
              canvasRef.current.height = videoRef?.current?.videoHeight ?? 0;
            }
          };
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

  useEffect(() => {
    const handlePlay = () => {
      intervalRef.current = setInterval(async () => {
        if (videoRef.current && canvasRef.current) {
          const options = new faceapi.TinyFaceDetectorOptions();
          const detections = await faceapi
            .detectAllFaces(videoRef.current, options)
            .withAgeAndGender()
            .withFaceExpressions();
            // .withFaceLandmarks();
          const resizedDetections = faceapi.resizeResults(detections, {
            width: videoRef.current.videoWidth,
            height: videoRef.current.videoHeight,
          });
          const canvas = canvasRef.current;
          faceapi.matchDimensions(canvas, {
            width: videoRef.current.videoWidth,
            height: videoRef.current.videoHeight,
          });
          faceapi.draw.drawDetections(canvas, resizedDetections);
          faceapi.draw.drawFaceExpressions(canvas, resizedDetections);

          resizedDetections.forEach(detection => {
            const { age, gender, genderProbability } = detection;
            const text = `${parseInt(age.toString(), 10)} years old ${gender} (${Math.round(genderProbability * 100)}%)`;
            const anchor = detection.detection.box.topRight;
            new faceapi.draw.DrawTextField(
              [text], anchor
            ).draw(canvas);
          });
        }
      }, 500);
    };

    const videoElement = videoRef.current;
    videoElement?.addEventListener("play", handlePlay);

    return () => {
      videoElement?.removeEventListener("play", handlePlay);
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  return (
    <section id="face-recognition"  className={styles.container}>
      <video
        id="face-recognition-video"
        className={styles.video}
        ref={videoRef}
        autoPlay
        muted
      />
      <canvas ref={canvasRef} className={styles.overlay} />
    </section>
  );
};

export default FaceRecognition;
