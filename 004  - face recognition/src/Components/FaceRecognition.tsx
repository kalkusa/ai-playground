import { useEffect, useRef } from "react";
import styles from "./FaceRecognition.module.css";

const FaceRecognition = () => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const startVideoStreaming = async () => {
      try {
        const constraints = { audio: false, video: true };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        // console.log("Got MediaStream:", stream);

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
    <section id="face-recognition" >
      <video id="face-recognition-video" className={styles.video} ref={videoRef} autoPlay muted />
    </section>
  );
};

export default FaceRecognition;
