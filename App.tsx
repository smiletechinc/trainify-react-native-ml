import React, { useEffect, useState, useRef } from "react";
import { StyleSheet, Text, View, Dimensions, Platform } from "react-native";

import { Camera } from "expo-camera";

import * as tf from "@tensorflow/tfjs";
import * as posedetection from "@tensorflow-models/pose-detection";
import * as ScreenOrientation from "expo-screen-orientation";
import {
  bundleResourceIO,
  cameraWithTensors,
} from "@tensorflow/tfjs-react-native";
import Svg, { Circle } from "react-native-svg";
import { ExpoWebGLRenderingContext } from "expo-gl";
import { CameraType } from "expo-camera/build/Camera.types";

// import DataFrame from "dataframe-js";

// tslint:disable-next-line: variable-name
const TensorCamera = cameraWithTensors(Camera);

const IS_ANDROID = Platform.OS === "android";
const IS_IOS = Platform.OS === "ios";

// Camera preview size.
//
// From experiments, to render camera feed without distortion, 16:9 ratio
// should be used fo iOS devices and 4:3 ratio should be used for android
// devices.
//
// This might not cover all cases.
const CAM_PREVIEW_WIDTH = Dimensions.get("window").width;
const CAM_PREVIEW_HEIGHT = CAM_PREVIEW_WIDTH / (IS_IOS ? 9 / 16 : 3 / 4);

// The score threshold for pose detection results.
const MIN_KEYPOINT_SCORE = 0.3;

// The size of the resized output from TensorCamera.
//
// For movenet, the size here doesn't matter too much because the model will
// preprocess the input (crop, resize, etc). For best result, use the size that
// doesn't distort the image.
const OUTPUT_TENSOR_WIDTH = 180;
const OUTPUT_TENSOR_HEIGHT = OUTPUT_TENSOR_WIDTH / (IS_IOS ? 9 / 16 : 3 / 4);

// Whether to auto-render TensorCamera preview.
const AUTO_RENDER = false;

export default function App() {
  const cameraRef = useRef(null);
  const [tfReady, setTfReady] = useState(false);
  const [model, setModel] = useState<posedetection.PoseDetector>();
  const [typeOfServeDetector, setTypeOfServeDetector] =
    useState<tf.LayersModel>();
  const [poses, setPoses] = useState<posedetection.Pose[]>();
  const [fps, setFps] = useState(0);
  const [serveType, setServeType] = useState("");
  const [orientation, setOrientation] =
    useState<ScreenOrientation.Orientation>();
  const [skipFrameCount, setSkipFrameCount] = useState(0);
  // let skipFrameCount = 0;
  const [totalServes, setTotalServes] = useState(0);
  const [cameraType, setCameraType] = useState<CameraType>(
    Camera.Constants.Type.front
  );
  // Use `useRef` so that changing it won't trigger a re-render.
  //
  // - null: unset (initial value).
  // - 0: animation frame/loop has been canceled.
  // - >0: animation frame has been scheduled.
  const rafId = useRef<number | null>(null);

  const serveTypeDetection = (poses: any) => {
    if (poses && poses.length > 0) {
      const object = poses[0];
      const keypoints = object.keypoints;

      let test_pose = new Array();

      for (let i = 0; i < keypoints.length; i++) {
        var temp = keypoints[i];
        console.log("I am here");
        test_pose.push(temp.x);
        test_pose.push(temp.y);
        test_pose.push(temp.score);
        // console.log(temp.x, temp.y, temp.score);
      }

      // console.log(test_pose);

      // var rightShoulder = keypoints.filter(function (item: any) {
      //   return item.name === "right_shoulder";
      // });
      // console.log("rightShoulder: ", rightShoulder[0].x, rightShoulder[0].y);

      // let result = typeOfServeDetector.predict(poses).data();

      let x = tf.tensor([test_pose]).reshape([-1, 99]);

      const res = typeOfServeDetector.predict(x) as tf.Tensor;

      const values = res.dataSync();
      const arr = Array.from(values);
      console.log(values);
      console.log(arr);

      const max = Math.max(...arr);
      const index = arr.indexOf(max);

      console.log("Index: ", index);

      if (index == 0) {
        setServeType("Flat");
      } else if (index == 1) {
        setServeType("Kick");
      } else if (index == 2) {
        setServeType("Slice");
      }

      console.log(serveType);
    }
  };

  const shotDetection = (poses: any) => {
    const keypoints3D = poses["keypoints3D"];
    if (poses && poses.length > 0) {
      const object = poses[0];
      const keypoints = object.keypoints;
      // console.log('keypoints: ', keypoints);

      // Right Shoulder
      var rightShoulder = keypoints.filter(function (item: any) {
        return item.name === "right_shoulder";
      });
      // console.log('rightShoulder: ', rightShoulder[0].x, rightShoulder[0].y);

      // Right Elbow
      var rightElbow = keypoints.filter(function (item: any) {
        return item.name === "right_elbow";
      });
      // console.log('rightElbow: ', rightElbow[0].x , rightElbow[0].y);

      console.log(skipFrameCount);

      if (rightShoulder[0].y > rightElbow[0].y && skipFrameCount === 0) {
        console.log("I am here");

        let skipFrameCountTemp = skipFrameCount + 1;
        setSkipFrameCount(skipFrameCountTemp);

        let totalServesTemp = totalServes + 1;
        setTotalServes(totalServesTemp);
        console.log(totalServesTemp);
      } else if (skipFrameCount > 0 && skipFrameCount < 60) {
        let skipFrameCountTemp = skipFrameCount + 1;
        setSkipFrameCount(skipFrameCountTemp);
      } else {
        setSkipFrameCount(0);
      }
    }
  };

  useEffect(() => {
    async function prepare() {
      rafId.current = null;

      // Set initial orientation.
      const curOrientation = await ScreenOrientation.getOrientationAsync();
      setOrientation(curOrientation);

      // Listens to orientation change.
      ScreenOrientation.addOrientationChangeListener((event) => {
        setOrientation(event.orientationInfo.orientation);
      });

      // Camera permission.
      await Camera.requestCameraPermissionsAsync();

      // Wait for tfjs to initialize the backend.
      await tf.ready();

      // Load movenet model.
      // https://github.com/tensorflow/tfjs-models/tree/master/pose-detection

      const model = posedetection.SupportedModels.BlazePose;
      const detectorConfig = {
        runtime: "tfjs", // or 'tfjs'
        modelType: "lite",
      };

      const detector = await posedetection.createDetector(
        model,
        detectorConfig
      );

      // const model = await posedetection.createDetector(
      //   // posedetection.SupportedModels.MoveNet,
      //   posedetection.SupportedModels.BlazePose,
      //   detectorConfig
      // );

      setModel(detector);

      console.log("Loading Type of Serve Model");

      let test_pose = [
        0.594638705, 0.369584292, 0.999754488, 0.589969754, 0.358422577,
        0.999666452, 0.588887513, 0.358062506, 0.999462545, 0.587873518,
        0.357742667, 0.999691844, 0.591217399, 0.359721988, 0.999851108,
        0.591017306, 0.360155702, 0.999814451, 0.590846598, 0.360536039,
        0.999869704, 0.581021726, 0.365049869, 0.9995116, 0.584779084,
        0.367410958, 0.999774396, 0.592306018, 0.379502505, 0.9993698,
        0.593814075, 0.380981237, 0.999513149, 0.572292209, 0.422788471,
        0.996126115, 0.584866822, 0.427995265, 0.999579489, 0.599811494,
        0.359128565, 0.246059641, 0.621611774, 0.418727636, 0.990606666,
        0.616181195, 0.287944049, 0.565571725, 0.65847981, 0.361873239,
        0.978898168, 0.61689496, 0.270983219, 0.604153454, 0.674021482,
        0.354280949, 0.970035493, 0.616020381, 0.269142509, 0.613040745,
        0.672619522, 0.345904052, 0.966704667, 0.615653872, 0.277275503,
        0.620513439, 0.668362796, 0.34725973, 0.942334592, 0.588169515,
        0.619173229, 0.999692798, 0.591820598, 0.628406227, 0.999647498,
        0.591561019, 0.730128884, 0.550906718, 0.595847547, 0.746534765,
        0.963611364, 0.570897996, 0.846207023, 0.711334169, 0.574136496,
        0.867012799, 0.979639292, 0.561005652, 0.865304351, 0.803088844,
        0.56385833, 0.886849761, 0.973807216, 0.600462675, 0.870158613,
        0.800599277, 0.607367218, 0.892697453, 0.967935383,
      ];

      const model_json = await require("./assets/model/model.json");
      const model_weight = await require("./assets/model/group1-shard.bin");

      const model_tos = await tf.loadLayersModel(
        bundleResourceIO(model_json, model_weight)
      );

      // const res = model_tos.predict(tf.randomNormal(test_pose)) as tf.Tensor;

      // let x = tf.tensor([null,test_pose]);

      // let x = tf.tensor([test_pose]).reshape([-1, 99]);

      // const res = model_tos.predict(x) as tf.Tensor;

      // const values = res.dataSync();
      // const arr = Array.from(values);
      // // console.log(values);
      // console.log(arr[0]);

      // x.shape;

      // console.log(x.shape);
      // console.log(res);

      // const tosDetector = await tf.loadLayersModel(
      //   bundleResourceIO(model_json, model_weight)
      // );

      setTypeOfServeDetector(model_tos);

      // console.log("Model Successfully loaded", tosDetector);

      // Ready!
      setTfReady(true);
    }

    prepare();
  }, []);

  useEffect(() => {
    // Called when the app is unmounted.
    return () => {
      if (rafId.current != null && rafId.current !== 0) {
        cancelAnimationFrame(rafId.current);
        rafId.current = 0;
      }
    };
  }, []);

  const handleCameraStream = async (
    images: IterableIterator<tf.Tensor3D>,
    updatePreview: () => void,
    gl: ExpoWebGLRenderingContext
  ) => {
    const loop = async () => {
      // Get the tensor and run pose detection.
      const imageTensor = images.next().value as tf.Tensor3D;

      const startTs = Date.now();
      const poses = await model!.estimatePoses(
        imageTensor,
        undefined,
        Date.now()
      );

      // shotDetection(poses);
      serveTypeDetection(poses);

      const latency = Date.now() - startTs;
      setFps(Math.floor(1000 / latency));
      setPoses(poses);
      tf.dispose([imageTensor]);

      if (rafId.current === 0) {
        return;
      }

      // Render camera preview manually when autorender=false.
      if (!AUTO_RENDER) {
        updatePreview();
        gl.endFrameEXP();
      }

      rafId.current = requestAnimationFrame(loop);
    };

    loop();
  };

  const renderPose = () => {
    if (poses != null && poses.length > 0) {
      const keypoints = poses[0].keypoints
        .filter((k) => (k.score ?? 0) > MIN_KEYPOINT_SCORE)
        .map((k) => {
          // Flip horizontally on android or when using back camera on iOS.
          const flipX =
            IS_ANDROID || cameraType === Camera.Constants.Type.front;
          const x = flipX ? getOutputTensorWidth() - k.x : k.x;
          const y = k.y;
          const cx =
            (x / getOutputTensorWidth()) *
            (isPortrait() ? CAM_PREVIEW_WIDTH : CAM_PREVIEW_HEIGHT);
          const cy =
            (y / getOutputTensorHeight()) *
            (isPortrait() ? CAM_PREVIEW_HEIGHT : CAM_PREVIEW_WIDTH);
          return (
            <Circle
              key={`skeletonkp_${k.name}`}
              cx={cx}
              cy={cy}
              r="4"
              strokeWidth="2"
              fill="#00AA00"
              stroke="white"
            />
          );
        });

      return <Svg style={styles.svg}>{keypoints}</Svg>;
    } else {
      return <View></View>;
    }
  };

  const renderFps = () => {
    return (
      <View style={styles.fpsContainer}>
        <Text>FPS: {serveType}</Text>
      </View>
    );
  };

  const renderCameraTypeSwitcher = () => {
    return (
      <View
        style={styles.cameraTypeSwitcher}
        onTouchEnd={handleSwitchCameraType}
      >
        <Text>
          Switch to{" "}
          {cameraType === Camera.Constants.Type.front ? "back" : "front"} camera
        </Text>
      </View>
    );
  };

  const handleSwitchCameraType = () => {
    if (cameraType === Camera.Constants.Type.front) {
      setCameraType(Camera.Constants.Type.front);
    } else {
      setCameraType(Camera.Constants.Type.back);
    }
  };

  const isPortrait = () => {
    return (
      orientation === ScreenOrientation.Orientation.PORTRAIT_UP ||
      orientation === ScreenOrientation.Orientation.PORTRAIT_DOWN
    );
  };

  const getOutputTensorWidth = () => {
    // On iOS landscape mode, switch width and height of the output tensor to
    // get better result. Without this, the image stored in the output tensor
    // would be stretched too much.
    //
    // Same for getOutputTensorHeight below.
    return isPortrait() || IS_ANDROID
      ? OUTPUT_TENSOR_WIDTH
      : OUTPUT_TENSOR_HEIGHT;
  };

  const getOutputTensorHeight = () => {
    return isPortrait() || IS_ANDROID
      ? OUTPUT_TENSOR_HEIGHT
      : OUTPUT_TENSOR_WIDTH;
  };

  const getTextureRotationAngleInDegrees = () => {
    // On Android, the camera texture will rotate behind the scene as the phone
    // changes orientation, so we don't need to rotate it in TensorCamera.
    if (IS_ANDROID) {
      return 0;
    }

    // For iOS, the camera texture won't rotate automatically. Calculate the
    // rotation angles here which will be passed to TensorCamera to rotate it
    // internally.
    switch (orientation) {
      // Not supported on iOS as of 11/2021, but add it here just in case.
      case ScreenOrientation.Orientation.PORTRAIT_DOWN:
        return 180;
      case ScreenOrientation.Orientation.LANDSCAPE_LEFT:
        return cameraType === Camera.Constants.Type.front ? 270 : 90;
      case ScreenOrientation.Orientation.LANDSCAPE_RIGHT:
        return cameraType === Camera.Constants.Type.front ? 90 : 270;
      default:
        return 0;
    }
  };

  if (!tfReady) {
    return (
      <View style={styles.loadingMsg}>
        <Text>Loading...</Text>
      </View>
    );
  } else {
    return (
      // Note that you don't need to specify `cameraTextureWidth` and
      // `cameraTextureHeight` prop in `TensorCamera` below.
      <View
        style={
          isPortrait() ? styles.containerPortrait : styles.containerLandscape
        }
      >
        <TensorCamera
          ref={cameraRef}
          style={styles.camera}
          autorender={AUTO_RENDER}
          type={cameraType}
          // tensor related props
          resizeWidth={getOutputTensorWidth()}
          resizeHeight={getOutputTensorHeight()}
          resizeDepth={3}
          rotation={getTextureRotationAngleInDegrees()}
          onReady={handleCameraStream}
        />
        {renderPose()}
        {renderFps()}
        {renderCameraTypeSwitcher()}
      </View>
    );
  }
}

const styles = StyleSheet.create({
  containerPortrait: {
    position: "relative",
    width: CAM_PREVIEW_WIDTH,
    height: CAM_PREVIEW_HEIGHT,
    marginTop: Dimensions.get("window").height / 2 - CAM_PREVIEW_HEIGHT / 2,
  },
  containerLandscape: {
    position: "relative",
    width: CAM_PREVIEW_HEIGHT,
    height: CAM_PREVIEW_WIDTH,
    marginLeft: Dimensions.get("window").height / 2 - CAM_PREVIEW_HEIGHT / 2,
  },
  loadingMsg: {
    position: "absolute",
    width: "100%",
    height: "100%",
    alignItems: "center",
    justifyContent: "center",
  },
  camera: {
    width: "100%",
    height: "100%",
    zIndex: 1,
  },
  svg: {
    width: "100%",
    height: "100%",
    position: "absolute",
    zIndex: 30,
  },
  fpsContainer: {
    position: "absolute",
    top: 10,
    left: 10,
    width: 80,
    alignItems: "center",
    backgroundColor: "rgba(255, 255, 255, .7)",
    borderRadius: 2,
    padding: 8,
    zIndex: 20,
  },
  cameraTypeSwitcher: {
    position: "absolute",
    top: 10,
    right: 10,
    width: 180,
    alignItems: "center",
    backgroundColor: "rgba(255, 255, 255, .7)",
    borderRadius: 2,
    padding: 8,
    zIndex: 20,
  },
});
