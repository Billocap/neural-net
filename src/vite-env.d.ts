/// <reference types="vite/client" />
type iActivation = (x: number) => number;
type iGenerator = () => iVector;
type iClassifier = (data: iVector) => iVector;

type iMatrix = number[][];
type iVector = number[];
type iGradient = [iMatrix, iVector];

interface iData {
  value: iVector;
  label: iVector;
}

interface iLayer {
  size: number[];
  weights: iMatrix;
  biases: iVector;
}

interface iModel {
  layers: iLayer[];
}
