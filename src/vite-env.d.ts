/// <reference types="vite/client" />
type iActivation = (x: number) => number;
type iMatrix = number[][];
type iVector = number[];
type iGradient = [iMatrix, iVector];

interface iPoint {
  x: number;
  y: number;
  label: iVector;
}
