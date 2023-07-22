/// <reference types="vite/client" />
interface iPoint {
  class: string;
  x: number;
  y: number;
}

interface Point {
  x: number;
  y: number;
  label: number[];
}

interface iGradient {
  weights: number[][];
  biases: number[];
}

type iMatrix = number[][];
type iVector = number[];
