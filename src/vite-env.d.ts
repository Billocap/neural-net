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

type iMatrix = number[][];
type iVector = number[];
type iGradient = [iMatrix, iVector];
