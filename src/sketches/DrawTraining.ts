import * as math from "mathjs";

import NeuralNet from "../lib/NeuralNet";
import Sketch from "../lib/Sketch";

class DrawTraining extends Sketch {
  private _width: number;
  private _height: number;
  private dataset: Point[];
  private neuralNet: NeuralNet;

  constructor(
    width: number,
    height: number,
    dataset: Point[],
    neuralNet: NeuralNet
  ) {
    super();

    this._width = width;
    this._height = height;
    this.dataset = dataset;
    this.neuralNet = neuralNet;
  }

  setup() {
    this.createCanvas(this._width, this._height);
  }

  draw() {
    this.background(128);

    this.strokeWeight(4);

    for (const point of this.dataset) {
      this.stroke(point.label[0] > 0 ? "red" : "green");
      this.point(point.x, point.y);
    }

    this.strokeWeight(3);

    const avg: iGradient = {
      weights: math.zeros([2, 1]) as number[][],
      biases: math.zeros([1]) as number[]
    };

    for (const point of this.dataset) {
      const ff = this.neuralNet.ff(point.x / 400, point.y / 400);

      const r = ff.next().value as number[];

      this.stroke((r[0] * 0.5 + 0.5) * 255);
      this.point(point.x, point.y);

      const dC = math.multiply(2, math.subtract(r, point.label)) as number[];

      const [gW, gB] = ff.next(dC).value as [number[][], number[]];

      avg.weights = math.add(avg.weights, gW) as number[][];
      avg.biases = math.add(avg.biases, gB) as number[];
    }

    const s = 1 / this.dataset.length;

    avg.weights = math.multiply(avg.weights, s) as number[][];
    avg.biases = math.multiply(avg.biases, s) as number[];

    this.neuralNet.train(avg.weights, avg.biases);
  }
}

export default DrawTraining;
