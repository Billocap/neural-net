import * as math from "mathjs";

import Sketch from "../lib/Sketch";
import NeuralNet from "../lib/NeuralNet";

class DrawTraining extends Sketch {
  private _width: number;
  private _height: number;
  private dataset: iData[];
  private neuralNet: NeuralNet;

  constructor(
    width: number,
    height: number,
    dataset: iData[],
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
    document.title = Math.round(this.frameRate()).toString();

    this.background(128);

    this.strokeWeight(4);

    for (const point of this.dataset) {
      this.stroke(point.label[0] * 128, point.label[1] * 128, 0);
      this.point(point.value[0] * 400, point.value[1] * 400);
    }

    this.strokeWeight(3);

    const t = this.neuralNet.train();

    t.next();

    for (const point of this.dataset) {
      const ff = this.neuralNet.ff(point.value);

      const r = ff.next().value as iVector;

      this.stroke(r[0] * 255, r[1] * 255, 128);
      this.point(point.value[0] * 400, point.value[1] * 400);

      let dC = math.multiply(2, math.subtract(r, point.label)) as iVector;

      const grad = ff.next(dC).value as iGradient[];

      t.next(grad);
    }

    t.next();
  }
}

export default DrawTraining;
